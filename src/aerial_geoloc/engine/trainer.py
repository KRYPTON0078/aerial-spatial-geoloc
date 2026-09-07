"""Training loop with logging, AMP, and checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aerial_geoloc.config import ExperimentConfig
from aerial_geoloc.data.datasets import build_retrieval_datasets, build_train_dataset
from aerial_geoloc.engine.checkpoint import load_checkpoint, save_checkpoint
from aerial_geoloc.engine.retrieval import evaluate_model
from aerial_geoloc.logging_utils import MetricsLogger, setup_logging
from aerial_geoloc.models.encoder import build_encoder
from aerial_geoloc.models.losses import GeoLocCriterion
from aerial_geoloc.seed import seed_everything


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_one_epoch(
    model,
    criterion: GeoLocCriterion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    log_interval: int,
    logger,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "n": 0}
    part_sum: dict[str, float] = {}
    use_amp = scaler is not None
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        query = batch["query"].to(device, non_blocking=True)
        gallery = batch["gallery"].to(device, non_blocking=True)
        location_id = batch["location_id"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        device_type = device.type
        amp_enabled = scaler is not None
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            outputs = model(query, gallery)
            loss_out = criterion(outputs, location_id)
        if use_amp:
            scaler.scale(loss_out.total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_out.total.backward()
            optimizer.step()
        bs = query.size(0)
        running["loss"] += float(loss_out.total.item()) * bs
        running["n"] += bs
        for k, v in loss_out.parts.items():
            part_sum[k] = part_sum.get(k, 0.0) + v * bs
        if step % log_interval == 0:
            logger.info("step %s loss=%.4f %s", step, loss_out.total.item(), loss_out.parts)
    denom = max(running["n"], 1)
    metrics = {"loss": running["loss"] / denom}
    metrics.update({k: v / denom for k, v in part_sum.items()})
    return metrics


def run_training(cfg: ExperimentConfig) -> dict[str, float]:
    seed_everything(cfg.train.seed, deterministic=cfg.train.deterministic)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    metrics_log = MetricsLogger(output_dir / "metrics.jsonl")
    device = resolve_device(cfg.train.device)
    logger.info("device=%s experiment=%s", device, cfg.experiment_name)

    train_ds = build_train_dataset(cfg)
    if cfg.model.num_classes is None:
        cfg.model.num_classes = train_ds.num_classes
    logger.info("train pairs=%s classes=%s", len(train_ds), train_ds.num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= cfg.train.batch_size,
    )
    query_ds, gallery_ds = build_retrieval_datasets(cfg)
    query_loader = DataLoader(query_ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    gallery_loader = DataLoader(
        gallery_ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )

    model = build_encoder(
        backbone=cfg.model.backbone,
        encoder=cfg.model.encoder,
        embed_dim=cfg.model.embed_dim,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
        image_size=cfg.data.image_size,
    ).to(device)
    criterion = GeoLocCriterion(
        name=cfg.loss.name,
        temperature=cfg.loss.temperature,
        id_weight=cfg.loss.id_weight,
        infonce_weight=cfg.loss.infonce_weight,
        triplet_margin=cfg.loss.triplet_margin,
        triplet_weight=cfg.loss.triplet_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = None
    if cfg.train.amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    start_epoch = 1
    best_r1 = -1.0

    if cfg.train.resume:
        ckpt = load_checkpoint(cfg.train.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_r1 = float(ckpt.get("metrics", {}).get("R@1", -1.0))
        logger.info("resumed from %s epoch=%s", cfg.train.resume, start_epoch)

    last_eval: dict[str, float] = {}
    history: list[dict] = []
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, scaler if cfg.train.amp else None,
            cfg.train.log_interval, logger,
        )
        logger.info("epoch %s train %s", epoch, {k: round(v, 4) for k, v in train_metrics.items()})
        metrics_log.log({"split": "train", "epoch": epoch, **train_metrics})

        do_eval = epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs
        if do_eval:
            last_eval = evaluate_model(
                model,
                query_loader,
                gallery_loader,
                query_view=cfg.data.query_view,
                gallery_view=cfg.data.gallery_view,
                device=device,
                recall_ks=cfg.eval.recall_ks,
            )
            logger.info("epoch %s eval %s", epoch, {k: round(v, 4) if isinstance(v, float) else v for k, v in last_eval.items()})
            metrics_log.log({"split": "eval", "epoch": epoch, **last_eval})
            history.append({"epoch": epoch, "train": train_metrics, "eval": last_eval})
            r1 = float(last_eval.get("R@1", 0.0))
            is_best = r1 >= best_r1
            if is_best:
                best_r1 = r1
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, last_eval, cfg, best=True)
                logger.info("new best R@1=%.4f", best_r1)
        if epoch % cfg.train.save_every == 0 or epoch == cfg.train.epochs:
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, last_eval, cfg, best=False)

    (output_dir / "config_resolved.yaml").write_text(
        _dump_yaml(cfg),
        encoding="utf-8",
    )
    import json

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return last_eval


def _dump_yaml(cfg: ExperimentConfig) -> str:
    import yaml

    return yaml.safe_dump(cfg.to_dict(), sort_keys=False)
