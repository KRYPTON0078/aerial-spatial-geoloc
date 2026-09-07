"""Standalone evaluation entry."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aerial_geoloc.config import ExperimentConfig
from aerial_geoloc.data.datasets import build_retrieval_datasets
from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint
from aerial_geoloc.engine.retrieval import evaluate_model
from aerial_geoloc.engine.trainer import resolve_device
from aerial_geoloc.logging_utils import setup_logging


def run_eval(cfg: ExperimentConfig, checkpoint: str | Path) -> dict[str, float]:
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    device = resolve_device(cfg.train.device)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    query_ds, gallery_ds = build_retrieval_datasets(cfg)
    n_classes = ckpt["config"].get("model", {}).get("num_classes")
    model = model_from_checkpoint(ckpt, num_classes=n_classes).to(device)
    query_loader = DataLoader(query_ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    gallery_loader = DataLoader(
        gallery_ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )
    metrics = evaluate_model(
        model,
        query_loader,
        gallery_loader,
        query_view=cfg.data.query_view,
        gallery_view=cfg.data.gallery_view,
        device=device,
        recall_ks=cfg.eval.recall_ks,
    )
    logger.info("eval %s from %s", metrics, checkpoint)
    return metrics
