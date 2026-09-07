"""Embed query + gallery and print top-k matches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from aerial_geoloc.config import ExperimentConfig
from aerial_geoloc.data.datasets import RetrievalDataset, build_retrieval_datasets
from aerial_geoloc.data.sample import Sample
from aerial_geoloc.data.transforms import build_eval_transforms
from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint
from aerial_geoloc.engine.retrieval import cosine_scores, embed_loader, format_topk, retrieve
from aerial_geoloc.engine.trainer import resolve_device
from aerial_geoloc.models.encoder import build_encoder


def _build_untrained(cfg: ExperimentConfig):
    return build_encoder(
        backbone=cfg.model.backbone,
        encoder=cfg.model.encoder,
        embed_dim=cfg.model.embed_dim,
        num_classes=cfg.model.num_classes or 16,
        pretrained=False,
        dropout=cfg.model.dropout,
        image_size=cfg.data.image_size,
    )


def run_retrieve(
    cfg: ExperimentConfig,
    checkpoint: Path | None,
    query_path: Path | None,
    gallery_dir: Path | None,
    topk: int,
    max_queries: int,
) -> list[dict]:
    device = resolve_device(cfg.train.device)
    if checkpoint:
        ckpt = load_checkpoint(checkpoint, map_location=device)
        n_classes = ckpt["config"].get("model", {}).get("num_classes")
        model = model_from_checkpoint(ckpt, num_classes=n_classes).to(device)
    else:
        model = _build_untrained(cfg).to(device)

    transform = build_eval_transforms(cfg.data.image_size)
    if query_path is not None:
        q_img = transform(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            q_feat = model.encode(q_img, view=cfg.data.query_view).cpu().numpy()
        q_ids = np.array([-1], dtype=np.int64)
        q_paths = [str(query_path)]
        if gallery_dir is None:
            _, gallery_ds = build_retrieval_datasets(cfg)
        else:
            samples = [
                Sample(path=p, location_id=-1, view=cfg.data.gallery_view, split="gallery")
                for p in sorted(Path(gallery_dir).rglob("*"))
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
            gallery_ds = RetrievalDataset(samples, view=cfg.data.gallery_view, transform=transform)
    else:
        query_ds, gallery_ds = build_retrieval_datasets(cfg)
        if max_queries > 0:
            query_ds.samples = query_ds.samples[:max_queries]
        q_loader = DataLoader(query_ds, batch_size=cfg.eval.batch_size, shuffle=False)
        q_feat, q_ids, q_paths = embed_loader(model, q_loader, cfg.data.query_view, device, desc="query")

    g_loader = DataLoader(gallery_ds, batch_size=cfg.eval.batch_size, shuffle=False)
    g_feat, g_ids, g_paths = embed_loader(model, g_loader, cfg.data.gallery_view, device, desc="gallery")
    scores = cosine_scores(q_feat, g_feat)
    idx = retrieve(q_feat, g_feat, topk=topk)
    rows = format_topk(q_paths, g_paths, q_ids, g_ids, idx, scores)
    _print_table(rows, trained=bool(checkpoint))
    return rows


def _print_table(rows: list[dict], trained: bool) -> None:
    banner = "top-k cross-view matches"
    if not trained:
        banner += " (untrained encoder — ranking is not meaningful; pass --checkpoint)"
    print(banner)
    for row in rows:
        print(f"\nquery  id={row['query_id']}  {row['query_path']}")
        for rank, hit in enumerate(row["topk"], start=1):
            mark = "OK" if hit["correct"] else "  "
            print(f"  #{rank} [{mark}] score={hit['score']:.4f}  id={hit['gallery_id']}  {hit['gallery_path']}")
