"""Embed queries/gallery and rank by cosine similarity."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aerial_geoloc.engine.metrics import evaluate_ranks, ranks_from_scores
from aerial_geoloc.models.encoder import DualViewEncoder


@torch.no_grad()
def embed_loader(
    model: DualViewEncoder,
    loader: DataLoader,
    view: str,
    device: torch.device,
    desc: str = "embed",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    feats: list[np.ndarray] = []
    ids: list[int] = []
    paths: list[str] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        z = model.encode(images, view=view)
        feats.append(z.cpu().numpy())
        ids.extend(batch["location_id"].tolist())
        paths.extend(batch["path"])
    if not feats:
        raise RuntimeError("empty embedding loader")
    return np.concatenate(feats, axis=0), np.asarray(ids, dtype=np.int64), paths


def cosine_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-8, None)
    g = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8, None)
    return q @ g.T


def retrieve(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    topk: int = 5,
) -> np.ndarray:
    scores = cosine_scores(query_feats, gallery_feats)
    return np.argsort(-scores, axis=1)[:, :topk]


def format_topk(
    query_paths: Iterable[str],
    gallery_paths: list[str],
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
) -> list[dict]:
    rows = []
    for q_i, q_path in enumerate(query_paths):
        matches = []
        for g_i in indices[q_i]:
            matches.append(
                {
                    "gallery_path": gallery_paths[int(g_i)],
                    "gallery_id": int(gallery_ids[int(g_i)]),
                    "score": float(scores[q_i, int(g_i)]),
                    "correct": int(gallery_ids[int(g_i)]) == int(query_ids[q_i]),
                }
            )
        rows.append({"query_path": q_path, "query_id": int(query_ids[q_i]), "topk": matches})
    return rows


def evaluate_model(
    model: DualViewEncoder,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    query_view: str,
    gallery_view: str,
    device: torch.device,
    recall_ks: list[int],
) -> dict[str, float]:
    q_feat, q_ids, _ = embed_loader(model, query_loader, query_view, device, desc="query")
    g_feat, g_ids, _ = embed_loader(model, gallery_loader, gallery_view, device, desc="gallery")
    scores = cosine_scores(q_feat, g_feat)
    match_lists = ranks_from_scores(scores, q_ids, g_ids)
    ks = sorted(set(recall_ks + [1, 5, 10]))
    return evaluate_ranks(match_lists, ks)
