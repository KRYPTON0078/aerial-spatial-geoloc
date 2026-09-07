"""Monte-Carlo dropout uncertainty for cross-view retrieval.

Inspired by Gal & Ghahramani (2016) MC-Dropout and the Ctrl-U idea of using
prediction variance as an uncertainty signal (Zhang, Gao, Zheng, ICLR 2025)
— here applied to retrieval scores, not generative rewards.

Gallery embeddings stay deterministic. Query embeddings are sampled T times
with dropout enabled; the mean is the descriptor, the feature-std is a
per-query uncertainty used to down-weight brittle matches.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from aerial_geoloc.engine.metrics import evaluate_ranks, ranks_from_scores
from aerial_geoloc.engine.retrieval import cosine_scores
from aerial_geoloc.models.encoder import DualViewEncoder


def _enable_dropout(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_encode(model: DualViewEncoder, images: torch.Tensor, view: str, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_embedding [B,D], scalar_uncertainty [B])."""
    was_training = model.training
    model.eval()
    _enable_dropout(model)
    draws = []
    for _ in range(samples):
        draws.append(model.encode(images, view=view))
    stack = torch.stack(draws, dim=0)
    mean = F.normalize(stack.mean(dim=0), dim=-1)
    # mean per-dim std → one uncertainty scalar per image
    uncertainty = stack.std(dim=0).mean(dim=-1)
    if not was_training:
        model.eval()
    else:
        model.train()
    return mean, uncertainty


@torch.no_grad()
def mc_embed_loader(
    model: DualViewEncoder,
    loader: DataLoader,
    view: str,
    device: torch.device,
    samples: int = 8,
    desc: str = "mc-embed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feats: list[np.ndarray] = []
    uncs: list[np.ndarray] = []
    ids: list[int] = []
    paths: list[str] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        mean, unc = mc_encode(model, images, view=view, samples=samples)
        feats.append(mean.cpu().numpy())
        uncs.append(unc.cpu().numpy())
        ids.extend(batch["location_id"].tolist())
        paths.extend(batch["path"])
    if not feats:
        raise RuntimeError("empty MC embedding loader")
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(uncs, axis=0),
        np.asarray(ids, dtype=np.int64),
        paths,
    )


def uncertainty_aware_scores(query: np.ndarray, gallery: np.ndarray, query_unc: np.ndarray) -> np.ndarray:
    """Ctrl-U-style: down-weight high-variance queries. Gallery stays as-is."""
    base = cosine_scores(query, gallery)
    conf = 1.0 / (1.0 + np.asarray(query_unc, dtype=np.float64).reshape(-1, 1))
    return base * conf


def evaluate_uncertainty_aware(
    model: DualViewEncoder,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    query_view: str,
    gallery_view: str,
    device: torch.device,
    recall_ks: list[int],
    mc_samples: int = 8,
) -> dict[str, float]:
    from aerial_geoloc.engine.retrieval import embed_loader

    q_feat, q_unc, q_ids, _ = mc_embed_loader(model, query_loader, query_view, device, samples=mc_samples)
    g_feat, g_ids, _ = embed_loader(model, gallery_loader, gallery_view, device, desc="gallery")
    ks = sorted(set(recall_ks + [1, 5, 10]))
    cosine = cosine_scores(q_feat, g_feat)
    ua = uncertainty_aware_scores(q_feat, g_feat, q_unc)
    out = {}
    for prefix, scores in (("cosine", cosine), ("ua", ua)):
        metrics = evaluate_ranks(ranks_from_scores(scores, q_ids, g_ids), ks)
        for k, v in metrics.items():
            out[f"{prefix}_{k}"] = v
    out["mean_query_uncertainty"] = float(np.mean(q_unc))
    return out
