"""CMC / Recall@K and mean Average Precision for location retrieval."""

from __future__ import annotations

import numpy as np


def ranks_from_scores(scores: np.ndarray, query_ids: np.ndarray, gallery_ids: np.ndarray) -> list[np.ndarray]:
    """Return, for each query, a boolean array of matches in ranked gallery order."""
    order = np.argsort(-scores, axis=1)
    ranked_ids = gallery_ids[order]
    return [(ranked_ids[i] == query_ids[i]) for i in range(len(query_ids))]


def recall_at_k(match_lists: list[np.ndarray], k: int) -> float:
    if not match_lists:
        return 0.0
    hits = sum(1 for matches in match_lists if matches[:k].any())
    return hits / len(match_lists)


def average_precision(matches: np.ndarray) -> float:
    """Standard AP over a ranked boolean relevance vector."""
    matches = np.asarray(matches, dtype=bool)
    n_gt = int(matches.sum())
    if n_gt == 0:
        return 0.0
    precisions = []
    hit = 0
    for i, is_match in enumerate(matches, start=1):
        if is_match:
            hit += 1
            precisions.append(hit / i)
    return float(np.mean(precisions)) if precisions else 0.0


def mean_average_precision(match_lists: list[np.ndarray]) -> float:
    if not match_lists:
        return 0.0
    return float(np.mean([average_precision(m) for m in match_lists]))


def evaluate_ranks(match_lists: list[np.ndarray], ks: list[int]) -> dict[str, float]:
    metrics = {f"R@{k}": recall_at_k(match_lists, k) for k in ks}
    metrics["mAP"] = mean_average_precision(match_lists)
    metrics["n_query"] = float(len(match_lists))
    return metrics
