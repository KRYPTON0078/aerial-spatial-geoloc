import numpy as np

from aerial_geoloc.engine.metrics import average_precision, evaluate_ranks, ranks_from_scores, recall_at_k


def test_perfect_retrieval_metrics():
    query_ids = np.array([0, 1, 2])
    gallery_ids = np.array([0, 1, 2])
    scores = np.eye(3, dtype=np.float32)
    matches = ranks_from_scores(scores, query_ids, gallery_ids)
    assert recall_at_k(matches, 1) == 1.0
    metrics = evaluate_ranks(matches, ks=[1, 5])
    assert metrics["R@1"] == 1.0
    assert metrics["mAP"] == 1.0


def test_worst_rank_r1_zero():
    query_ids = np.array([0, 1])
    gallery_ids = np.array([0, 1])
    scores = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    matches = ranks_from_scores(scores, query_ids, gallery_ids)
    assert recall_at_k(matches, 1) == 0.0
    assert recall_at_k(matches, 2) == 1.0


def test_ap_two_positives():
    matches = np.array([True, False, True])
    # P@1=1, P@3=2/3 → mean 5/6
    assert abs(average_precision(matches) - (1.0 + 2 / 3) / 2) < 1e-6
