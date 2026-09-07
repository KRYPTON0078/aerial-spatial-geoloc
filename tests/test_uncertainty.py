import torch

from aerial_geoloc.engine.uncertainty import mc_encode, uncertainty_aware_scores
from aerial_geoloc.models.encoder import DualViewEncoder
from aerial_geoloc.seed import seed_everything


def test_mc_dropout_variance_positive_with_dropout():
    seed_everything(0)
    model = DualViewEncoder(backbone="tiny_cnn", encoder="shared", embed_dim=32, dropout=0.5, image_size=32)
    x = torch.randn(4, 3, 32, 32)
    mean, unc = mc_encode(model, x, "query", samples=6)
    assert mean.shape == (4, 32)
    assert unc.shape == (4,)
    assert float(unc.mean()) > 0.0


def test_mc_dropout_near_zero_without_dropout():
    seed_everything(0)
    model = DualViewEncoder(backbone="tiny_cnn", encoder="shared", embed_dim=32, dropout=0.0, image_size=32)
    x = torch.randn(4, 3, 32, 32)
    _, unc = mc_encode(model, x, "query", samples=4)
    assert float(unc.max()) < 1e-6


def test_uncertainty_aware_downweights_uncertain_queries():
    import numpy as np

    q = np.eye(2, dtype=np.float32)
    g = np.eye(2, dtype=np.float32)
    unc = np.array([0.0, 10.0], dtype=np.float32)
    scores = uncertainty_aware_scores(q, g, unc)
    assert scores[0, 0] > scores[1, 1]
