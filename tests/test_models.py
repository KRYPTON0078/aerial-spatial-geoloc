import torch

from aerial_geoloc.models.backbones import build_backbone
from aerial_geoloc.models.encoder import DualViewEncoder


def test_tiny_cnn_shape():
    net = build_backbone("tiny_cnn")
    x = torch.randn(2, 3, 64, 64)
    y = net(x)
    assert y.shape == (2, net.num_features)


def test_vit_tiny_shape():
    net = build_backbone("vit_tiny", image_size=32)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    assert y.shape == (2, net.num_features)


def test_shared_encoder_l2_normalized():
    model = DualViewEncoder(backbone="tiny_cnn", encoder="shared", embed_dim=32, num_classes=4, image_size=32)
    q = torch.randn(3, 3, 32, 32)
    g = torch.randn(3, 3, 32, 32)
    out = model(q, g)
    norms = out.query_embed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert out.query_logits.shape == (3, 4)


def test_dual_encoder_has_separate_params():
    shared = DualViewEncoder(backbone="tiny_cnn", encoder="shared", embed_dim=16, image_size=32)
    dual = DualViewEncoder(backbone="tiny_cnn", encoder="dual", embed_dim=16, image_size=32)
    assert sum(p.numel() for p in dual.parameters()) > sum(p.numel() for p in shared.parameters())
