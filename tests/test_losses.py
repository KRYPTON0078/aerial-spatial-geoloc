import torch

from aerial_geoloc.models.losses import GeoLocCriterion, InfoNCELoss
from aerial_geoloc.models.encoder import EncoderOutput


def test_infonce_lower_for_aligned_pairs():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss_fn = InfoNCELoss(temperature=0.07)
    aligned = loss_fn(z, z)
    shuffled = loss_fn(z, z[torch.randperm(8)])
    assert aligned.item() < shuffled.item()


def test_id_loss_needs_classifier():
    out = EncoderOutput(query_embed=torch.randn(2, 4), gallery_embed=torch.randn(2, 4))
    crit = GeoLocCriterion(name="id")
    try:
        crit(out, torch.tensor([0, 1]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_combined_loss_backward():
    torch.manual_seed(1)
    q = torch.nn.functional.normalize(torch.randn(4, 8, requires_grad=True), dim=-1)
    g = torch.nn.functional.normalize(torch.randn(4, 8, requires_grad=True), dim=-1)
    q_logits = torch.randn(4, 5, requires_grad=True)
    g_logits = torch.randn(4, 5, requires_grad=True)
    out = EncoderOutput(query_embed=q, gallery_embed=g, query_logits=q_logits, gallery_logits=g_logits)
    crit = GeoLocCriterion(name="infonce_id", id_weight=0.5)
    result = crit(out, torch.tensor([0, 1, 2, 3]))
    result.total.backward()
    assert "infonce" in result.parts and "id" in result.parts
