"""Retrieval / geo-localization losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from aerial_geoloc.models.encoder import EncoderOutput


@dataclass
class LossOutput:
    total: torch.Tensor
    parts: dict[str, float]


class InfoNCELoss(nn.Module):
    """Symmetric in-batch InfoNCE (CLIP-style drone ↔ satellite)."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
        logits = query @ gallery.t() / self.temperature
        labels = torch.arange(query.size(0), device=query.device)
        loss_q = F.cross_entropy(logits, labels)
        loss_g = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_q + loss_g)


class TripletLoss(nn.Module):
    """Hard-negative triplet on cosine distance within the batch."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, query: torch.Tensor, gallery: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim = query @ gallery.t()
        dist = 1.0 - sim
        pos = dist[torch.arange(query.size(0), device=query.device), torch.arange(query.size(0), device=query.device)]
        # hardest negative: smallest distance among different ids
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        inf = torch.full_like(dist, 1e4)
        neg_dist = torch.where(same, inf, dist)
        hard_neg, _ = neg_dist.min(dim=1)
        return F.relu(pos - hard_neg + self.margin).mean()


class GeoLocCriterion(nn.Module):
    def __init__(
        self,
        name: str = "infonce_id",
        temperature: float = 0.07,
        id_weight: float = 1.0,
        infonce_weight: float = 1.0,
        triplet_margin: float = 0.3,
        triplet_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.id_weight = id_weight
        self.infonce_weight = infonce_weight
        self.triplet_weight = triplet_weight
        self.infonce = InfoNCELoss(temperature)
        self.triplet = TripletLoss(triplet_margin)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: EncoderOutput, location_id: torch.Tensor) -> LossOutput:
        parts: dict[str, torch.Tensor] = {}
        if self.name in {"infonce", "infonce_id"}:
            parts["infonce"] = self.infonce_weight * self.infonce(outputs.query_embed, outputs.gallery_embed)
        if self.name in {"id", "infonce_id"}:
            if outputs.query_logits is None or outputs.gallery_logits is None:
                raise ValueError("ID loss requires a classifier head (num_classes > 0)")
            id_loss = 0.5 * (
                self.ce(outputs.query_logits, location_id) + self.ce(outputs.gallery_logits, location_id)
            )
            parts["id"] = self.id_weight * id_loss
        if self.name == "triplet":
            parts["triplet"] = self.triplet_weight * self.triplet(
                outputs.query_embed, outputs.gallery_embed, location_id
            )
        if not parts:
            raise ValueError(f"unknown loss {self.name!r}")
        total = sum(parts.values())
        return LossOutput(total=total, parts={k: float(v.detach().item()) for k, v in parts.items()})
