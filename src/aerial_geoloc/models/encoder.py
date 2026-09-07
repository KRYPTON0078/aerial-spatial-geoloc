"""Shared or dual-branch encoders for drone ↔ satellite retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from aerial_geoloc.models.backbones import build_backbone


def _projection(in_dim: int, embed_dim: int, dropout: float) -> nn.Sequential:
    hidden = max(embed_dim, in_dim // 2)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, embed_dim),
    )


@dataclass
class EncoderOutput:
    query_embed: torch.Tensor
    gallery_embed: torch.Tensor
    query_logits: torch.Tensor | None = None
    gallery_logits: torch.Tensor | None = None


class DualViewEncoder(nn.Module):
    """Cross-view encoder.

    * shared: one backbone + projection (University-1652 CNN baseline style).
    * dual: view-specific backbones/projections (asymmetric drone vs satellite).
    """

    def __init__(
        self,
        backbone: str = "tiny_cnn",
        encoder: str = "shared",
        embed_dim: int = 128,
        num_classes: int | None = None,
        pretrained: bool = False,
        dropout: float = 0.0,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.encoder_mode = encoder
        self.query_backbone = build_backbone(backbone, pretrained=pretrained, image_size=image_size, dropout=dropout)
        if encoder == "shared":
            self.gallery_backbone = self.query_backbone
        elif encoder == "dual":
            self.gallery_backbone = build_backbone(
                backbone, pretrained=pretrained, image_size=image_size, dropout=dropout
            )
        else:
            raise ValueError(f"encoder must be 'shared' or 'dual', got {encoder!r}")

        in_dim = self.query_backbone.num_features
        self.query_proj = _projection(in_dim, embed_dim, dropout)
        self.gallery_proj = self.query_proj if encoder == "shared" else _projection(in_dim, embed_dim, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None
        self.embed_dim = embed_dim

    def _branch(self, x: torch.Tensor, view: str) -> torch.Tensor:
        backbone = self.query_backbone if view == "query" else self.gallery_backbone
        proj = self.query_proj if view == "query" else self.gallery_proj
        feat = backbone(x)
        z = proj(feat)
        return F.normalize(z, dim=-1)

    def encode(self, x: torch.Tensor, view: str = "query") -> torch.Tensor:
        if view in {"query", "drone", "street"}:
            return self._branch(x, "query")
        if view in {"gallery", "satellite"}:
            return self._branch(x, "gallery")
        raise ValueError(f"unknown view {view!r}")

    def forward(self, query: torch.Tensor, gallery: torch.Tensor) -> EncoderOutput:
        q = self.encode(query, "query")
        g = self.encode(gallery, "gallery")
        q_logits = self.classifier(q) if self.classifier is not None else None
        g_logits = self.classifier(g) if self.classifier is not None else None
        return EncoderOutput(query_embed=q, gallery_embed=g, query_logits=q_logits, gallery_logits=g_logits)


def build_encoder(
    backbone: str,
    encoder: str,
    embed_dim: int,
    num_classes: int | None,
    pretrained: bool,
    dropout: float,
    image_size: int,
) -> DualViewEncoder:
    return DualViewEncoder(
        backbone=backbone,
        encoder=encoder,
        embed_dim=embed_dim,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        image_size=image_size,
    )
