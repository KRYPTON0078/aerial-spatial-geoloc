"""Image backbones: TinyCNN (CPU demo), ResNet, Tiny ViT."""

from __future__ import annotations

import os

import torch
from torch import nn
from torchvision import models


class TinyCNN(nn.Module):
    """~0.2M-param conv encoder for CPU demos and unit tests."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.num_features = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.drop(x)


class TinyViT(nn.Module):
    """Minimal ViT (not ViT-B/16). Enough to ablate transformer vs CNN."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        dim: int = 192,
        depth: int = 4,
        heads: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.num_features = dim
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        return self.norm(x[:, 0])


def _allow_pretrained_download() -> bool:
    if os.environ.get("AERIAL_GEOL_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("CI", "").lower() in {"1", "true"}:
        return False
    return True


def _resnet(name: str, pretrained: bool) -> nn.Module:
    weights = None
    ctor = {"resnet18": models.resnet18, "resnet50": models.resnet50}[name]
    if pretrained and _allow_pretrained_download():
        weight_enum = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        }[name]
        weights = weight_enum
    net = ctor(weights=weights)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    net.num_features = feat_dim
    return net


def build_backbone(
    name: str,
    pretrained: bool = False,
    image_size: int = 64,
    dropout: float = 0.0,
) -> nn.Module:
    if name == "tiny_cnn":
        return TinyCNN(dropout=dropout)
    if name in {"resnet18", "resnet50"}:
        return _resnet(name, pretrained=pretrained)
    if name == "vit_tiny":
        patch = 8 if image_size >= 64 else 4
        if image_size % patch != 0:
            patch = 4 if image_size % 4 == 0 else 2
        return TinyViT(img_size=image_size, patch_size=patch, dropout=dropout)
    raise ValueError(f"unknown backbone {name!r}")
