"""Checkpoint save / load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from aerial_geoloc.config import ExperimentConfig
from aerial_geoloc.models.encoder import DualViewEncoder, build_encoder


def save_checkpoint(
    path: str | Path,
    model: DualViewEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: ExperimentConfig,
    best: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config.to_dict(),
            "best": best,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(ckpt: dict[str, Any], num_classes: int | None = None) -> DualViewEncoder:
    exp = ExperimentConfig.from_mapping(ckpt["config"])
    n_classes = num_classes if num_classes is not None else exp.model.num_classes
    model = build_encoder(
        backbone=exp.model.backbone,
        encoder=exp.model.encoder,
        embed_dim=exp.model.embed_dim,
        num_classes=n_classes,
        pretrained=False,
        dropout=exp.model.dropout,
        image_size=exp.data.image_size,
    )
    model.load_state_dict(ckpt["model"])
    return model
