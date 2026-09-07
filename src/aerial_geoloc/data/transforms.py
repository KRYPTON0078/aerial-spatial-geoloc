"""Train / eval image transforms."""

from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms as T

from aerial_geoloc.data.weather import RandomWeather


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(
    image_size: int,
    weather_aug: bool = False,
    weather_types: Sequence[str] | None = None,
    weather_prob: float = 0.5,
) -> T.Compose:
    ops: list = [
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        T.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    ]
    if weather_aug:
        ops.append(RandomWeather(types=weather_types, p=weather_prob))
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return T.Compose(ops)


def build_eval_transforms(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def denormalize_tensor(tensor):
    """Undo ImageNet normalization for visualization. tensor: (C,H,W) or (B,C,H,W)."""
    import torch

    mean = tensor.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    if tensor.ndim == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
