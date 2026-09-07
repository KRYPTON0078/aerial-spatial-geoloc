"""Geometric nuisances beyond weather: occlusion, yaw, altitude."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from aerial_geoloc.data.weather import apply_weather


def apply_occlusion(image: Image.Image, intensity: float = 0.45, seed: int = 0) -> Image.Image:
    """Cover a fraction of the frame with opaque patches (trees / vehicles / self-occlusion)."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    rng = random.Random(seed + width)
    n_boxes = 1 + int(3 * intensity)
    for i in range(n_boxes):
        bw = int(width * (0.18 + 0.22 * intensity))
        bh = int(height * (0.14 + 0.2 * intensity))
        x0 = rng.randint(0, max(1, width - bw))
        y0 = rng.randint(0, max(1, height - bh))
        color = (28 + i * 12, 32, 24)
        draw.rectangle((x0, y0, x0 + bw, y0 + bh), fill=color)
    return img


def apply_yaw(image: Image.Image, degrees: float = 28.0) -> Image.Image:
    return image.convert("RGB").rotate(degrees, resample=Image.BICUBIC, fillcolor=(90, 140, 90))


def apply_altitude(image: Image.Image, zoom: float = 1.35) -> Image.Image:
    """Simulate a lower-altitude (closer) drone by center-zooming."""
    img = image.convert("RGB")
    width, height = img.size
    zoom = max(1.05, float(zoom))
    nw, nh = int(width / zoom), int(height / zoom)
    left = (width - nw) // 2
    top = (height - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((width, height), Image.BICUBIC)


CONDITION_NAMES = ("clean", "fog", "rain", "night", "snow", "occlusion", "yaw", "altitude")


def apply_condition(image: Image.Image, condition: str, intensity: float = 0.55, seed: int = 0) -> Image.Image:
    if condition in {"clean", "none", ""}:
        return image.convert("RGB")
    if condition in {"fog", "rain", "night", "snow", "glare"}:
        return apply_weather(image, condition, intensity=intensity)
    if condition == "occlusion":
        return apply_occlusion(image, intensity=intensity, seed=seed)
    if condition == "yaw":
        return apply_yaw(image, degrees=18 + 20 * intensity)
    if condition == "altitude":
        return apply_altitude(image, zoom=1.15 + 0.4 * intensity)
    raise ValueError(f"unknown condition {condition!r}")


class ConditionTransform:
    """Eval transform: optional nuisance, then tensor normalize."""

    def __init__(self, image_size: int, condition: str = "clean", intensity: float = 0.55) -> None:
        from aerial_geoloc.data.transforms import build_eval_transforms

        self.condition = condition
        self.intensity = intensity
        self._tensor = build_eval_transforms(image_size)

    def __call__(self, image: Image.Image):
        image = apply_condition(image, self.condition, intensity=self.intensity, seed=sum(ord(c) for c in self.condition))
        return self._tensor(image)
