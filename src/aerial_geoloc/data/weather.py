"""Synthetic weather overlays inspired by WeatherPrompt-style robustness eval.

These are classical image corruptions (fog, rain, night, snow), not the
WeatherPrompt text-gating model. Use them as a train-time ablation or a
held-out stress test. See Wen, Yu, Zheng, NeurIPS 2025.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


WEATHER_TYPES = ("fog", "rain", "night", "snow", "glare")


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def apply_fog(image: Image.Image, intensity: float = 0.45) -> Image.Image:
    image = _to_rgb(image)
    haze = Image.new("RGB", image.size, (210, 215, 220))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.5 + 3.0 * intensity))
    return Image.blend(blurred, haze, intensity)


def apply_rain(image: Image.Image, intensity: float = 0.5) -> Image.Image:
    image = _to_rgb(image)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    n_streaks = int(80 * intensity * (width * height) / (64 * 64))
    n_streaks = max(12, min(n_streaks, 400))
    rng = random.Random(int(intensity * 1000) + width)
    for _ in range(n_streaks):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        length = rng.randint(6, 18)
        draw.line((x, y, x + 2, y + length), fill=(200, 210, 230, 140), width=1)
    wet = Image.blend(image, image.filter(ImageFilter.BLUR), 0.15 * intensity)
    return Image.alpha_composite(wet.convert("RGBA"), overlay).convert("RGB")


def apply_night(image: Image.Image, intensity: float = 0.55) -> Image.Image:
    image = _to_rgb(image)
    darkened = ImageEnhance.Brightness(image).enhance(1.0 - 0.65 * intensity)
    arr = np.asarray(darkened).astype(np.float32)
    arr[..., 0] *= 0.75
    arr[..., 1] *= 0.80
    arr[..., 2] = np.clip(arr[..., 2] * 1.15, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def apply_snow(image: Image.Image, intensity: float = 0.45) -> Image.Image:
    image = _to_rgb(image)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    n_flakes = int(60 * intensity * (width * height) / (64 * 64))
    n_flakes = max(20, min(n_flakes, 500))
    rng = random.Random(width * 17 + int(intensity * 50))
    for _ in range(n_flakes):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        r = rng.randint(1, 2)
        draw.ellipse((x, y, x + r, y + r), fill=(255, 255, 255, 200))
    cooled = ImageEnhance.Color(image).enhance(0.7)
    return Image.alpha_composite(cooled.convert("RGBA"), overlay).convert("RGB")


def apply_glare(image: Image.Image, intensity: float = 0.4) -> Image.Image:
    image = _to_rgb(image)
    width, height = image.size
    overlay = Image.new("RGB", image.size, (255, 255, 240))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = int(width * 0.7), int(height * 0.25)
    radius = int(min(width, height) * (0.25 + 0.2 * intensity))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=int(180 * intensity))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=radius * 0.4))
    return Image.composite(overlay, image, mask)


_DISPATCH = {
    "fog": apply_fog,
    "rain": apply_rain,
    "night": apply_night,
    "snow": apply_snow,
    "glare": apply_glare,
}


def apply_weather(image: Image.Image, weather: str, intensity: float | None = None) -> Image.Image:
    if weather not in _DISPATCH:
        raise ValueError(f"unknown weather '{weather}', expected one of {sorted(_DISPATCH)}")
    if intensity is None:
        intensity = random.uniform(0.25, 0.7)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    return _DISPATCH[weather](image, intensity)


class RandomWeather:
    """Stochastic weather corruption used as a torchvision-style transform."""

    def __init__(self, types: Sequence[str] | None = None, p: float = 0.5) -> None:
        self.types = tuple(types) if types else WEATHER_TYPES
        unknown = set(self.types) - set(_DISPATCH)
        if unknown:
            raise ValueError(f"unknown weather types: {sorted(unknown)}")
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        weather = random.choice(self.types)
        return apply_weather(image, weather)
