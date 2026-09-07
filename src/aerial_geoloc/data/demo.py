"""Synthetic multi-view campus tiles for CPU demos (no University-1652 download).

Each location has a unique hue + footprint + roof pattern. Drone tiles are
oblique/rotated; satellite tiles are north-up; street tiles show a facade strip.
Identity is visually consistent across views, so a small encoder can retrieve.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from aerial_geoloc.data.sample import Sample

VIEWS = ("drone", "satellite", "street")
FOOTPRINTS = ("rect", "l", "u", "plus", "t", "court")
ROOFS = ("solid", "stripes", "grid", "dots", "diag")


def _hsv_rgb(h: float, s: float = 0.62, v: float = 0.78) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = int(255 * v * (1.0 - s))
    q = int(255 * v * (1.0 - f * s))
    t = int(255 * v * (1.0 - (1.0 - f) * s))
    vv = int(255 * v)
    i %= 6
    return ((vv, t, p), (q, vv, p), (p, vv, t), (p, q, vv), (t, p, vv), (vv, p, q))[i]


def _palette(location_id: int) -> dict[str, tuple[int, int, int]]:
    hue = (location_id * 0.137) % 1.0
    building = _hsv_rgb(hue, 0.58, 0.80)
    roof = _hsv_rgb((hue + 0.08) % 1.0, 0.45, 0.55)
    accent = _hsv_rgb((hue + 0.42) % 1.0, 0.7, 0.9)
    grass = (52, 110 + (location_id * 7) % 40, 58)
    road = (70, 70, 74)
    return {"building": building, "roof": roof, "accent": accent, "grass": grass, "road": road}


def _draw_footprint(draw: ImageDraw.ImageDraw, kind: str, box: tuple[int, int, int, int], fill, outline) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if kind == "rect":
        draw.rectangle(box, fill=fill, outline=outline, width=2)
    elif kind == "l":
        draw.rectangle((x0, y0, x0 + w // 3, y1), fill=fill, outline=outline, width=2)
        draw.rectangle((x0, y1 - h // 3, x1, y1), fill=fill, outline=outline, width=2)
    elif kind == "u":
        draw.rectangle((x0, y0, x0 + w // 4, y1), fill=fill, outline=outline, width=2)
        draw.rectangle((x1 - w // 4, y0, x1, y1), fill=fill, outline=outline, width=2)
        draw.rectangle((x0, y1 - h // 4, x1, y1), fill=fill, outline=outline, width=2)
    elif kind == "plus":
        draw.rectangle((x0 + w // 3, y0, x1 - w // 3, y1), fill=fill, outline=outline, width=2)
        draw.rectangle((x0, y0 + h // 3, x1, y1 - h // 3), fill=fill, outline=outline, width=2)
    elif kind == "t":
        draw.rectangle((x0, y0, x1, y0 + h // 3), fill=fill, outline=outline, width=2)
        draw.rectangle((x0 + w // 3, y0, x1 - w // 3, y1), fill=fill, outline=outline, width=2)
    else:  # courtyard
        draw.rectangle(box, fill=fill, outline=outline, width=2)
        inner = (x0 + w // 4, y0 + h // 4, x1 - w // 4, y1 - h // 4)
        draw.rectangle(inner, fill=(40, 90, 50), outline=outline, width=1)


def _draw_roof_pattern(draw: ImageDraw.ImageDraw, kind: str, box: tuple[int, int, int, int], color) -> None:
    x0, y0, x1, y1 = box
    if kind == "solid":
        return
    if kind == "stripes":
        for x in range(x0 + 4, x1 - 2, 6):
            draw.line((x, y0 + 2, x, y1 - 2), fill=color, width=2)
    elif kind == "grid":
        for x in range(x0 + 5, x1, 8):
            draw.line((x, y0 + 2, x, y1 - 2), fill=color, width=1)
        for y in range(y0 + 5, y1, 8):
            draw.line((x0 + 2, y, x1 - 2, y), fill=color, width=1)
    elif kind == "dots":
        for x in range(x0 + 6, x1 - 4, 8):
            for y in range(y0 + 6, y1 - 4, 8):
                draw.ellipse((x, y, x + 3, y + 3), fill=color)
    else:
        for i in range(-20, 40, 6):
            draw.line((x0 + i, y0, x0 + i + (y1 - y0), y1), fill=color, width=2)


def render_satellite(location_id: int, size: int = 128) -> Image.Image:
    pal = _palette(location_id)
    img = Image.new("RGB", (size, size), pal["grass"])
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, size // 2 - 5, size, size // 2 + 5), fill=pal["road"])
    draw.rectangle((size // 2 - 5, 0, size // 2 + 5, size), fill=pal["road"])
    box = (size // 4, size // 4, 3 * size // 4, 3 * size // 4)
    fp = FOOTPRINTS[location_id % len(FOOTPRINTS)]
    roof = ROOFS[location_id % len(ROOFS)]
    _draw_footprint(draw, fp, box, pal["roof"], pal["accent"])
    _draw_roof_pattern(draw, roof, box, pal["accent"])
    # north marker unique to satellite / "map" view
    draw.polygon([(size // 2, 6), (size // 2 - 5, 16), (size // 2 + 5, 16)], fill=(220, 40, 40))
    return img


def render_drone(location_id: int, view_idx: int, size: int = 128) -> Image.Image:
    sat = render_satellite(location_id, size=size)
    angle = -28 + view_idx * 11
    scale = 1.12 + 0.04 * (view_idx % 3)
    rotated = sat.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(90, 140, 200))
    new_w = int(rotated.width / scale)
    new_h = int(rotated.height / scale)
    left = max(0, (rotated.width - new_w) // 2 + (view_idx % 5) - 2)
    top = max(0, (rotated.height - new_h) // 2 + (view_idx % 3) - 1)
    crop = rotated.crop((left, top, left + new_w, top + new_h)).resize((size, size), Image.BICUBIC)
    # simulate slightly lower altitude / warmer illumination
    arr = crop.convert("RGB")
    from PIL import ImageEnhance

    arr = ImageEnhance.Brightness(arr).enhance(1.05 + 0.04 * (view_idx % 2))
    arr = ImageEnhance.Color(arr).enhance(1.1)
    overlay = Image.new("RGB", (size, size), (90, 140, 200))
    return Image.blend(arr, overlay, 0.06)


def render_street(location_id: int, view_idx: int, size: int = 128) -> Image.Image:
    pal = _palette(location_id)
    img = Image.new("RGB", (size, size), (140, 185, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, int(size * 0.62), size, size), fill=(90, 90, 94))
    facade_w = int(size * (0.45 + 0.08 * (view_idx % 3)))
    x0 = size // 2 - facade_w // 2 + (view_idx - 1) * 6
    y0 = int(size * 0.18)
    draw.rectangle((x0, y0, x0 + facade_w, int(size * 0.72)), fill=pal["building"], outline=pal["accent"], width=2)
    # windows encode location_id in a 2-row grid
    cols = 3 + (location_id % 3)
    rows = 2 + (location_id % 2)
    for r in range(rows):
        for c in range(cols):
            wx = x0 + 8 + c * ((facade_w - 16) // max(cols, 1))
            wy = y0 + 10 + r * 18
            draw.rectangle((wx, wy, wx + 8, wy + 12), fill=(220, 230, 180) if (r + c + location_id) % 2 == 0 else (40, 50, 70))
    draw.rectangle((0, int(size * 0.72), size, size), fill=(60, 60, 64))
    return img


def generate_demo_dataset(
    root: str | Path,
    train_locations: int = 16,
    test_locations: int = 8,
    drones_per_location: int = 6,
    streets_per_location: int = 2,
    image_size: int = 128,
) -> Path:
    """Write a University-1652-style folder tree of synthetic tiles."""
    root = Path(root)
    train_ids = list(range(train_locations))
    test_ids = list(range(train_locations, train_locations + test_locations))
    layout = {
        "train": {"drone": drones_per_location, "satellite": 1, "street": streets_per_location},
        "test": {
            "query_drone": drones_per_location,
            "query_satellite": 1,
            "query_street": streets_per_location,
            "gallery_drone": drones_per_location,
            "gallery_satellite": 1,
            "gallery_street": streets_per_location,
        },
    }

    def save(view: str, loc: int, idx: int, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if view.endswith("satellite") or view == "satellite":
            img = render_satellite(loc, size=image_size)
        elif view.endswith("street") or view == "street":
            img = render_street(loc, idx, size=image_size)
        else:
            img = render_drone(loc, idx, size=image_size)
        img.save(dest)

    for loc in train_ids:
        folder = f"{loc:04d}"
        for view, n in layout["train"].items():
            for i in range(n):
                save(view, loc, i, root / "train" / view / folder / f"{i:02d}.png")
    for loc in test_ids:
        folder = f"{loc:04d}"
        for view, n in layout["test"].items():
            for i in range(n):
                save(view, loc, i, root / "test" / view / folder / f"{i:02d}.png")

    meta = {
        "kind": "synthetic_university1652_style",
        "train_locations": train_ids,
        "test_locations": test_ids,
        "note": "Synthetic tiles only. Not University-1652 imagery.",
    }
    (root / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def _scan(root: Path, split_dir: str, view: str, split: str) -> list[Sample]:
    samples: list[Sample] = []
    view_root = root / split_dir / view
    if not view_root.exists():
        return samples
    for folder in sorted(p for p in view_root.iterdir() if p.is_dir()):
        location_id = int(folder.name)
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            samples.append(
                Sample(
                    path=path,
                    location_id=location_id,
                    view=view.replace("query_", "").replace("gallery_", ""),
                    split=split,
                    location_name=folder.name,
                )
            )
    return samples


def load_demo_samples(root: str | Path, split: str) -> list[Sample]:
    root = Path(root)
    if split == "train":
        samples: list[Sample] = []
        for view in ("drone", "satellite", "street"):
            samples.extend(_scan(root, "train", view, "train"))
        return samples
    if split == "query":
        samples = []
        for view in ("query_drone", "query_satellite", "query_street"):
            samples.extend(_scan(root, "test", view, "query"))
        return samples
    if split == "gallery":
        samples = []
        for view in ("gallery_drone", "gallery_satellite", "gallery_street"):
            samples.extend(_scan(root, "test", view, "gallery"))
        return samples
    raise ValueError(f"unknown split {split!r}")


def ensure_demo_dataset(root: str | Path, **kwargs) -> Path:
    root = Path(root)
    marker = root / "manifest.json"
    if marker.exists():
        return root
    generate_demo_dataset(root, **kwargs)
    return root
