"""Simulated campus with known location IDs and camera pose.

Each building sits on a metric grid. Satellite tiles are nadir / north-up.
Drone tiles are rendered at a recorded altitude (m) and yaw (deg). Weather
and geometric nuisances are applied as labeled corruptions, not as mystery
noise — so robustness plots have a ground-truth cause.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from aerial_geoloc.data.demo import load_demo_samples, render_drone, render_satellite, render_street
from aerial_geoloc.data.sample import Sample


@dataclass
class Pose:
    x_m: float
    y_m: float
    altitude_m: float
    yaw_deg: float
    pitch_deg: float = 0.0


@dataclass
class FrameRecord:
    path: str
    location_id: int
    location_name: str
    view: str
    split: str
    pose: dict
    weather: str = "clean"


def location_pose(location_id: int) -> tuple[float, float]:
    """Place buildings on a 20 m grid."""
    col, row = location_id % 8, location_id // 8
    return float(col * 20.0), float(row * 20.0)


def drone_pose(location_id: int, view_idx: int) -> Pose:
    x, y = location_pose(location_id)
    return Pose(
        x_m=x,
        y_m=y,
        altitude_m=70.0 + 12.0 * (view_idx % 4),
        yaw_deg=-28.0 + 11.0 * view_idx,
        pitch_deg=-18.0,
    )


def satellite_pose(location_id: int) -> Pose:
    x, y = location_pose(location_id)
    return Pose(x_m=x, y_m=y, altitude_m=500.0, yaw_deg=0.0, pitch_deg=-90.0)


def generate_sim_dataset(
    root: str | Path,
    train_locations: int = 12,
    test_locations: int = 8,
    drones_per_location: int = 4,
    streets_per_location: int = 1,
    image_size: int = 64,
    seed: int = 42,
) -> Path:
    """Write a University-1652-style tree plus pose metadata."""
    del seed
    root = Path(root)
    train_ids = list(range(train_locations))
    test_ids = list(range(train_locations, train_locations + test_locations))
    records: list[FrameRecord] = []

    def save(loc: int, idx: int, dest: Path, split: str, canonical_view: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if canonical_view == "satellite":
            img = render_satellite(loc, size=image_size)
            pose = satellite_pose(loc)
        elif canonical_view == "street":
            img = render_street(loc, idx, size=image_size)
            x, y = location_pose(loc)
            pose = Pose(x_m=x, y_m=y, altitude_m=1.7, yaw_deg=float(idx * 40), pitch_deg=0.0)
        else:
            img = render_drone(loc, idx, size=image_size)
            pose = drone_pose(loc, idx)
        img.save(dest)
        records.append(
            FrameRecord(
                path=str(dest.relative_to(root)),
                location_id=loc,
                location_name=f"{loc:04d}",
                view=canonical_view,
                split=split,
                pose=asdict(pose),
            )
        )

    for loc in train_ids:
        folder = f"{loc:04d}"
        for i in range(drones_per_location):
            save(loc, i, root / "train" / "drone" / folder / f"{i:02d}.png", "train", "drone")
        save(loc, 0, root / "train" / "satellite" / folder / "00.png", "train", "satellite")
        for i in range(streets_per_location):
            save(loc, i, root / "train" / "street" / folder / f"{i:02d}.png", "train", "street")

    for loc in test_ids:
        folder = f"{loc:04d}"
        for i in range(drones_per_location):
            save(loc, i, root / "test" / "query_drone" / folder / f"{i:02d}.png", "query", "drone")
            save(loc, i, root / "test" / "gallery_drone" / folder / f"{i:02d}.png", "gallery", "drone")
        save(loc, 0, root / "test" / "query_satellite" / folder / "00.png", "query", "satellite")
        save(loc, 0, root / "test" / "gallery_satellite" / folder / "00.png", "gallery", "satellite")
        for i in range(streets_per_location):
            save(loc, i, root / "test" / "query_street" / folder / f"{i:02d}.png", "query", "street")
            save(loc, i, root / "test" / "gallery_street" / folder / f"{i:02d}.png", "gallery", "street")

    manifest = {
        "kind": "simulated_campus",
        "note": "Procedural tiles with known pose. Not University-1652 imagery.",
        "image_size": image_size,
        "train_locations": train_ids,
        "test_locations": test_ids,
        "drones_per_location": drones_per_location,
        "grid_spacing_m": 20.0,
        "frames": [asdict(r) for r in records],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def load_sim_samples(root: str | Path, split: str) -> list[Sample]:
    return load_demo_samples(root, split)


def ensure_sim_dataset(root: str | Path, **kwargs) -> Path:
    root = Path(root)
    if (root / "manifest.json").exists():
        return root
    return generate_sim_dataset(root, **kwargs)


def load_manifest(root: str | Path) -> dict:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"sim manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
