"""University-1652 dataset adapter (drone / satellite / street).

Official paper:
  Zheng, Wei, Yang. University-1652: A Multi-view Multi-source Benchmark
  for Drone-based Geo-localization. ACM MM 2020.

This module does **not** download the imagery. University-1652 is released
for non-commercial academic research; you must obtain it from the authors'
baseline repo or Hugging Face and cite the paper. See docs/DATASETS.md.

Expected directory layout (matches layumi/University1652-Baseline)::

    <root>/
      train/
        drone/<building_id>/*.jpg
        satellite/<building_id>/*.jpg
        street/<building_id>/*.jpg
      test/
        query_drone/<building_id>/*.jpg
        query_satellite/<building_id>/*.jpg
        gallery_drone/<building_id>/*.jpg
        gallery_satellite/<building_id>/*.jpg
        gallery_street/<building_id>/*.jpg
"""

from __future__ import annotations

from pathlib import Path

from aerial_geoloc.data.sample import Sample

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DOWNLOAD_HELP = """\
University-1652 was not found at {root}.

Download (academic / non-commercial use only) and cite Zheng et al., ACM MM 2020:

  1. Official baseline + links:
     https://github.com/layumi/University1652-Baseline
  2. Hugging Face (gated, Terms of Use apply):
     https://huggingface.co/datasets/layumi/university-1652
  3. Place the extracted `University-Release` (or equivalent) tree at:
     {root}

Expected folders: train/drone, train/satellite, test/query_drone, test/gallery_satellite, ...
"""


def university1652_root_ok(root: Path) -> bool:
    train_drone = root / "train" / "drone"
    test_gallery = root / "test" / "gallery_satellite"
    return train_drone.is_dir() and test_gallery.is_dir()


def _remap_ids(folder_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(sorted(folder_names))}


def _iter_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _scan_view(root: Path, split_dir: str, view: str, split: str, id_map: dict[str, int] | None = None) -> list[Sample]:
    view_root = root / split_dir / view
    if not view_root.is_dir():
        return []
    folders = sorted(p.name for p in view_root.iterdir() if p.is_dir())
    if id_map is None:
        id_map = _remap_ids(folders)
    samples: list[Sample] = []
    canonical_view = view.replace("query_", "").replace("gallery_", "")
    for name in folders:
        if name not in id_map:
            id_map[name] = len(id_map)
        location_id = id_map[name]
        for path in _iter_images(view_root / name):
            samples.append(
                Sample(
                    path=path,
                    location_id=location_id,
                    view=canonical_view,
                    split=split,
                    location_name=name,
                )
            )
    return samples


def load_university1652_samples(root: str | Path, split: str) -> list[Sample]:
    root = Path(root)
    if not university1652_root_ok(root):
        raise FileNotFoundError(DOWNLOAD_HELP.format(root=root))

    if split == "train":
        drone = _scan_view(root, "train", "drone", "train")
        id_map = {s.location_name: s.location_id for s in drone}
        samples = list(drone)
        for view in ("satellite", "street"):
            samples.extend(_scan_view(root, "train", view, "train", id_map=id_map))
        return samples

    if split in {"query", "gallery"}:
        prefix = "query_" if split == "query" else "gallery_"
        # Build a shared name→id map from gallery satellite (standard U1652 protocol).
        gallery_sat = root / "test" / "gallery_satellite"
        names = sorted(p.name for p in gallery_sat.iterdir() if p.is_dir()) if gallery_sat.is_dir() else []
        id_map = _remap_ids(names)
        samples: list[Sample] = []
        for view in ("drone", "satellite", "street"):
            samples.extend(_scan_view(root, "test", f"{prefix}{view}", split, id_map=id_map))
        return samples

    raise ValueError(f"unknown split {split!r}")
