"""University-1652 dataset adapter (drone / satellite / street).

Official paper:
  Zheng, Wei, Yang. University-1652: A Multi-view Multi-source Benchmark
  for Drone-based Geo-localization. ACM MM 2020.

This module does **not** download imagery and does **not** invent metrics.
University-1652 is gated academic / non-commercial data. Obtain it yourself
(see docs/DATASETS.md and scripts/prepare_u1652.md), then point ``data.root``
at the extracted tree.

Folder names follow layumi/University1652-Baseline (``University-1652/``)::

    <root>/
      train/
        drone/<building_id>/*.jpg
        satellite/<building_id>/*.jpg
        street/<building_id>/*.jpg
        google/<building_id>/*.jpg          # optional noisy web street
      test/
        query_drone/<building_id>/*.jpg
        query_satellite/<building_id>/*.jpg
        query_street/<building_id>/*.jpg
        gallery_drone/<building_id>/*.jpg
        gallery_satellite/<building_id>/*.jpg
        gallery_street/<building_id>/*.jpg
        4K_drone/                           # optional; not used by default splits
"""

from __future__ import annotations

from pathlib import Path

from aerial_geoloc.data.sample import Sample

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Nested names used by the official zip / Hugging Face snapshot.
NESTED_ROOT_NAMES = (
    "University-1652",
    "University-Release",
    "university-1652",
    "University1652",
)

# Protocol folders from the University1652-Baseline README (train + drone→sat test).
REQUIRED_RELATIVE = (
    Path("train") / "drone",
    Path("train") / "satellite",
    Path("test") / "query_drone",
    Path("test") / "gallery_satellite",
)

OPTIONAL_RELATIVE = (
    Path("train") / "street",
    Path("train") / "google",
    Path("test") / "query_satellite",
    Path("test") / "query_street",
    Path("test") / "gallery_drone",
    Path("test") / "gallery_street",
    Path("test") / "4K_drone",
)

TRAIN_VIEWS = ("drone", "satellite", "street", "google")
TEST_VIEWS = ("drone", "satellite", "street")

REQUEST_MD = "https://github.com/layumi/University1652-Baseline/blob/master/Request.md"
BASELINE_REPO = "https://github.com/layumi/University1652-Baseline"
HF_DATASET = "https://huggingface.co/datasets/layumi/university-1652"

DOWNLOAD_HELP = """\
University-1652 was not found at {root}.

This toolkit does not download the dataset. It is gated academic /
non-commercial imagery (Zheng, Wei, Yang, ACM MM 2020). Do not commit it.

Obtain access, then place the extracted tree so it contains train/drone and
test/gallery_satellite (often named University-1652 or University-Release).

  1. Email request (academic address) using Request.md:
     {request_md}
     Send to zdzheng12@gmail.com with title: Request of University1652 Dataset
  2. Hugging Face (gated; request access, then snapshot_download):
     {hf}
  3. Baseline repo (layout + evaluation protocol):
     {baseline}

Missing required folders:
{missing}

Optional Baseline folders not present (ok): {optional_missing}

Checklist: scripts/prepare_u1652.md
"""


def _is_protocol_root(root: Path) -> bool:
    return all((root / rel).is_dir() for rel in REQUIRED_RELATIVE)


def candidate_roots(root: Path) -> list[Path]:
    """Possible locations if the user pointed at a parent or nested folder."""
    root = root.expanduser()
    seen: list[Path] = []
    for cand in (root, *[(root / name) for name in NESTED_ROOT_NAMES]):
        if cand not in seen:
            seen.append(cand)
    return seen


def resolve_university1652_root(root: str | Path) -> Path:
    """Return the directory that actually contains ``train/`` and ``test/``.

    Raises FileNotFoundError with request instructions when the tree is absent.
    Never downloads.
    """
    raw = Path(root).expanduser()
    tried = candidate_roots(raw)
    for cand in tried:
        if _is_protocol_root(cand):
            return cand
    missing_lines = []
    for cand in tried:
        missing = [str(rel) for rel in REQUIRED_RELATIVE if not (cand / rel).is_dir()]
        missing_lines.append(f"  - {cand}: missing {', '.join(missing)}")
    opt_missing = [str(rel) for rel in OPTIONAL_RELATIVE if not (raw / rel).is_dir()]
    raise FileNotFoundError(
        DOWNLOAD_HELP.format(
            root=raw,
            request_md=REQUEST_MD,
            hf=HF_DATASET,
            baseline=BASELINE_REPO,
            missing="\n".join(missing_lines),
            optional_missing=", ".join(opt_missing) if opt_missing else "(none checked; root missing)",
        )
    )


def university1652_root_ok(root: Path) -> bool:
    try:
        resolve_university1652_root(root)
    except FileNotFoundError:
        return False
    return True


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
    root = resolve_university1652_root(root)

    if split == "train":
        drone = _scan_view(root, "train", "drone", "train")
        id_map = {s.location_name: s.location_id for s in drone}
        samples = list(drone)
        for view in TRAIN_VIEWS:
            if view == "drone":
                continue
            samples.extend(_scan_view(root, "train", view, "train", id_map=id_map))
        return samples

    if split in {"query", "gallery"}:
        prefix = "query_" if split == "query" else "gallery_"
        # Shared name→id map from gallery satellite (University-1652 drone→sat protocol).
        gallery_sat = root / "test" / "gallery_satellite"
        names = sorted(p.name for p in gallery_sat.iterdir() if p.is_dir()) if gallery_sat.is_dir() else []
        id_map = _remap_ids(names)
        samples: list[Sample] = []
        for view in TEST_VIEWS:
            samples.extend(_scan_view(root, "test", f"{prefix}{view}", split, id_map=id_map))
        return samples

    raise ValueError(f"unknown split {split!r}; expected train, query, or gallery")
