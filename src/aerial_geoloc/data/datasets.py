"""PyTorch datasets for paired train batches and single-view retrieval."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from aerial_geoloc.config import DataConfig, ExperimentConfig
from aerial_geoloc.data.demo import ensure_demo_dataset, load_demo_samples
from aerial_geoloc.data.sample import Sample
from aerial_geoloc.data.transforms import build_eval_transforms, build_train_transforms
from aerial_geoloc.data.university1652 import load_university1652_samples
from aerial_geoloc.sim.world import ensure_sim_dataset, load_sim_samples


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_samples(dataset: str, root: Path, split: str) -> list[Sample]:
    if dataset == "demo":
        return load_demo_samples(root, split)
    if dataset == "sim":
        return load_sim_samples(root, split)
    if dataset == "university1652":
        return load_university1652_samples(root, split)
    raise ValueError(f"unknown dataset {dataset!r}")


class PairedCrossViewDataset(Dataset):
    """Each item is a (query-view, gallery-view) pair from the same location."""

    def __init__(
        self,
        samples: list[Sample],
        query_view: str,
        gallery_view: str,
        transform: Callable | None = None,
    ) -> None:
        self.transform = transform
        self.query_view = query_view
        self.gallery_view = gallery_view
        self.query_samples = [s for s in samples if s.view == query_view]
        gallery = [s for s in samples if s.view == gallery_view]
        self._gallery_by_id: dict[int, list[Sample]] = defaultdict(list)
        for sample in gallery:
            self._gallery_by_id[sample.location_id].append(sample)
        self.query_samples = [s for s in self.query_samples if s.location_id in self._gallery_by_id]
        if not self.query_samples:
            raise ValueError(
                f"no paired samples for {query_view!r}→{gallery_view!r} "
                f"(query={len([s for s in samples if s.view == query_view])}, "
                f"gallery={len(gallery)})"
            )
        self.num_classes = 1 + max(s.location_id for s in self.query_samples)

    def __len__(self) -> int:
        return len(self.query_samples)

    def __getitem__(self, index: int) -> dict:
        query = self.query_samples[index]
        gallery = random.choice(self._gallery_by_id[query.location_id])
        q_img = _open_rgb(query.path)
        g_img = _open_rgb(gallery.path)
        if self.transform is not None:
            q_img = self.transform(q_img)
            g_img = self.transform(g_img)
        return {
            "query": q_img,
            "gallery": g_img,
            "location_id": query.location_id,
            "query_path": str(query.path),
            "gallery_path": str(gallery.path),
        }


class RetrievalDataset(Dataset):
    """Single-view dataset used to embed queries or the gallery."""

    def __init__(self, samples: list[Sample], view: str, transform: Callable | None = None) -> None:
        self.samples = [s for s in samples if s.view == view]
        self.view = view
        self.transform = transform
        if not self.samples:
            available = sorted({s.view for s in samples})
            raise ValueError(f"no samples for view {view!r}; available={available}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = _open_rgb(sample.path)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "location_id": sample.location_id,
            "path": str(sample.path),
            "view": sample.view,
            "location_name": sample.location_name,
        }


def prepare_data_root(cfg: DataConfig) -> Path:
    root = Path(cfg.root)
    if cfg.dataset == "demo":
        ensure_demo_dataset(
            root,
            train_locations=cfg.demo_train_locations,
            test_locations=cfg.demo_test_locations,
            drones_per_location=cfg.demo_drones_per_location,
            streets_per_location=cfg.demo_streets_per_location,
            image_size=max(cfg.image_size, 64),
        )
    elif cfg.dataset == "sim":
        ensure_sim_dataset(
            root,
            train_locations=cfg.demo_train_locations,
            test_locations=cfg.demo_test_locations,
            drones_per_location=cfg.demo_drones_per_location,
            streets_per_location=cfg.demo_streets_per_location,
            image_size=max(cfg.image_size, 64),
        )
    return root


def build_train_dataset(cfg: ExperimentConfig) -> PairedCrossViewDataset:
    root = prepare_data_root(cfg.data)
    samples = load_samples(cfg.data.dataset, root, "train")
    transform = build_train_transforms(
        image_size=cfg.data.image_size,
        weather_aug=cfg.data.weather_aug,
        weather_types=cfg.data.weather_types,
        weather_prob=cfg.data.weather_prob,
    )
    return PairedCrossViewDataset(
        samples,
        query_view=cfg.data.query_view,
        gallery_view=cfg.data.gallery_view,
        transform=transform,
    )


def build_retrieval_datasets(cfg: ExperimentConfig) -> tuple[RetrievalDataset, RetrievalDataset]:
    root = prepare_data_root(cfg.data)
    query_samples = load_samples(cfg.data.dataset, root, "query")
    gallery_samples = load_samples(cfg.data.dataset, root, "gallery")
    transform = build_eval_transforms(cfg.data.image_size)
    query_ds = RetrievalDataset(query_samples, view=cfg.data.query_view, transform=transform)
    gallery_ds = RetrievalDataset(gallery_samples, view=cfg.data.gallery_view, transform=transform)
    return query_ds, gallery_ds
