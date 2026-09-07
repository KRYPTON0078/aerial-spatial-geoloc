"""Dataset adapters for cross-view aerial geo-localization."""

from aerial_geoloc.data.datasets import (
    PairedCrossViewDataset,
    RetrievalDataset,
    build_retrieval_datasets,
    build_train_dataset,
    load_samples,
    prepare_data_root,
)
from aerial_geoloc.data.demo import ensure_demo_dataset, generate_demo_dataset
from aerial_geoloc.data.sample import Sample
from aerial_geoloc.data.transforms import build_eval_transforms, build_train_transforms
from aerial_geoloc.data.university1652 import DOWNLOAD_HELP, load_university1652_samples
from aerial_geoloc.data.weather import RandomWeather, apply_weather

__all__ = [
    "DOWNLOAD_HELP",
    "PairedCrossViewDataset",
    "RandomWeather",
    "RetrievalDataset",
    "Sample",
    "apply_weather",
    "build_eval_transforms",
    "build_retrieval_datasets",
    "build_train_dataset",
    "build_train_transforms",
    "ensure_demo_dataset",
    "generate_demo_dataset",
    "load_samples",
    "load_university1652_samples",
    "prepare_data_root",
]
