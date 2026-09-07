"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

BackboneName = Literal["tiny_cnn", "resnet18", "resnet50", "vit_tiny"]
LossName = Literal["infonce", "id", "infonce_id", "triplet"]
EncoderMode = Literal["shared", "dual"]
DatasetName = Literal["demo", "university1652"]
ViewName = Literal["drone", "satellite", "street"]


def _unwrap_optional(cls: Any) -> Any:
    origin = get_origin(cls)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(cls) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return cls


def _coerce(cls: Any, value: Any) -> Any:
    if value is None:
        return None
    cls = _unwrap_optional(cls)
    if is_dataclass(cls) and isinstance(value, dict):
        return from_dict(cls, value)
    origin = get_origin(cls)
    if origin is list:
        inner = get_args(cls)[0] if get_args(cls) else Any
        return [_coerce(inner, v) for v in value]
    if origin is Literal:
        allowed = get_args(cls)
        if value not in allowed:
            raise ValueError(f"expected one of {allowed}, got {value!r}")
        return value
    if cls is Path:
        return Path(value)
    if cls in (int, float, str, bool):
        if cls is bool and isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return cls(value)
    return value


def from_dict(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    valid = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    unknown = set(data) - set(valid)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name, f in valid.items():
        if name not in data:
            continue
        hint = hints.get(name, f.type)
        kwargs[name] = _coerce(hint, data[name])
    return cls(**kwargs)


@dataclass
class ModelConfig:
    backbone: BackboneName = "tiny_cnn"
    encoder: EncoderMode = "shared"
    embed_dim: int = 256
    pretrained: bool = False
    dropout: float = 0.0
    num_classes: int | None = None


@dataclass
class LossConfig:
    name: LossName = "infonce_id"
    temperature: float = 0.07
    id_weight: float = 1.0
    infonce_weight: float = 1.0
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0


@dataclass
class DataConfig:
    dataset: DatasetName = "demo"
    root: Path = Path("data/demo")
    image_size: int = 128
    query_view: ViewName = "drone"
    gallery_view: ViewName = "satellite"
    weather_aug: bool = False
    weather_types: list[str] = field(default_factory=lambda: ["fog", "rain", "night", "snow"])
    weather_prob: float = 0.5
    demo_train_locations: int = 16
    demo_test_locations: int = 8
    demo_drones_per_location: int = 6
    demo_streets_per_location: int = 2


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 16
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    output_dir: Path = Path("outputs/demo")
    log_interval: int = 10
    eval_every: int = 1
    save_every: int = 1
    amp: bool = False
    deterministic: bool = True
    resume: Path | None = None


@dataclass
class EvalConfig:
    recall_ks: list[int] = field(default_factory=lambda: [1, 5, 10])
    batch_size: int = 32


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str = "demo"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ExperimentConfig:
        return from_dict(cls, raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping, got {type(raw)}")
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        def convert(obj: Any) -> Any:
            if is_dataclass(obj):
                return {f.name: convert(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(self)


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path)
