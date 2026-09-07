"""CPU study: baseline vs weather-aug vs uncertainty-aware scoring."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from torch.utils.data import DataLoader

from aerial_geoloc.config import ExperimentConfig, load_config
from aerial_geoloc.data.datasets import RetrievalDataset, build_retrieval_datasets, load_samples, prepare_data_root
from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint
from aerial_geoloc.engine.retrieval import evaluate_model
from aerial_geoloc.engine.trainer import resolve_device, run_training
from aerial_geoloc.engine.uncertainty import evaluate_uncertainty_aware
from aerial_geoloc.seed import seed_everything
from aerial_geoloc.sim.nuisances import CONDITION_NAMES, ConditionTransform
from aerial_geoloc.sim.world import generate_sim_dataset


def _cfg_variant(cfg: ExperimentConfig, *, weather_aug: bool, output_dir: Path, name: str) -> ExperimentConfig:
    variant = copy.deepcopy(cfg)
    variant.data.weather_aug = weather_aug
    variant.train.output_dir = output_dir
    variant.experiment_name = name
    return variant


def _query_loader(cfg: ExperimentConfig, condition: str) -> DataLoader:
    root = prepare_data_root(cfg.data)
    samples = load_samples(cfg.data.dataset, root, "query")
    transform = ConditionTransform(cfg.data.image_size, condition=condition, intensity=0.55)
    ds = RetrievalDataset(samples, view=cfg.data.query_view, transform=transform)
    return DataLoader(ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=0)


def evaluate_conditions(cfg: ExperimentConfig, checkpoint: Path, conditions: tuple[str, ...] | None = None) -> dict:
    device = resolve_device(cfg.train.device)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    n_classes = ckpt["config"].get("model", {}).get("num_classes")
    model = model_from_checkpoint(ckpt, num_classes=n_classes).to(device)
    _, gallery_ds = build_retrieval_datasets(cfg)
    gallery_loader = DataLoader(gallery_ds, batch_size=cfg.eval.batch_size, shuffle=False)
    conditions = conditions or CONDITION_NAMES
    rows = {}
    for cond in conditions:
        q_loader = _query_loader(cfg, cond)
        cosine = evaluate_model(
            model, q_loader, gallery_loader, cfg.data.query_view, cfg.data.gallery_view, device, cfg.eval.recall_ks
        )
        ua = evaluate_uncertainty_aware(
            model,
            q_loader,
            gallery_loader,
            cfg.data.query_view,
            cfg.data.gallery_view,
            device,
            cfg.eval.recall_ks,
            mc_samples=6,
        )
        rows[cond] = {"cosine": cosine, "uncertainty_aware": ua}
    return rows


def run_study(
    config_path: str | Path = "configs/study.yaml",
    reports_dir: str | Path = "reports",
    figures_dir: str | Path = "figures",
    regenerate_data: bool = False,
) -> dict:
    cfg = load_config(config_path)
    seed_everything(cfg.train.seed, deterministic=True)
    root = Path(cfg.data.root)
    if regenerate_data or not (root / "manifest.json").exists():
        generate_sim_dataset(
            root,
            train_locations=cfg.data.demo_train_locations,
            test_locations=cfg.data.demo_test_locations,
            drones_per_location=cfg.data.demo_drones_per_location,
            streets_per_location=cfg.data.demo_streets_per_location,
            image_size=cfg.data.image_size,
            seed=cfg.train.seed,
        )

    baseline_dir = Path("outputs/study_baseline")
    weather_dir = Path("outputs/study_weather")
    baseline_cfg = _cfg_variant(cfg, weather_aug=False, output_dir=baseline_dir, name="sim_baseline")
    weather_cfg = _cfg_variant(cfg, weather_aug=True, output_dir=weather_dir, name="sim_weather_aug")

    print("=== train baseline (clean) ===")
    run_training(baseline_cfg)
    print("=== train weather-aug ===")
    run_training(weather_cfg)

    results = {
        "protocol": {
            "dataset": "simulated_campus",
            "not_university1652": True,
            "seed": cfg.train.seed,
            "epochs": cfg.train.epochs,
            "backbone": cfg.model.backbone,
            "image_size": cfg.data.image_size,
            "train_locations": cfg.data.demo_train_locations,
            "test_locations": cfg.data.demo_test_locations,
            "mc_samples": 6,
            "conditions": list(CONDITION_NAMES),
        },
        "baseline": {
            "history": _read_json(baseline_dir / "history.json"),
            "conditions": evaluate_conditions(baseline_cfg, baseline_dir / "best.pt"),
        },
        "weather_aug": {
            "history": _read_json(weather_dir / "history.json"),
            "conditions": evaluate_conditions(weather_cfg, weather_dir / "best.pt"),
        },
    }

    reports_dir = Path(reports_dir)
    figures_dir = Path(figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    from aerial_geoloc.study.figures import render_all_figures

    render_all_figures(results, cfg, figures_dir, baseline_ckpt=baseline_dir / "best.pt", weather_ckpt=weather_dir / "best.pt")
    print(f"wrote {metrics_path} and figures under {figures_dir}")
    return results


def _read_json(path: Path):
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
