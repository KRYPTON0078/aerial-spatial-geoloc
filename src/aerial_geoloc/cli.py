"""Command-line interface: train, eval, retrieve, make-demo-data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aerial_geoloc.config import load_config
from aerial_geoloc.data.demo import generate_demo_dataset


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"), help="YAML experiment config")


def cmd_train(args: argparse.Namespace) -> int:
    from aerial_geoloc.engine.trainer import run_training

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.train.output_dir = Path(args.output_dir)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.encoder:
        cfg.model.encoder = args.encoder
    if args.loss:
        cfg.loss.name = args.loss
    if args.weather_aug is not None:
        cfg.data.weather_aug = args.weather_aug
    if args.seed is not None:
        cfg.train.seed = args.seed
    metrics = run_training(cfg)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from aerial_geoloc.engine.evaluator import run_eval

    cfg = load_config(args.config)
    metrics = run_eval(cfg, args.checkpoint)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    from aerial_geoloc.engine.demo_retrieve import run_retrieve

    cfg = load_config(args.config)
    rows = run_retrieve(
        cfg,
        checkpoint=args.checkpoint,
        query_path=args.query,
        gallery_dir=args.gallery_dir,
        topk=args.topk,
        max_queries=args.max_queries,
    )
    if args.json:
        print(json.dumps(rows, indent=2))
    return 0


def cmd_make_demo(args: argparse.Namespace) -> int:
    root = generate_demo_dataset(
        args.root,
        train_locations=args.train_locations,
        test_locations=args.test_locations,
        drones_per_location=args.drones,
        image_size=args.image_size,
    )
    print(f"wrote synthetic University-1652-style demo data to {root}")
    return 0


def cmd_sim_generate(args: argparse.Namespace) -> int:
    from aerial_geoloc.sim.world import generate_sim_dataset

    root = generate_sim_dataset(
        args.root,
        train_locations=args.train_locations,
        test_locations=args.test_locations,
        drones_per_location=args.drones,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"wrote simulated campus with pose metadata to {root}")
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    from aerial_geoloc.study.runner import run_study

    run_study(
        config_path=args.config,
        reports_dir=args.reports,
        figures_dir=args.figures,
        regenerate_data=args.regenerate,
    )
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    from aerial_geoloc.app import launch_app

    launch_app(config_path=args.config, checkpoint=args.checkpoint, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerial-geoloc",
        description="Cross-view aerial geo-localization toolkit (drone ↔ satellite).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a retrieval model")
    _add_config(train)
    train.add_argument("--output-dir", type=Path, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--backbone", choices=["tiny_cnn", "resnet18", "resnet50", "vit_tiny"], default=None)
    train.add_argument("--encoder", choices=["shared", "dual"], default=None)
    train.add_argument("--loss", choices=["infonce", "id", "infonce_id", "triplet"], default=None)
    train.add_argument("--weather-aug", dest="weather_aug", action="store_true")
    train.add_argument("--no-weather-aug", dest="weather_aug", action="store_false")
    train.add_argument("--seed", type=int, default=None)
    train.set_defaults(func=cmd_train, weather_aug=None)

    ev = sub.add_parser("eval", help="Evaluate a checkpoint on the configured query/gallery split")
    _add_config(ev)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.set_defaults(func=cmd_eval)

    ret = sub.add_parser("retrieve", help="Print top-k gallery matches for query images")
    _add_config(ret)
    ret.add_argument("--checkpoint", type=Path, default=None, help="Optional; random encoder if omitted")
    ret.add_argument("--query", type=Path, default=None, help="Single query image; default: demo split")
    ret.add_argument("--gallery-dir", type=Path, default=None)
    ret.add_argument("--topk", type=int, default=5)
    ret.add_argument("--max-queries", type=int, default=8)
    ret.add_argument("--json", action="store_true", help="Also print machine-readable matches")
    ret.set_defaults(func=cmd_retrieve)

    demo = sub.add_parser("make-demo-data", help="Generate synthetic drone/satellite tiles")
    demo.add_argument("--root", type=Path, default=Path("data/demo"))
    demo.add_argument("--train-locations", type=int, default=16)
    demo.add_argument("--test-locations", type=int, default=8)
    demo.add_argument("--drones", type=int, default=6)
    demo.add_argument("--image-size", type=int, default=128)
    demo.set_defaults(func=cmd_make_demo)

    sim = sub.add_parser("sim-generate", help="Generate the simulated campus with pose metadata")
    sim.add_argument("--root", type=Path, default=Path("data/sim"))
    sim.add_argument("--train-locations", type=int, default=12)
    sim.add_argument("--test-locations", type=int, default=8)
    sim.add_argument("--drones", type=int, default=4)
    sim.add_argument("--image-size", type=int, default=64)
    sim.add_argument("--seed", type=int, default=42)
    sim.set_defaults(func=cmd_sim_generate)

    study = sub.add_parser("study", help="Run CPU simulation study and write figures/metrics")
    _add_config(study)
    study.set_defaults(config=Path("configs/study.yaml"))
    study.add_argument("--reports", type=Path, default=Path("reports"))
    study.add_argument("--figures", type=Path, default=Path("figures"))
    study.add_argument("--regenerate", action="store_true", help="Rebuild data/sim even if present")
    study.set_defaults(func=cmd_study)

    app = sub.add_parser("app", help="Optional Gradio UI (pip install '.[ui]')")
    _add_config(app)
    app.add_argument("--checkpoint", type=Path, default=None)
    app.add_argument("--port", type=int, default=7860)
    app.set_defaults(func=cmd_app)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
