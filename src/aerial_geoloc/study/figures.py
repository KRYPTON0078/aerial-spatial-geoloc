"""Commit-quality matplotlib figures from measured study metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": 140,
        }
    )


def plot_training_curves(results: dict, dest: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for key, label, color in (
        ("baseline", "baseline (clean train)", "#1f4e79"),
        ("weather_aug", "weather-aug train", "#c45911"),
    ):
        hist = results[key]["history"]
        epochs = [h["epoch"] for h in hist]
        loss = [h["train"]["loss"] for h in hist]
        r1 = [h["eval"].get("R@1", float("nan")) for h in hist]
        axes[0].plot(epochs, loss, marker="o", label=label, color=color)
        axes[1].plot(epochs, r1, marker="o", label=label, color=color)
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Clean-test Recall@1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Simulation study — training dynamics (synthetic campus, not U1652)", fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def plot_weather_bars(results: dict, dest: Path) -> None:
    _style()
    conditions = list(results["protocol"]["conditions"])
    x = np.arange(len(conditions))
    width = 0.22
    series = [
        ("baseline cosine", [results["baseline"]["conditions"][c]["cosine"]["R@1"] for c in conditions], "#1f4e79"),
        ("weather-aug cosine", [results["weather_aug"]["conditions"][c]["cosine"]["R@1"] for c in conditions], "#c45911"),
        (
            "weather-aug + UA",
            [results["weather_aug"]["conditions"][c]["uncertainty_aware"]["ua_R@1"] for c in conditions],
            "#548235",
        ),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    for i, (label, vals, color) in enumerate(series):
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20)
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, 1.08)
    ax.set_title("Drone→Sat Recall@1 under labeled nuisances (simulated test set)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def plot_map_bars(results: dict, dest: Path) -> None:
    _style()
    conditions = list(results["protocol"]["conditions"])
    x = np.arange(len(conditions))
    width = 0.35
    b = [results["baseline"]["conditions"][c]["cosine"]["mAP"] for c in conditions]
    w = [results["weather_aug"]["conditions"][c]["cosine"]["mAP"] for c in conditions]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.bar(x - width / 2, b, width, label="baseline", color="#1f4e79")
    ax.bar(x + width / 2, w, width, label="weather-aug", color="#c45911")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20)
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.08)
    ax.set_title("mAP across conditions (simulated test set)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def plot_architecture(dest: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", facecolor=color, edgecolor="#222")
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)

    box(0.2, 2.3, 2.1, 1.1, "Drone query\n(+ weather / yaw)", "#d6eaf8")
    box(0.2, 0.4, 2.1, 1.1, "Satellite gallery\n(nadir, north-up)", "#e8daef")
    box(2.8, 2.3, 2.3, 1.1, "Query encoder\nTinyCNN + proj", "#fdebd0")
    box(2.8, 0.4, 2.3, 1.1, "Gallery encoder\nshared or dual", "#fdebd0")
    box(5.6, 1.2, 2.4, 1.4, "L2 embed z\nInfoNCE + ID", "#d5f5e3")
    box(8.5, 2.2, 3.1, 1.2, "Cosine rank\nR@K / mAP", "#fadbd8")
    box(8.5, 0.35, 3.1, 1.2, "MC-Dropout UA\nscore × 1/(1+σ)", "#f9e79f")
    ax.add_patch(FancyArrowPatch((2.3, 2.85), (2.8, 2.85), arrowstyle="->", mutation_scale=12, color="#333"))
    ax.add_patch(FancyArrowPatch((2.3, 0.95), (2.8, 0.95), arrowstyle="->", mutation_scale=12, color="#333"))
    ax.add_patch(FancyArrowPatch((5.1, 2.85), (6.2, 2.3), arrowstyle="->", mutation_scale=12, color="#333"))
    ax.add_patch(FancyArrowPatch((5.1, 0.95), (6.2, 1.4), arrowstyle="->", mutation_scale=12, color="#333"))
    ax.add_patch(FancyArrowPatch((8.0, 2.1), (8.5, 2.6), arrowstyle="->", mutation_scale=12, color="#333"))
    ax.add_patch(FancyArrowPatch((8.0, 1.6), (8.5, 1.1), arrowstyle="->", mutation_scale=12, color="#333"))
    ax.set_title("aerial-spatial-geoloc — dual-view retrieval with uncertainty-aware scoring")
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def _pca_2d(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T


def plot_embedding_pca(cfg, checkpoint: Path, dest: Path) -> None:
    from aerial_geoloc.data.datasets import build_retrieval_datasets
    from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint
    from aerial_geoloc.engine.retrieval import embed_loader
    from aerial_geoloc.engine.trainer import resolve_device
    from torch.utils.data import DataLoader

    _style()
    device = resolve_device(cfg.train.device)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model = model_from_checkpoint(ckpt, num_classes=ckpt["config"].get("model", {}).get("num_classes")).to(device)
    query_ds, gallery_ds = build_retrieval_datasets(cfg)
    q, q_ids, _ = embed_loader(model, DataLoader(query_ds, batch_size=32), cfg.data.query_view, device, desc="pca-q")
    g, g_ids, _ = embed_loader(model, DataLoader(gallery_ds, batch_size=32), cfg.data.gallery_view, device, desc="pca-g")
    feats = np.concatenate([q, g], axis=0)
    ids = np.concatenate([q_ids, g_ids], axis=0)
    views = np.array(["drone"] * len(q) + ["sat"] * len(g))
    xy = _pca_2d(feats)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for loc in sorted(set(ids.tolist())):
        m = ids == loc
        ax.scatter(xy[m & (views == "drone"), 0], xy[m & (views == "drone"), 1], s=28, marker="o", alpha=0.85)
        ax.scatter(xy[m & (views == "sat"), 0], xy[m & (views == "sat"), 1], s=70, marker="*", alpha=0.95)
    ax.set_title("PCA of embeddings (o drone, * satellite), colored by location")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_reliability(results: dict, dest: Path) -> None:
    """Bar of mean MC-dropout uncertainty per condition for the weather-aug model."""
    _style()
    conditions = list(results["protocol"]["conditions"])
    unc = [results["weather_aug"]["conditions"][c]["uncertainty_aware"]["mean_query_uncertainty"] for c in conditions]
    r1 = [results["weather_aug"]["conditions"][c]["cosine"]["R@1"] for c in conditions]
    fig, ax1 = plt.subplots(figsize=(9.2, 3.8))
    ax1.bar(conditions, unc, color="#7d3c98", alpha=0.75, label="mean query σ (MC-Dropout)")
    ax1.set_ylabel("mean embedding std")
    ax2 = ax1.twinx()
    ax2.plot(conditions, r1, color="#c0392b", marker="o", label="R@1")
    ax2.set_ylabel("Recall@1")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Does uncertainty rise where retrieval breaks? (weather-aug model)")
    fig.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


def _border(img: Image.Image, color: tuple[int, int, int], width: int = 3) -> Image.Image:
    canvas = Image.new("RGB", (img.width + 2 * width, img.height + 2 * width), color)
    canvas.paste(img, (width, width))
    return canvas


def plot_retrieval_grid(cfg, checkpoint: Path, dest: Path, condition: str = "fog", n_queries: int = 4, topk: int = 5) -> None:
    from aerial_geoloc.data.datasets import RetrievalDataset, build_retrieval_datasets, load_samples, prepare_data_root
    from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint
    from aerial_geoloc.engine.retrieval import cosine_scores, embed_loader
    from aerial_geoloc.engine.trainer import resolve_device
    from aerial_geoloc.sim.nuisances import ConditionTransform, apply_condition
    from torch.utils.data import DataLoader

    device = resolve_device(cfg.train.device)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model = model_from_checkpoint(ckpt, num_classes=ckpt["config"].get("model", {}).get("num_classes")).to(device)
    root = prepare_data_root(cfg.data)
    query_samples = load_samples(cfg.data.dataset, root, "query")
    q_ds = RetrievalDataset(query_samples, view=cfg.data.query_view, transform=ConditionTransform(cfg.data.image_size, condition))
    _, g_ds = build_retrieval_datasets(cfg)
    q_feat, q_ids, q_paths = embed_loader(model, DataLoader(q_ds, batch_size=32), cfg.data.query_view, device, desc="grid-q")
    g_feat, g_ids, g_paths = embed_loader(model, DataLoader(g_ds, batch_size=32), cfg.data.gallery_view, device, desc="grid-g")
    scores = cosine_scores(q_feat, g_feat)
    order = np.argsort(-scores, axis=1)

    cell = 72
    rows = min(n_queries, len(q_paths))
    cols = 1 + topk
    canvas = Image.new("RGB", (cols * (cell + 6) + 8, rows * (cell + 22) + 28), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), f"Query ({condition}) → top-{topk} satellite  |  green=correct  red=wrong", fill=(20, 20, 20))
    for r in range(rows):
        q_img = apply_condition(Image.open(q_paths[r]).convert("RGB"), condition, intensity=0.55, seed=r)
        q_img = q_img.resize((cell, cell))
        canvas.paste(_border(q_img, (40, 80, 160), 2), (6, 24 + r * (cell + 22)))
        for k in range(topk):
            gi = int(order[r, k])
            g_img = Image.open(g_paths[gi]).convert("RGB").resize((cell, cell))
            ok = int(g_ids[gi]) == int(q_ids[r])
            bordered = _border(g_img, (32, 140, 70) if ok else (180, 40, 40), 3)
            canvas.paste(bordered, (6 + (k + 1) * (cell + 6), 24 + r * (cell + 22)))
    canvas.save(dest)


def plot_failure_collage(cfg, checkpoint: Path, dest: Path, condition: str = "night") -> None:
    from aerial_geoloc.data.datasets import RetrievalDataset, build_retrieval_datasets, load_samples, prepare_data_root
    from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint
    from aerial_geoloc.engine.retrieval import cosine_scores, embed_loader
    from aerial_geoloc.engine.trainer import resolve_device
    from aerial_geoloc.sim.nuisances import ConditionTransform, apply_condition
    from torch.utils.data import DataLoader

    device = resolve_device(cfg.train.device)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model = model_from_checkpoint(ckpt, num_classes=ckpt["config"].get("model", {}).get("num_classes")).to(device)
    root = prepare_data_root(cfg.data)
    q_ds = RetrievalDataset(
        load_samples(cfg.data.dataset, root, "query"),
        view=cfg.data.query_view,
        transform=ConditionTransform(cfg.data.image_size, condition),
    )
    _, g_ds = build_retrieval_datasets(cfg)
    q_feat, q_ids, q_paths = embed_loader(model, DataLoader(q_ds, batch_size=32), cfg.data.query_view, device, desc="fail-q")
    g_feat, g_ids, g_paths = embed_loader(model, DataLoader(g_ds, batch_size=32), cfg.data.gallery_view, device, desc="fail-g")
    scores = cosine_scores(q_feat, g_feat)
    pred = np.argmax(scores, axis=1)
    wrong = [i for i in range(len(q_ids)) if int(g_ids[pred[i]]) != int(q_ids[i])]
    if not wrong:
        # all correct — still dump a "no failures" card
        img = Image.new("RGB", (480, 120), (240, 248, 240))
        ImageDraw.Draw(img).text((20, 50), f"No top-1 failures under {condition} on this split.", fill=(20, 80, 30))
        img.save(dest)
        return
    cell = 80
    n = min(6, len(wrong))
    canvas = Image.new("RGB", (3 * (cell + 10) + 20, n * (cell + 16) + 30), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"Failures under {condition}: query | predicted sat | true sat", fill=(20, 20, 20))
    for row, qi in enumerate(wrong[:n]):
        q_img = apply_condition(Image.open(q_paths[qi]).convert("RGB"), condition, 0.55, seed=qi).resize((cell, cell))
        pred_img = Image.open(g_paths[int(pred[qi])]).convert("RGB").resize((cell, cell))
        true_idx = int(np.where(g_ids == q_ids[qi])[0][0])
        true_img = Image.open(g_paths[true_idx]).convert("RGB").resize((cell, cell))
        y = 26 + row * (cell + 16)
        canvas.paste(_border(q_img, (40, 80, 160), 2), (8, y))
        canvas.paste(_border(pred_img, (180, 40, 40), 3), (8 + cell + 10, y))
        canvas.paste(_border(true_img, (32, 140, 70), 3), (8 + 2 * (cell + 10), y))
    canvas.save(dest)


def render_all_figures(results: dict, cfg, figures_dir: Path, baseline_ckpt: Path, weather_ckpt: Path) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_architecture(figures_dir / "architecture.png")
    plot_training_curves(results, figures_dir / "training_curves.png")
    plot_weather_bars(results, figures_dir / "recall_by_condition.png")
    plot_map_bars(results, figures_dir / "map_by_condition.png")
    plot_uncertainty_reliability(results, figures_dir / "uncertainty_vs_recall.png")
    plot_embedding_pca(cfg, weather_ckpt, figures_dir / "embedding_pca.png")
    plot_retrieval_grid(cfg, weather_ckpt, figures_dir / "retrieval_grid_fog.png", condition="fog")
    plot_failure_collage(cfg, baseline_ckpt, figures_dir / "failures_night.png", condition="night")
