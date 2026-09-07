"""Optional Gradio UI for top-k drone → satellite retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from aerial_geoloc.config import load_config
from aerial_geoloc.data.datasets import build_retrieval_datasets
from aerial_geoloc.data.transforms import build_eval_transforms
from aerial_geoloc.engine.checkpoint import load_checkpoint, model_from_checkpoint
from aerial_geoloc.engine.retrieval import cosine_scores, embed_loader
from aerial_geoloc.engine.trainer import resolve_device
from aerial_geoloc.models.encoder import build_encoder


def _load_model(config_path: Path, checkpoint: Path | None):
    cfg = load_config(config_path)
    device = resolve_device(cfg.train.device)
    if checkpoint and Path(checkpoint).is_file():
        ckpt = load_checkpoint(checkpoint, map_location=device)
        n_classes = ckpt["config"].get("model", {}).get("num_classes")
        model = model_from_checkpoint(ckpt, num_classes=n_classes)
        trained = True
    else:
        model = build_encoder(
            backbone=cfg.model.backbone,
            encoder=cfg.model.encoder,
            embed_dim=cfg.model.embed_dim,
            num_classes=16,
            pretrained=False,
            dropout=cfg.model.dropout,
            image_size=cfg.data.image_size,
        )
        trained = False
    model.to(device).eval()
    return cfg, model, device, trained


def launch_app(config_path: str | Path, checkpoint: str | Path | None = None, port: int = 7860) -> None:
    try:
        import gradio as gr
        import torch
    except ImportError as exc:
        raise SystemExit("Gradio is optional. Install with: pip install '.[ui]'") from exc

    cfg, model, device, trained = _load_model(Path(config_path), Path(checkpoint) if checkpoint else None)
    _, gallery_ds = build_retrieval_datasets(cfg)
    transform = build_eval_transforms(cfg.data.image_size)
    gallery_loader = DataLoader(gallery_ds, batch_size=cfg.eval.batch_size, shuffle=False)
    g_feat, g_ids, g_paths = embed_loader(model, gallery_loader, cfg.data.gallery_view, device, desc="gallery")

    def search(query_img: Image.Image | None, topk: int):
        if query_img is None:
            return [], "Provide a query image (drone-view)."
        tensor = transform(query_img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            q = model.encode(tensor, view=cfg.data.query_view).cpu().numpy()
        scores = cosine_scores(q, g_feat)[0]
        order = np.argsort(-scores)[: int(topk)]
        previews = []
        for idx in order:
            img = Image.open(g_paths[int(idx)]).convert("RGB")
            caption = f"id={int(g_ids[int(idx)])}  score={scores[int(idx)]:.3f}"
            previews.append((img, caption))
        note = "Using trained checkpoint." if trained else "Untrained encoder — pass --checkpoint after training."
        return previews, note

    demo = gr.Interface(
        fn=search,
        inputs=[
            gr.Image(type="pil", label="Drone / query image"),
            gr.Slider(1, 8, value=5, step=1, label="top-k"),
        ],
        outputs=[
            gr.Gallery(label="Satellite gallery matches", columns=5, height=280),
            gr.Textbox(label="Status"),
        ],
        title="aerial-spatial-geoloc",
        description=(
            "Cross-view retrieval (drone query → satellite gallery). "
            "Default gallery is the synthetic demo split unless --config points at University-1652."
        ),
    )
    demo.launch(server_name="0.0.0.0", server_port=port)
