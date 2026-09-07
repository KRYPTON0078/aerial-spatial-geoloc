from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aerial_geoloc.config import load_config
from aerial_geoloc.data.datasets import build_retrieval_datasets, build_train_dataset
from aerial_geoloc.engine.retrieval import evaluate_model
from aerial_geoloc.models.encoder import build_encoder
from aerial_geoloc.models.losses import GeoLocCriterion
from aerial_geoloc.seed import seed_everything


def test_one_train_step_and_eval(tmp_path: Path):
    seed_everything(0)
    cfg = load_config(Path("configs/demo.yaml"))
    cfg.data.root = tmp_path / "demo"
    cfg.data.demo_train_locations = 4
    cfg.data.demo_test_locations = 3
    cfg.data.demo_drones_per_location = 2
    cfg.data.image_size = 32
    cfg.model.embed_dim = 32
    cfg.train.batch_size = 4

    train_ds = build_train_dataset(cfg)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    model = build_encoder("tiny_cnn", "shared", 32, train_ds.num_classes, False, 0.0, 32)
    crit = GeoLocCriterion(name="infonce_id", id_weight=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = next(iter(loader))
    opt.zero_grad()
    out = model(batch["query"], batch["gallery"])
    loss = crit(out, batch["location_id"])
    loss.total.backward()
    opt.step()
    assert torch.isfinite(loss.total)

    query_ds, gallery_ds = build_retrieval_datasets(cfg)
    metrics = evaluate_model(
        model,
        DataLoader(query_ds, batch_size=8),
        DataLoader(gallery_ds, batch_size=8),
        query_view="drone",
        gallery_view="satellite",
        device=torch.device("cpu"),
        recall_ks=[1, 5],
    )
    assert "R@1" in metrics and "mAP" in metrics
    assert metrics["n_query"] == len(query_ds)
