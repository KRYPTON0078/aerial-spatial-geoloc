from pathlib import Path

from aerial_geoloc.config import ExperimentConfig, load_config


def test_all_shipped_configs_load():
    for name in (
        "demo.yaml",
        "default.yaml",
        "ablation_dual_branch.yaml",
        "ablation_weather.yaml",
        "ablation_vit.yaml",
        "ablation_id_loss.yaml",
    ):
        cfg = load_config(Path("configs") / name)
        assert cfg.experiment_name
        assert cfg.model.backbone



def test_unknown_key_rejected(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  nope: 1\n", encoding="utf-8")
    try:
        load_config(p)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown config keys" in str(exc)


def test_roundtrip_dict():
    cfg = ExperimentConfig()
    restored = ExperimentConfig.from_mapping(cfg.to_dict())
    assert restored.model.embed_dim == cfg.model.embed_dim
    assert restored.data.weather_types == cfg.data.weather_types
