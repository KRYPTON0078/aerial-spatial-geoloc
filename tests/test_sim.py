from pathlib import Path

from aerial_geoloc.sim.nuisances import apply_condition, apply_occlusion
from aerial_geoloc.sim.world import generate_sim_dataset, load_manifest, load_sim_samples


def test_sim_dataset_has_pose_metadata(tmp_path: Path):
    root = generate_sim_dataset(tmp_path / "sim", train_locations=3, test_locations=2, drones_per_location=2, image_size=32)
    manifest = load_manifest(root)
    assert manifest["kind"] == "simulated_campus"
    assert "frames" in manifest
    assert any(f["pose"]["altitude_m"] > 0 for f in manifest["frames"])
    train = load_sim_samples(root, "train")
    query = load_sim_samples(root, "query")
    assert {s.view for s in train} >= {"drone", "satellite"}
    assert min(s.location_id for s in query) >= 3


def test_conditions_preserve_size():
    from PIL import Image

    src = Image.new("RGB", (48, 48), (10, 80, 20))
    for cond in ("clean", "fog", "occlusion", "yaw", "altitude"):
        out = apply_condition(src, cond, intensity=0.4, seed=1)
        assert out.size == src.size


def test_occlusion_changes_pixels():
    from PIL import Image
    import numpy as np

    src = Image.new("RGB", (40, 40), (200, 200, 200))
    out = apply_occlusion(src, intensity=0.8, seed=3)
    assert not np.array_equal(np.asarray(src), np.asarray(out))
