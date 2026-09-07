from pathlib import Path

from aerial_geoloc.data.datasets import PairedCrossViewDataset, RetrievalDataset
from aerial_geoloc.data.demo import generate_demo_dataset, load_demo_samples
from aerial_geoloc.data.transforms import build_eval_transforms
from aerial_geoloc.data.university1652 import load_university1652_samples


def test_generate_demo_layout(tmp_path: Path):
    root = generate_demo_dataset(
        tmp_path / "demo",
        train_locations=4,
        test_locations=2,
        drones_per_location=3,
        streets_per_location=1,
        image_size=32,
    )
    assert (root / "manifest.json").is_file()
    train = load_demo_samples(root, "train")
    query = load_demo_samples(root, "query")
    gallery = load_demo_samples(root, "gallery")
    assert {s.view for s in train} == {"drone", "satellite", "street"}
    assert len([s for s in train if s.view == "drone"]) == 4 * 3
    assert len([s for s in gallery if s.view == "satellite"]) == 2
    assert max(s.location_id for s in train) == 3
    assert min(s.location_id for s in query) == 4


def test_paired_dataset_same_location(tmp_path: Path):
    root = generate_demo_dataset(tmp_path / "demo", train_locations=3, test_locations=1, drones_per_location=2, image_size=32)
    samples = load_demo_samples(root, "train")
    ds = PairedCrossViewDataset(samples, query_view="drone", gallery_view="satellite", transform=build_eval_transforms(32))
    item = ds[0]
    assert item["query"].shape == (3, 32, 32)
    assert item["query"].shape == item["gallery"].shape
    # gallery sampled from same id
    q_id = item["location_id"]
    assert any(s.location_id == q_id and s.view == "satellite" for s in samples)


def test_retrieval_dataset_filters_view(tmp_path: Path):
    root = generate_demo_dataset(tmp_path / "demo", train_locations=2, test_locations=2, drones_per_location=2, image_size=32)
    query = load_demo_samples(root, "query")
    ds = RetrievalDataset(query, view="drone", transform=build_eval_transforms(16))
    assert all(ds[i]["view"] == "drone" for i in range(len(ds)))


def test_university1652_missing_root(tmp_path: Path):
    try:
        load_university1652_samples(tmp_path / "missing", "train")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "University-1652" in str(exc)


def test_university1652_fake_tree(tmp_path: Path):
    root = tmp_path / "u1652"
    for split_view in [
        ("train", "drone"),
        ("train", "satellite"),
        ("test", "query_drone"),
        ("test", "gallery_satellite"),
    ]:
        folder = root / split_view[0] / split_view[1] / "0001"
        folder.mkdir(parents=True)
        from PIL import Image

        Image.new("RGB", (8, 8), (1, 2, 3)).save(folder / "x.jpg")
    samples = load_university1652_samples(root, "train")
    assert {s.view for s in samples} >= {"drone", "satellite"}
    query = load_university1652_samples(root, "query")
    gallery = load_university1652_samples(root, "gallery")
    assert query[0].location_id == gallery[0].location_id
