from pathlib import Path

from PIL import Image

from aerial_geoloc.data.university1652 import (
    load_university1652_samples,
    resolve_university1652_root,
    university1652_root_ok,
)


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)


def _minimal_u1652_tree(root: Path, extra_street: bool = False, extra_google: bool = False) -> Path:
    for rel in [
        ("train", "drone", "0001"),
        ("train", "satellite", "0001"),
        ("test", "query_drone", "0701"),
        ("test", "gallery_satellite", "0701"),
    ]:
        _touch_jpg(root.joinpath(*rel) / "a.jpg")
    if extra_street:
        _touch_jpg(root / "train" / "street" / "0001" / "s.jpg")
        _touch_jpg(root / "test" / "query_street" / "0701" / "s.jpg")
        _touch_jpg(root / "test" / "gallery_street" / "0701" / "s.jpg")
    if extra_google:
        _touch_jpg(root / "train" / "google" / "0001" / "g.jpg")
    return root


def test_university1652_missing_root_explains_request(tmp_path: Path):
    try:
        load_university1652_samples(tmp_path / "missing", "train")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        text = str(exc)
        assert "University-1652" in text
        assert "zdzheng12@gmail.com" in text
        assert "Request.md" in text
        assert "huggingface.co/datasets/layumi/university-1652" in text
        assert "prepare_u1652.md" in text


def test_university1652_nested_university_release(tmp_path: Path):
    nested = tmp_path / "data" / "University-Release"
    _minimal_u1652_tree(nested)
    resolved = resolve_university1652_root(tmp_path / "data")
    assert resolved == nested
    samples = load_university1652_samples(tmp_path / "data", "train")
    assert {s.view for s in samples} == {"drone", "satellite"}


def test_university1652_protocol_views_and_ids(tmp_path: Path):
    root = _minimal_u1652_tree(tmp_path / "u1652", extra_street=True, extra_google=True)
    assert university1652_root_ok(root)
    train = load_university1652_samples(root, "train")
    assert {s.view for s in train} == {"drone", "satellite", "street", "google"}
    query = load_university1652_samples(root, "query")
    gallery = load_university1652_samples(root, "gallery")
    assert {s.view for s in query} >= {"drone", "street"}
    assert query[0].location_id == gallery[0].location_id
    assert all(s.split == "query" for s in query)


def test_university1652_4k_drone_not_in_default_splits(tmp_path: Path):
    root = _minimal_u1652_tree(tmp_path / "u1652")
    _touch_jpg(root / "test" / "4K_drone" / "clip" / "frame.jpg")
    query = load_university1652_samples(root, "query")
    assert all("4K" not in s.view for s in query)


def test_university1652_unknown_split(tmp_path: Path):
    root = _minimal_u1652_tree(tmp_path / "u1652")
    try:
        load_university1652_samples(root, "val")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "val" in str(exc)
