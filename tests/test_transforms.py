import torch

from aerial_geoloc.data.transforms import IMAGENET_MEAN, build_eval_transforms, build_train_transforms, denormalize_tensor
from PIL import Image


def test_eval_transform_shape_and_norm():
    img = Image.new("RGB", (40, 30), (10, 20, 30))
    tensor = build_eval_transforms(32)(img)
    assert tensor.shape == (3, 32, 32)
    # Inverse should land in [0, 1]
    rec = denormalize_tensor(tensor)
    assert rec.min() >= -1e-5
    assert rec.max() <= 1.0 + 1e-5


def test_train_transform_with_weather():
    img = Image.new("RGB", (48, 48), (100, 80, 40))
    t = build_train_transforms(24, weather_aug=True, weather_types=["fog"], weather_prob=1.0)
    tensor = t(img)
    assert tensor.shape == (3, 24, 24)
    assert torch.isfinite(tensor).all()


def test_imagenet_stats_len():
    assert len(IMAGENET_MEAN) == 3
