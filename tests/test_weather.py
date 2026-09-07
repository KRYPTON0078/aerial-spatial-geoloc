import numpy as np
from PIL import Image

from aerial_geoloc.data.weather import RandomWeather, apply_fog, apply_night, apply_weather


def _solid(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (32, 32), color)


def test_fog_moves_toward_white():
    src = _solid((10, 10, 10))
    out = np.asarray(apply_fog(src, intensity=0.8), dtype=np.float32)
    assert out.mean() > np.asarray(src).mean()


def test_night_darkens():
    src = _solid((200, 200, 200))
    out = np.asarray(apply_night(src, intensity=0.8), dtype=np.float32)
    assert out.mean() < np.asarray(src).mean()


def test_unknown_weather_raises():
    try:
        apply_weather(_solid((0, 0, 0)), "hail")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_random_weather_identity_when_p_zero():
    src = _solid((12, 34, 56))
    out = RandomWeather(p=0.0)(src)
    assert np.array_equal(np.asarray(src), np.asarray(out))


def test_all_weather_types_preserve_size():
    src = _solid((80, 90, 100))
    for name in ("fog", "rain", "night", "snow", "glare"):
        out = apply_weather(src, name, intensity=0.5)
        assert out.size == src.size
        assert out.mode == "RGB"
