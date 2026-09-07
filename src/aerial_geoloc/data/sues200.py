"""SUES-200 adapter stub.

SUES-200 is a real-drone geo-localization set often paired with University-1652
in robustness papers (including WeatherPrompt). This module documents the
expected layout; a full height-aware loader is left as an extension.
"""

from __future__ import annotations

from pathlib import Path

from aerial_geoloc.data.sample import Sample

DOWNLOAD_HELP = """\
SUES-200 was not found at {root}.

Typical layout after extraction::

    <root>/
      Training/
        drone/<height>/<id>/*.jpg
        satellite/<id>/*.jpg
      Testing/
        query_drone/<height>/<id>/*.jpg
        gallery_satellite/<id>/*.jpg

This toolkit currently ships University-1652 + the synthetic demo set.
"""


def load_sues200_samples(root: str | Path, split: str) -> list[Sample]:
    del split
    raise FileNotFoundError(DOWNLOAD_HELP.format(root=Path(root)))
