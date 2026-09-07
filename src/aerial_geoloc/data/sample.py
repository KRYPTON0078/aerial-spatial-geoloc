"""Image-sample records shared by dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    path: Path
    location_id: int
    view: str
    split: str
    location_name: str = ""
