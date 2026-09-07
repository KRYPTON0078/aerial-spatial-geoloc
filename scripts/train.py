"""Thin wrappers so `python scripts/train.py` works after `pip install -e .`."""

from aerial_geoloc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *(__import__("sys").argv[1:])]))
