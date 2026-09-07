#!/usr/bin/env python3
"""Regenerate the CPU simulation study, metrics, and figures."""

from aerial_geoloc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["study", *(__import__("sys").argv[1:])]))
