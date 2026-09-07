.PHONY: install test demo retrieve

install:
	pip install -e ".[dev]"

test:
	pytest

demo:
	python -m aerial_geoloc.cli train --config configs/demo.yaml

retrieve:
	python -m aerial_geoloc.cli retrieve --config configs/demo.yaml --checkpoint outputs/demo/best.pt --topk 5
