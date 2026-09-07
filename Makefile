.PHONY: install test demo retrieve study

install:
	pip install -e ".[dev,study]"

test:
	python -m pytest -q

demo:
	python -m aerial_geoloc.cli train --config configs/demo.yaml

retrieve:
	python -m aerial_geoloc.cli retrieve --config configs/demo.yaml --checkpoint outputs/demo/best.pt --topk 5

study:
	python -m aerial_geoloc.cli study --config configs/study.yaml

install:
	pip install -e ".[dev]"

test:
	pytest

demo:
	python -m aerial_geoloc.cli train --config configs/demo.yaml

retrieve:
	python -m aerial_geoloc.cli retrieve --config configs/demo.yaml --checkpoint outputs/demo/best.pt --topk 5
