.PHONY: install dev test run figure clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

run:
	beetle-analyze blimp

figure:
	beetle-analyze blimp --plot-out out/blimp_toy.png

clean:
	rm -rf out *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
