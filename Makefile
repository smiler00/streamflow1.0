# StreamFlow Development Makefile

.PHONY: help install install-dev test test-all lint format clean docs

help:
	@echo "Available commands:"
	@echo "  install     - Install the package in development mode"
	@echo "  install-dev - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  test-all    - Run tests with coverage"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black"
	@echo "  clean       - Clean build artifacts"
	@echo "  docs        - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-all:
	pytest tests/ --cov=streamflow --cov-report=html

lint:
	flake8 streamflow/
	mypy streamflow/

format:
	black streamflow/
	isort streamflow/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/

docs:
	sphinx-build docs/ docs/_build/
