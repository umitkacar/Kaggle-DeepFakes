# Makefile for DeepFake Detector

.PHONY: help install install-dev format lint test clean build

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

format:  ## Format code with black and ruff
	black src tests
	ruff --fix src tests

lint:  ## Run linters
	black --check src tests
	ruff src tests
	mypy src

test:  ## Run tests
	pytest tests/ -v --cov=src/deepfake_detector

test-verbose:  ## Run tests with verbose output
	pytest tests/ -vv --cov=src/deepfake_detector --cov-report=term-missing

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	hatch build

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

setup-hooks:  ## Setup git hooks
	pre-commit install
	@echo "Git hooks installed successfully!"
