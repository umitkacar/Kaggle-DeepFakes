# Makefile for DeepFake Detector

.PHONY: help install install-dev format lint test test-fast test-cov test-unit test-integration clean build validate setup production-check

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  DeepFake Detector - Production Makefile Commands$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install:  ## Install package (production)
	@echo "$(YELLOW)Installing production dependencies...$(NC)"
	pip install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev:  ## Install package with development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	pip install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Development installation complete$(NC)"

setup:  ## Complete development setup (automated)
	@echo "$(YELLOW)Running automated setup...$(NC)"
	chmod +x scripts/setup.sh
	./scripts/setup.sh

validate:  ## Run validation checks
	@echo "$(YELLOW)Running validation checks...$(NC)"
	python3 scripts/validate.py

format:  ## Format code with black and ruff
	@echo "$(YELLOW)Formatting code...$(NC)"
	ruff --fix src tests
	black src tests
	isort src tests
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint:  ## Run linters (ruff + black + mypy)
	@echo "$(YELLOW)Running linters...$(NC)"
	ruff src tests
	black --check src tests
	isort --check src tests
	mypy src
	@echo "$(GREEN)✓ Linting complete$(NC)"

test:  ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v

test-fast:  ## Run tests in parallel (pytest-xdist)
	@echo "$(YELLOW)Running tests in parallel...$(NC)"
	pytest tests/ -n auto -v

test-cov:  ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src/deepfake_detector --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(NC)"

test-unit:  ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	pytest tests/ -m unit -v

test-integration:  ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/ -m integration -v

test-verbose:  ## Run tests with verbose output
	@echo "$(YELLOW)Running tests (verbose)...$(NC)"
	pytest tests/ -vv --cov=src/deepfake_detector --cov-report=term-missing

clean:  ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

build:  ## Build package with hatch
	@echo "$(YELLOW)Building package...$(NC)"
	hatch build
	@echo "$(GREEN)✓ Build complete$(NC)"

pre-commit:  ## Run pre-commit hooks on all files
	@echo "$(YELLOW)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

setup-hooks:  ## Setup git hooks
	@echo "$(YELLOW)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)✓ Git hooks installed successfully!$(NC)"

check:  ## Run all quality checks (format + lint + test-fast)
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Running all quality checks...$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) test-fast
	@echo "$(GREEN)✓ All checks passed$(NC)"

production-check:  ## Complete production readiness check
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Production Readiness Check$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"
	@$(MAKE) validate
	@$(MAKE) lint
	@$(MAKE) test-cov
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ Production checks complete - Ready to deploy!$(NC)"
	@echo "$(BLUE)════════════════════════════════════════════════════$(NC)"

cli-help:  ## Show CLI help
	deepfake-detector --help

cli-version:  ## Show version
	deepfake-detector --version
