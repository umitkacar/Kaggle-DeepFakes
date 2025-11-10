# Production Setup Guide

> **Ultra-modern production deployment guide for DeepFake Detector**

## 🚀 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd Kaggle-DeepFakes

# Run automated setup
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## 📋 Prerequisites

- **Python:** 3.8 or higher
- **pip:** Latest version
- **Git:** For version control
- **GPU (Optional):** NVIDIA GPU with CUDA for training acceleration

## 🔧 Installation Methods

### Method 1: Automated Setup (Recommended)

```bash
./scripts/setup.sh
```

This script will:
- ✅ Check Python version
- ✅ Install all dependencies
- ✅ Set up pre-commit hooks
- ✅ Run validation checks
- ✅ Run test suite

### Method 2: Manual Setup

#### Step 1: Install Dependencies

```bash
# Install package with development dependencies
pip install -e ".[dev]"

# Or production dependencies only
pip install -e .
```

#### Step 2: Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

#### Step 3: Validate Installation

```bash
python3 scripts/validate.py
```

## 🧪 Testing

### Run All Tests

```bash
# Using pytest directly
pytest tests/

# Using hatch
hatch run test

# Parallel execution (faster)
hatch run test-fast
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run benchmarks
pytest -m benchmark
```

### Coverage Reports

```bash
# Terminal report
hatch run test-cov

# HTML report (opens in browser)
hatch run test-cov-html
open htmlcov/index.html

# XML report (for CI/CD)
hatch run test-cov-xml
```

## 🎨 Code Quality

### Formatting

```bash
# Auto-format all code
hatch run fmt

# Or run individually
black .
ruff --fix .
```

### Linting

```bash
# Run all linters
hatch run lint

# Individual tools
ruff .
black --check .
mypy src/deepfake_detector
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

### Configured Hooks

| Hook | Purpose | Auto-fix |
|------|---------|----------|
| **trailing-whitespace** | Remove trailing whitespace | ✅ |
| **end-of-file-fixer** | Ensure files end with newline | ✅ |
| **black** | Code formatter | ✅ |
| **ruff** | Fast linter | ✅ |
| **isort** | Import sorting | ✅ |
| **mypy** | Type checking | ❌ |
| **bandit** | Security checks | ❌ |
| **codespell** | Spell checker | ❌ |
| **check-yaml/json/toml** | Config validation | ❌ |

## 🛠️ Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature
```

### 2. Make Changes

Edit code, add tests, update docs.

### 3. Run Quality Checks

```bash
# Format code
hatch run fmt

# Run tests
hatch run test-fast

# Type checking
mypy src/deepfake_detector

# Or run everything
hatch run check
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: your feature description"
# Pre-commit hooks run automatically
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature
```

## 📦 Package Management

### Using Hatch (Recommended)

```bash
# Create virtual environment
hatch env create

# Run commands in environment
hatch run test
hatch run fmt
hatch run lint

# Shell into environment
hatch shell
```

### Available Hatch Scripts

| Command | Description |
|---------|-------------|
| `hatch run test` | Run all tests |
| `hatch run test-fast` | Run tests in parallel (faster) |
| `hatch run test-cov` | Run tests with coverage report |
| `hatch run test-cov-html` | Generate HTML coverage report |
| `hatch run fmt` | Format code (ruff + black) |
| `hatch run lint` | Lint code (ruff + black check + mypy) |
| `hatch run check` | Run all checks (fmt + lint + test) |
| `hatch run all` | Format + lint + test with coverage |

## 🎭 CLI Usage

### Installation Verification

```bash
# Check installation
deepfake-detector --version
dfd --version  # Short alias

# Show help
deepfake-detector --help
```

### Training

```bash
# Basic training
deepfake-detector train \
  -d ./data/fake \
  -d ./data/real \
  --epochs 100

# With validation data
deepfake-detector train \
  -d ./data/train/fake \
  -d ./data/train/real \
  --val-dir ./data/val/fake \
  --val-dir ./data/val/real \
  --epochs 100 \
  --batch-size 32

# Using config file
deepfake-detector train --config config.yaml
```

### Testing

```bash
# Test trained model
deepfake-detector test \
  -d ./data/test \
  -m ./logs/model.ckpt \
  -o results.csv
```

### Prediction

```bash
# Predict single image
deepfake-detector predict image.jpg -m model.ckpt

# With visualization
deepfake-detector predict image.jpg \
  -m model.ckpt \
  -o output.png \
  --visualize

# Custom threshold
deepfake-detector predict image.jpg \
  -m model.ckpt \
  --threshold 0.7
```

### Configuration

```bash
# Show current configuration
deepfake-detector config --show

# Generate config template
deepfake-detector config --generate config.yaml
```

## 🔍 Validation & Quality Assurance

### Pre-deployment Checklist

```bash
# 1. Run validation script
python3 scripts/validate.py

# 2. Run all tests with coverage
hatch run test-cov

# 3. Run linters
hatch run lint

# 4. Check for security issues
bandit -r src/

# 5. Run pre-commit on all files
pre-commit run --all-files
```

### Continuous Integration

The repository includes configurations for:
- ✅ GitHub Actions (see `docs/workflows/`)
- ✅ Pre-commit CI
- ✅ Automated testing
- ✅ Coverage reporting

## 🐛 Troubleshooting

### Import Errors

```bash
# Reinstall in editable mode
pip install -e ".[dev]"

# Verify installation
python3 -c "from deepfake_detector import __version__; print(__version__)"
```

### TensorFlow Issues

```bash
# CPU version
pip install tensorflow

# GPU version (requires CUDA)
pip install tensorflow-gpu

# Check installation
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

### Pre-commit Hook Failures

```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean

# Reinstall
pre-commit uninstall
pre-commit install
```

### Test Failures

```bash
# Verbose output
pytest -vv

# Stop on first failure
pytest -x

# Show local variables
pytest --showlocals

# Specific test
pytest tests/test_model.py::test_dtn_forward
```

## 📊 Code Quality Metrics

### Coverage Requirements

- **Minimum:** 80%
- **Target:** 90%+
- **Critical paths:** 100%

### Linting Configuration

- **Line length:** 100 characters
- **Ruff rules:** 30+ categories enabled
- **MyPy:** Strict mode with type hints
- **Bandit:** Security scanning enabled

## 🚢 Deployment

### Production Checklist

- [ ] All tests passing (100%)
- [ ] Coverage >= 80%
- [ ] No linting errors
- [ ] No security vulnerabilities
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version bumped

### Build Package

```bash
# Using hatch
hatch build

# Or setuptools
python3 -m build
```

### Install from Wheel

```bash
pip install dist/deepfake_detector-*.whl
```

## 📚 Additional Resources

- **Documentation:** See `docs/` directory
- **Examples:** See `examples/` directory
- **API Reference:** Generated with Sphinx
- **Contributing:** See `CONTRIBUTING.md`
- **Security:** See `SECURITY.md`

## 🆘 Support

If you encounter issues:

1. Check this guide
2. Run validation: `python3 scripts/validate.py`
3. Check logs in `./logs/`
4. Open an issue on GitHub

---

**Last Updated:** 2025-11-09
**Version:** 1.0.0
**Maintained by:** Umit Kacar
