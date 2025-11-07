# Contributing to DeepFake Detector

Thank you for your interest in contributing to DeepFake Detector! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Kaggle-DeepFakes.git
   cd Kaggle-DeepFakes
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   # Or using make
   make install-dev
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   # Or using make
   make setup-hooks
   ```

## 🔧 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow these guidelines:

- Write clean, readable code
- Add docstrings to functions and classes
- Include type hints
- Follow PEP 8 style guide (enforced by Black and Ruff)

### 3. Run Quality Checks

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test

# Or run everything
make format && make lint && make test
```

### 4. Commit Your Changes

We use conventional commits:

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in model"
git commit -m "docs: update README"
git commit -m "test: add tests for config"
```

**Commit types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📝 Code Style

### Python Style

We use the following tools (configured in `pyproject.toml`):

- **Black**: Code formatting (line length: 100)
- **Ruff**: Fast linting
- **MyPy**: Type checking
- **isort**: Import sorting

All of these run automatically via pre-commit hooks.

### Example Code Style

```python
"""Module docstring explaining the purpose."""

from typing import List, Optional

import numpy as np
import tensorflow as tf
from loguru import logger

from deepfake_detector.core.config import Settings


class MyClass:
    """Class docstring.

    Args:
        param1: Description of param1
        param2: Description of param2
    """

    def __init__(self, param1: str, param2: int = 10):
        self.param1 = param1
        self.param2 = param2

    def my_method(self, input_data: np.ndarray) -> Optional[tf.Tensor]:
        """Method docstring.

        Args:
            input_data: Input array

        Returns:
            Processed tensor or None
        """
        if input_data.size == 0:
            logger.warning("Empty input data")
            return None

        result = tf.constant(input_data)
        return result
```

## 🧪 Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use pytest fixtures for reusable components
- Aim for >80% code coverage

### Example Test

```python
"""Tests for my module."""

import pytest
from deepfake_detector.core.config import Settings


def test_default_settings():
    """Test default settings initialization."""
    settings = Settings()
    assert settings.mode == "training"


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration."""
    return Settings(mode="testing")


def test_custom_settings(sample_config):
    """Test custom settings."""
    assert sample_config.mode == "testing"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/deepfake_detector

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::test_default_settings

# Using make
make test
make test-verbose
```

## 📚 Documentation

### Docstrings

Use Google-style docstrings:

```python
def train_model(data_dir: Path, epochs: int = 100) -> Model:
    """Train the DeepFake detection model.

    Args:
        data_dir: Path to training data directory
        epochs: Number of training epochs

    Returns:
        Trained model instance

    Raises:
        ValueError: If data_dir doesn't exist
        RuntimeError: If training fails

    Example:
        >>> model = train_model(Path("./data"), epochs=50)
        >>> model.evaluate()
    """
    ...
```

### Type Hints

Always include type hints:

```python
from typing import Dict, List, Optional, Union
from pathlib import Path

def process_images(
    image_paths: List[Path],
    batch_size: int = 32,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Process a list of images."""
    ...
```

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal code to reproduce the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, package versions
6. **Logs**: Relevant error messages or logs

## 💡 Suggesting Features

When suggesting features, include:

1. **Use case**: Why is this feature needed?
2. **Description**: Detailed description of the feature
3. **Examples**: Code examples showing how it would work
4. **Alternatives**: Alternative solutions you've considered

## 📋 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure all tests pass** (`make test`)
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts

## 🏗️ Project Structure

```
src/deepfake_detector/
├── core/           # Core functionality (config, logging)
├── model/          # Model architecture (layers, loss, DTN)
├── training/       # Training logic
├── inference/      # Inference/prediction
└── cli.py          # CLI interface
```

## 🔄 Using Hatch

We use Hatch for project management:

```bash
# Create environment
hatch env create

# Run tests
hatch run test

# Run linters
hatch run check

# Format code
hatch run fmt

# Build package
hatch build
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Documentation**: Check README.md and code docstrings

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints

## 🙏 Thank You!

Your contributions make this project better. We appreciate your time and effort!

---

**Happy Contributing! 🎉**
