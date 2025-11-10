"""
Pytest configuration and fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import sys
from pathlib import Path
from typing import Generator

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "benchmark: Benchmark performance tests")


def pytest_addoption(parser: Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


def pytest_collection_modifyitems(config: Config, items: list) -> None:
    """Modify test collection to handle custom markers."""
    if config.getoption("--run-slow"):
        # Run all tests if --run-slow is specified
        return

    skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def sample_config():
    """Provide a sample Settings configuration for tests."""
    from deepfake_detector.core.config import Settings

    return Settings(
        mode="testing",
        log_dir=Path("./test_logs"),
        verbose=True,
    )


@pytest.fixture
def mock_image_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary image file for testing."""
    import numpy as np
    from PIL import Image

    # Create a random RGB image
    image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, "RGB")

    # Save to temp file
    image_path = tmp_path / "test_image.jpg"
    image.save(image_path)

    yield image_path

    # Cleanup is automatic with tmp_path


@pytest.fixture
def mock_model_checkpoint(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock model checkpoint for testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create a dummy checkpoint file
    checkpoint_file = checkpoint_dir / "checkpoint-1"
    checkpoint_file.write_text("mock checkpoint")

    yield checkpoint_file

    # Cleanup is automatic with tmp_path


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    import logging

    # Remove all handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    yield

    # Cleanup
    for handler in root.handlers[:]:
        root.removeHandler(handler)


@pytest.fixture
def sample_batch_data():
    """Provide sample batch data for training tests."""
    import numpy as np
    import tensorflow as tf

    batch_size = 4
    image_size = 256
    map_size = 64

    # Create random images (RGB + HSV = 6 channels)
    images = np.random.rand(batch_size, image_size, image_size, 6).astype(np.float32)

    # Create random depth maps
    dmaps = np.random.rand(batch_size, map_size, map_size, 1).astype(np.float32)

    # Create random labels (0 or 1)
    labels = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)

    return (
        tf.constant(images),
        tf.constant(dmaps),
        tf.constant(labels),
    )


@pytest.fixture
def performance_timer():
    """Fixture to measure execution time."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self):
            """Get elapsed time in seconds."""
            if self.end_time is None:
                return time.perf_counter() - self.start_time
            return self.end_time - self.start_time

    return Timer
