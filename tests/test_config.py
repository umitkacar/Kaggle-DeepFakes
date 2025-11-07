"""Tests for configuration module."""

import pytest
from pathlib import Path
from deepfake_detector.core.config import Settings, ModelConfig, TrainingConfig, TRUParameters


def test_default_settings():
    """Test default settings initialization."""
    settings = Settings()
    assert settings.log_dir == Path("./logs/dtn")
    assert settings.mode == "training"
    assert settings.gpu_usage == 1


def test_model_config():
    """Test model configuration."""
    model_config = ModelConfig()
    assert model_config.image_size == 256
    assert model_config.map_size == 64
    assert model_config.filters == 32


def test_training_config():
    """Test training configuration."""
    training_config = TrainingConfig()
    assert training_config.batch_size == 20
    assert training_config.learning_rate == 0.0001
    assert training_config.max_epochs == 70


def test_tru_parameters():
    """Test TRU parameters."""
    tru_params = TRUParameters()
    assert tru_params.alpha == 1e-3
    assert tru_params.beta == 1e-2
    assert tru_params.mu_update_rate == 1e-3


def test_settings_validation():
    """Test settings validation."""
    settings = Settings()

    # Test batch size validation
    settings.training.batch_size = 32
    assert settings.training.batch_size == 32

    # Test learning rate validation
    settings.training.learning_rate = 0.001
    assert settings.training.learning_rate == 0.001


def test_settings_display(capsys):
    """Test settings display method."""
    settings = Settings()
    settings.display()
    # Just check it doesn't crash - Rich output is hard to test
