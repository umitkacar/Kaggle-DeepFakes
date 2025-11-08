"""Tests for model components."""

import pytest
import tensorflow as tf

from deepfake_detector.core.config import Settings
from deepfake_detector.model import DTN, DTNModel


@pytest.mark.unit
def test_dtn_initialization(sample_config):
    """Test DTN model initialization."""
    model = DTN(filters=32, config=sample_config)
    assert model is not None
    assert model.config == sample_config


@pytest.mark.unit
def test_dtn_model_wrapper(sample_config):
    """Test DTNModel wrapper initialization."""
    model = DTNModel(sample_config)
    assert model.config == sample_config
    assert model.dtn is not None
    assert model.optimizer is not None


@pytest.mark.unit
def test_dtn_forward_pass(sample_config, sample_batch_data):
    """Test DTN forward pass."""
    images, dmaps, labels = sample_batch_data

    model = DTN(filters=32, config=sample_config)

    # Forward pass
    outputs = model(images, labels, training=True)

    # Check outputs
    assert len(outputs) == 8  # maps, clss, routes, masks, losses, mu, eigen, trace
    maps, clss, routes, masks, losses, mu, eigen, trace = outputs

    # Verify output shapes
    assert len(maps) == 8  # 8 leaf nodes
    assert len(clss) == 8  # 8 classifications
    assert len(masks) == 8  # 8 leaf masks


@pytest.mark.unit
def test_dtn_inference_mode(sample_config, sample_batch_data):
    """Test DTN in inference mode."""
    images, dmaps, labels = sample_batch_data

    model = DTN(filters=32, config=sample_config)

    # Inference mode
    outputs = model(images, labels, training=False)

    # Check outputs (no training info)
    assert len(outputs) == 4  # maps, clss, routes, masks
    maps, clss, routes, masks = outputs

    assert len(maps) == 8
    assert len(clss) == 8


@pytest.mark.slow
def test_model_checkpoint_save_restore(sample_config, tmp_path):
    """Test model checkpoint save and restore."""
    # Set checkpoint directory
    sample_config.log_dir = tmp_path / "checkpoints"

    # Create and compile model
    model = DTNModel(sample_config)
    model.compile()

    # Save checkpoint
    model.save_checkpoint(epoch=1)

    # Check checkpoint was created
    checkpoints = list(sample_config.log_dir.glob("ckpt-*"))
    assert len(checkpoints) > 0


@pytest.mark.benchmark
def test_dtn_forward_pass_performance(sample_config, sample_batch_data, benchmark):
    """Benchmark DTN forward pass performance."""
    images, dmaps, labels = sample_batch_data
    model = DTN(filters=32, config=sample_config)

    def run_forward():
        return model(images, labels, training=True)

    # Benchmark the forward pass
    result = benchmark(run_forward)
    assert result is not None
