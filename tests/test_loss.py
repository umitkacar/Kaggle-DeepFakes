"""Tests for loss functions."""

import numpy as np
import pytest
import tensorflow as tf

from deepfake_detector.model.loss import (
    ErrorMetric,
    l1_loss,
    l2_loss,
    leaf_l1_loss,
    leaf_l2_loss,
)


@pytest.mark.unit
def test_l1_loss_basic():
    """Test basic L1 loss computation."""
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[1.5, 2.5], [3.5, 4.5]])

    loss = l1_loss(x, y)

    assert loss.numpy() == pytest.approx(0.5, rel=1e-5)


@pytest.mark.unit
def test_l1_loss_with_mask():
    """Test L1 loss with masking."""
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[1.5, 2.5], [3.5, 4.5]])
    mask = tf.constant([[1.0], [0.0]])  # Only first sample

    loss = l1_loss(x, y, mask)

    assert loss.numpy() > 0


@pytest.mark.unit
def test_l2_loss_basic():
    """Test basic L2 loss computation."""
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[2.0, 3.0], [4.0, 5.0]])

    loss = l2_loss(x, y)

    assert loss.numpy() == pytest.approx(1.0, rel=1e-5)


@pytest.mark.unit
def test_leaf_l1_loss():
    """Test leaf node L1 loss."""
    batch_size = 4
    num_leaves = 8

    # Create random predictions for each leaf
    x_list = [tf.random.uniform((batch_size, 32, 32, 1)) for _ in range(num_leaves)]

    # Ground truth
    y = tf.random.uniform((batch_size, 32, 32, 1))

    # Create masks for leaf routing
    mask_list = [tf.random.uniform((batch_size, 2)) for _ in range(num_leaves)]

    # Compute loss
    loss = leaf_l1_loss(x_list, y, mask_list)

    assert loss.numpy() >= 0
    assert not np.isnan(loss.numpy())


@pytest.mark.unit
def test_error_metric():
    """Test ErrorMetric tracker."""
    metric = ErrorMetric()

    # Add training values
    metric(1.0, val=0)
    metric(2.0, val=0)
    metric(3.0, val=0)

    avg = metric(0.0, val=0)  # Just to get average
    assert avg > 0

    # Add validation values
    metric(0.5, val=1)
    metric(1.5, val=1)

    avg_val = metric(0.0, val=1)
    assert avg_val > 0

    # Reset
    metric.reset()
    assert metric.value == 0
    assert metric.step == 0


@pytest.mark.unit
def test_error_metric_running_average():
    """Test ErrorMetric running average calculation."""
    metric = ErrorMetric()

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_avg = sum(values) / len(values)

    for val in values:
        result = metric(val, val=0)

    assert result == pytest.approx(expected_avg, rel=1e-5)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_l1_loss_different_batch_sizes(batch_size):
    """Test L1 loss with different batch sizes."""
    x = tf.random.uniform((batch_size, 10))
    y = tf.random.uniform((batch_size, 10))

    loss = l1_loss(x, y)

    assert loss.numpy() >= 0
    assert not np.isnan(loss.numpy())


@pytest.mark.unit
def test_loss_gradient_flow():
    """Test that gradients flow through loss functions."""
    x = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[1.5, 2.5], [3.5, 4.5]])

    with tf.GradientTape() as tape:
        loss = l1_loss(x, y)

    gradients = tape.gradient(loss, x)

    assert gradients is not None
    assert gradients.numpy().shape == x.numpy().shape
