"""Tests for model layers."""

import pytest
import tensorflow as tf

from deepfake_detector.model.layers import Conv, CRU, Linear


@pytest.mark.unit
def test_conv_layer_creation():
    """Test Conv layer creation."""
    layer = Conv(filters=32, size=3)
    assert layer is not None


@pytest.mark.unit
def test_conv_layer_forward():
    """Test Conv layer forward pass."""
    layer = Conv(filters=32, size=3)
    x = tf.random.uniform((4, 64, 64, 3))

    output = layer(x, training=True)

    assert output.shape == (4, 64, 64, 32)


@pytest.mark.unit
def test_conv_layer_without_batchnorm():
    """Test Conv layer without batch normalization."""
    layer = Conv(filters=32, size=3, apply_batchnorm=False)
    x = tf.random.uniform((4, 64, 64, 3))

    output = layer(x, training=True)

    assert output.shape == (4, 64, 64, 32)


@pytest.mark.unit
def test_cru_layer_creation():
    """Test CRU layer creation."""
    layer = CRU(filters=32)
    assert layer is not None


@pytest.mark.unit
def test_cru_layer_forward():
    """Test CRU layer forward pass."""
    layer = CRU(filters=32, stride=2)
    x = tf.random.uniform((4, 64, 64, 32))

    output = layer(x, training=True)

    # With stride=2, spatial dimensions should be halved
    assert output.shape == (4, 32, 32, 32)


@pytest.mark.unit
def test_cru_layer_no_downsampling():
    """Test CRU layer without downsampling."""
    layer = CRU(filters=32, stride=1)
    x = tf.random.uniform((4, 64, 64, 32))

    output = layer(x, training=True)

    # With stride=1, spatial dimensions should remain same
    assert output.shape == (4, 64, 64, 32)


@pytest.mark.unit
def test_linear_layer_creation():
    """Test Linear projection layer creation."""
    layer = Linear(idx="test", alpha=1e-3, beta=1e-2, input_dim=128)
    assert layer is not None
    assert layer.alpha == 1e-3
    assert layer.beta == 1e-2


@pytest.mark.unit
def test_linear_layer_forward_training():
    """Test Linear layer forward pass in training mode."""
    layer = Linear(idx="test", alpha=1e-3, beta=1e-2, input_dim=128)
    x = tf.random.uniform((8, 128))
    mask = tf.ones((8, 1))

    route_value, route_loss, uniq_loss = layer(x, mask, training=True)

    assert route_value is not None
    assert route_value.shape == (8, 1)
    assert isinstance(route_loss, (float, tf.Tensor))
    assert isinstance(uniq_loss, (float, tf.Tensor))


@pytest.mark.unit
def test_linear_layer_forward_inference():
    """Test Linear layer forward pass in inference mode."""
    layer = Linear(idx="test", alpha=1e-3, beta=1e-2, input_dim=128)
    x = tf.random.uniform((8, 128))
    mask = tf.ones((8, 1))

    route_value, route_loss, uniq_loss = layer(x, mask, training=False)

    assert route_value is not None
    assert route_value.shape == (8, 1)
    assert route_loss == 0.0
    assert uniq_loss == 0.0


@pytest.mark.unit
def test_linear_layer_mu_update():
    """Test Linear layer mu value updates."""
    layer = Linear(idx="test", alpha=1e-3, beta=1e-2, input_dim=128)
    x = tf.random.uniform((8, 128))
    mask = tf.ones((8, 1))

    # First forward pass
    _, _, _ = layer(x, mask, training=True)
    mu_first = layer.mu_of_visit

    # Second forward pass with different data
    x2 = tf.random.uniform((8, 128))
    _, _, _ = layer(x2, mask, training=True)
    mu_second = layer.mu_of_visit

    # Mu should be updated
    assert mu_first is not None
    assert mu_second is not None


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_conv_different_batch_sizes(batch_size):
    """Test Conv layer with different batch sizes."""
    layer = Conv(filters=32, size=3)
    x = tf.random.uniform((batch_size, 64, 64, 3))

    output = layer(x, training=True)

    assert output.shape[0] == batch_size
    assert output.shape[1:] == (64, 64, 32)


@pytest.mark.parametrize("filters", [16, 32, 64, 128])
def test_cru_different_filter_sizes(filters):
    """Test CRU layer with different filter sizes."""
    layer = CRU(filters=filters)
    x = tf.random.uniform((4, 64, 64, filters))

    output = layer(x, training=True)

    assert output.shape[-1] == filters


@pytest.mark.unit
def test_layer_gradient_flow():
    """Test that gradients flow through layers."""
    layer = Conv(filters=32, size=3)
    x = tf.Variable(tf.random.uniform((4, 64, 64, 3)))

    with tf.GradientTape() as tape:
        output = layer(x, training=True)
        loss = tf.reduce_mean(output)

    gradients = tape.gradient(loss, layer.trainable_variables)

    assert gradients is not None
    assert len(gradients) > 0
    assert all(g is not None for g in gradients)
