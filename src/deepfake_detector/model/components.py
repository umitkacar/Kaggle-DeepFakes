"""
TRU (Tree Routing Unit) and SFL (Supervised Feature Learning) layers.

Additional custom layers for the Deep Tree Network architecture.
"""

import tensorflow as tf
from tensorflow.keras import layers

from deepfake_detector.model.layers import Conv, CRU, Linear


class Downsample(tf.keras.Model):
    """Downsampling layer with convolution."""

    def __init__(self, filters: int, size: int, padding: str = "SAME", apply_batchnorm: bool = True):
        super().__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0.0, 0.02)

        self.conv1 = layers.Conv2D(
            int(filters),
            (size, size),
            strides=2,
            padding=padding,
            kernel_initializer=initializer,
            use_bias=False,
        )
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class TRU(tf.keras.Model):
    """Tree Routing Unit - routes samples through the decision tree."""

    def __init__(self, filters: int, idx: str, alpha: float = 1e-3, beta: float = 1e-4, size: int = 3):
        super().__init__()
        self.conv1 = Downsample(filters, size)
        self.conv2 = Downsample(filters, size)
        self.conv3 = Downsample(filters, size)
        self.flatten = layers.Flatten()
        self.project = Linear(idx, alpha, beta, input_dim=2048)

    def call(self, x, mask, training):
        # Downsampling
        x_small = self.conv1(x, training=training)
        depth = 0
        if x_small.shape[1] > 16:
            x_small = self.conv2(x_small, training=training)
            depth += 1
            if x_small.shape[1] > 16:
                x_small = self.conv3(x_small, training=training)
                depth += 1

        x_flatten = self.flatten(tf.nn.avg_pool(x_small, ksize=3, strides=2, padding="SAME"))

        # PCA Projection
        route_value, route_loss, uniq_loss = self.project(x_flatten, mask, training=training)

        # Generate splitting mask
        mask_l = mask * tf.cast(tf.greater_equal(route_value, 0.0), tf.float32)
        mask_r = mask * tf.cast(tf.less(route_value, 0.0), tf.float32)

        return [mask_l, mask_r], route_value, [route_loss, uniq_loss]


class SFL(tf.keras.Model):
    """Supervised Feature Learning - generates depth map and classification."""

    def __init__(self, filters: int, size: int = 3):
        super().__init__()
        # Depth map branch
        self.cru1 = CRU(filters, size, stride=1)
        self.conv1 = Conv(2, size, activation=False, apply_batchnorm=False)

        # Classification branch
        self.conv2 = Downsample(filters * 1, size)
        self.conv3 = Downsample(filters * 1, size)
        self.conv4 = Downsample(filters * 2, size)
        self.conv5 = Downsample(filters * 4, 4, padding="VALID")
        self.flatten = layers.Flatten()

        # Dense layers
        self.fc1 = layers.Dense(
            256,
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            use_bias=False,
        )
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(
            1,
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            use_bias=False,
        )
        self.dropout = layers.Dropout(0.3)

    def call(self, x, training):
        # Depth map branch
        xd = self.cru1(x, training=training)
        xd = self.conv1(xd, training=training)
        dmap = tf.nn.sigmoid(xd)

        # Classification branch
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.leaky_relu(x)
        cls = self.fc2(x)

        return dmap, cls
