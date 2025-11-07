"""Custom layers for Deep Tree Network."""

import tensorflow as tf
from tensorflow.keras import layers


class Linear(layers.Layer):
    """Linear projection layer for TRU."""

    def __init__(self, idx: str, alpha: float, beta: float, input_dim: int = 32):
        super().__init__()
        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.v = tf.Variable(
            initializer(shape=(1, input_dim), dtype="float32"),
            trainable=True,
            name=f"tru/v/{idx}",
        )
        self.mu = tf.Variable(
            tf.zeros((1, input_dim), dtype="float32"),
            trainable=True,
            name=f"tru/mu/{idx}",
        )
        self.alpha = alpha
        self.beta = beta
        self.mu_of_visit = 0
        self.eigenvalue = 0.0
        self.trace = 0.0

    def call(self, x, mask, training):
        norm_v = self.v / (tf.norm(self.v) + 1e-8)
        norm_v_t = tf.transpose(norm_v, [1, 0])
        num_of_visit = tf.reduce_sum(mask)

        if training and num_of_visit > 1:
            index = tf.where(tf.greater(mask[:, 0], 0.0))
            index_not = tf.where(tf.equal(mask[:, 0], 0.0))
            x_sub = tf.gather_nd(x, index) - tf.stop_gradient(self.mu)
            x_not = tf.gather_nd(x, index_not)
            x_sub_t = tf.transpose(x_sub, [1, 0])

            covar = tf.matmul(x_sub_t, x_sub) / num_of_visit
            eigenvalue = tf.reshape(tf.matmul(tf.matmul(norm_v, covar), norm_v_t), [])
            trace = tf.linalg.trace(covar)
            route_loss = tf.exp(-self.alpha * eigenvalue) + self.beta * trace
            uniq_loss = -tf.reduce_mean(tf.square(tf.matmul(x_sub, norm_v_t))) + tf.reduce_mean(
                tf.square(tf.matmul(x_not, norm_v_t))
            )

            self.mu_of_visit = tf.reduce_mean(x_sub, axis=0, keepdims=True)
            self.eigenvalue = eigenvalue
            self.trace = trace
            x -= tf.stop_gradient(self.mu_of_visit)
            route_value = tf.matmul(x, norm_v_t)
        else:
            self.mu_of_visit = self.mu
            self.eigenvalue = 0.0
            self.trace = 0.0
            x -= self.mu
            route_value = tf.matmul(x, norm_v_t)
            route_loss = 0.0
            uniq_loss = 0.0

        return route_value, route_loss, uniq_loss


class Conv(tf.keras.Model):
    """Convolutional layer with optional batch normalization."""

    def __init__(self, filters, size, stride=1, activation=True, apply_batchnorm=True):
        super().__init__()
        self.apply_batchnorm = apply_batchnorm
        self.activation = activation
        self.conv1 = layers.Conv2D(
            filters,
            (size, size),
            strides=stride,
            padding="SAME",
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            use_bias=False,
        )
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        return x


class CRU(tf.keras.Model):
    """Convolutional Routing Unit with residual connections."""

    def __init__(self, filters, size=3, stride=2):
        super().__init__()
        self.stride = stride
        init = tf.random_normal_initializer(0.0, 0.02)

        self.conv1 = layers.Conv2D(filters, (size, size), strides=1, padding="SAME", kernel_initializer=init, use_bias=False)
        self.conv2 = layers.Conv2D(filters, (size, size), strides=1, padding="SAME", kernel_initializer=init, use_bias=False)
        self.conv3 = layers.Conv2D(filters, (size, size), strides=1, padding="SAME", kernel_initializer=init, use_bias=False)
        self.conv4 = layers.Conv2D(filters, (size, size), strides=1, padding="SAME", kernel_initializer=init, use_bias=False)

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()

    def call(self, x, training):
        _x = self.bn1(self.conv1(x), training=training)
        _x = tf.nn.leaky_relu(_x)
        _x = self.bn2(self.conv2(_x), training=training)
        x = tf.nn.leaky_relu(x + _x)

        _x = self.bn3(self.conv3(x), training=training)
        _x = tf.nn.leaky_relu(_x)
        _x = self.bn4(self.conv4(_x), training=training)
        x = tf.nn.leaky_relu(x + _x)

        if self.stride > 1:
            x = tf.nn.max_pool(x, 3, 2, padding="SAME")
        return x
