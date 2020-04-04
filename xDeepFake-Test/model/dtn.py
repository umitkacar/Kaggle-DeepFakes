
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.layers as layers

class Linear(layers.Layer):
    def __init__(self, idx, alpha, beta, input_dim=32):
        super(Linear, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        initializer0 = tf.zeros_initializer()
        self.v = tf.Variable(initial_value=initializer(shape=(1, input_dim), dtype='float32'),
                             trainable=True, name='tru/v/'+idx)
        self.mu = tf.Variable(initial_value=initializer0(shape=(1, input_dim), dtype='float32'),
                              trainable=True, name='tru/mu/'+idx)
        # training hyper-parameters
        self.alpha = alpha
        self.beta = beta
        # mean, eigenvalue and trace for each mini-batch
        self.mu_of_visit = 0
        self.eigenvalue = 0.
        self.trace = 0.

    def call(self, x, mask, training):
        norm_v = self.v / (tf.norm(self.v) + 1e-8)
        norm_v_t = tf.transpose(norm_v, [1, 0])
        num_of_visit = tf.reduce_sum(mask)

        if training and num_of_visit > 1:
            # use only the visiting samples
            index = tf.where(tf.greater(mask[:, 0], tf.constant(0.)))
            index_not = tf.where(tf.equal(mask[:, 0], tf.constant(0.)))
            x_sub = tf.gather_nd(x, index) - tf.stop_gradient(self.mu)
            x_not = tf.gather_nd(x, index_not)
            x_sub_t = tf.transpose(x_sub, [1, 0])

            # compute the covariance matrix, eigenvalue, and the trace
            covar = tf.matmul(x_sub_t, x_sub) / num_of_visit
            eigenvalue = tf.reshape(tf.matmul(tf.matmul(norm_v, covar), norm_v_t), [])
            trace = tf.linalg.trace(covar)
            # compute the route loss
            # print(tf.exp(-self.alpha * eigenvalue), self.beta * trace)
            route_loss = tf.exp(-self.alpha * eigenvalue) + self.beta * trace
            uniq_loss = -tf.reduce_mean(tf.square(tf.matmul(x_sub, norm_v_t))) + \
                         tf.reduce_mean(tf.square(tf.matmul(x_not, norm_v_t)))
            # compute mean and response for this batch
            self.mu_of_visit = tf.reduce_mean(x_sub, axis=0, keepdims=True)
            self.eigenvalue = eigenvalue
            self.trace = trace
            x -= tf.stop_gradient(self.mu_of_visit)
            route_value = tf.matmul(x, norm_v_t)
        else:
            self.mu_of_visit = self.mu
            self.eigenvalue = 0.
            self.trace = 0.
            x -= self.mu
            route_value = tf.matmul(x, norm_v_t)
            route_loss = 0.
            uniq_loss = 0.

        return route_value, route_loss, uniq_loss

class Downsample(tf.keras.Model):
    def __init__(self, filters, size, padding='SAME', apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv1 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=2,
                                   padding=padding,
                                   kernel_initializer=initializer,
                                   use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x

class Upsample(tf.keras.Model):
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x, training):
        x = self.up_conv(x)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x

class Conv(tf.keras.Model):
    def __init__(self, filters, size, stride=1, activation=True, padding='SAME', apply_batchnorm=True):
        super(Conv, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.activation = activation
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv1 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=stride,
                                   padding=padding,
                                   kernel_initializer=initializer,
                                   use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        return x

class Dense(tf.keras.Model):
    def __init__(self, filters, activation=True, apply_batchnorm=True, apply_dropout=False):
        super(Dense, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.activation = activation
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.dense = layers.Dense(filters,
                                  kernel_initializer=initializer,
                                  use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, training):
        x = self.dense(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        return x

class CRU(tf.keras.Model):

    def __init__(self, filters, size=3, stride=2, apply_batchnorm=True):
        super(CRU, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.stride = stride
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)

        self.conv1 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)
        self.conv2 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)
        self.conv3 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)
        self.conv4 = layers.Conv2D(filters,
                                   (size, size),
                                   strides=1,
                                   padding='SAME',
                                   kernel_initializer=initializer,
                                   use_bias=False)

        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        # first residual block
        _x = self.conv1(x)
        _x = self.batchnorm1(_x, training=training)
        _x = tf.nn.leaky_relu(_x)
        _x = self.conv2(_x)
        _x = self.batchnorm2(_x, training=training)
        _x  = x + _x
        x  = tf.nn.leaky_relu(_x)

        # second residual block
        _x = self.conv3(x)
        _x = self.batchnorm3(_x, training=training)
        _x = tf.nn.leaky_relu(_x)
        _x = self.conv4(_x)
        _x = self.batchnorm4(_x, training=training)
        _x = x + _x
        x = tf.nn.leaky_relu(_x)

        if self.stride > 1:
            x = tf.nn.max_pool(x, 3, 2, padding='SAME')
        return x

class TRU(tf.keras.Model):

    def __init__(self, filters, idx, alpha=1e-3, beta=1e-4, size=3, apply_batchnorm=True):
        super(TRU, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        # variables
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
        x_small_shape = x_small.shape
        x_flatten = self.flatten(tf.nn.avg_pool(x_small, ksize=3, strides=2, padding='SAME'))

        # PCA Projection
        route_value, route_loss, uniq_loss = self.project(x_flatten, mask, training=training)

        # Generate the splitting mask
        mask_l = mask * tf.cast(tf.greater_equal(route_value, tf.constant(0.)), tf.float32)
        mask_r = mask * tf.cast(tf.less(route_value, tf.constant(0.)), tf.float32)

        return [mask_l, mask_r], route_value, [route_loss, uniq_loss]

class SFL(tf.keras.Model):

    def __init__(self, filters, size=3, apply_batchnorm=True):
        super(SFL, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        # depth map
        self.cru1 = CRU(filters, size, stride=1)
        self.conv1 = Conv(2, size, activation=False, apply_batchnorm=False)

        # class
        self.conv2 = Downsample(filters*1, size)
        self.conv3 = Downsample(filters*1, size)
        self.conv4 = Downsample(filters*2, size)
        self.conv5 = Downsample(filters*4, 4, padding='VALID')
        self.flatten = layers.Flatten()
        self.fc1 = Dense(256)
        self.fc2 = Dense(1, activation=False, apply_batchnorm=False)

        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, training):
        # depth map branch
        xd = self.cru1(x)
        xd = self.conv1(xd)
        dmap = tf.nn.sigmoid(xd)
        # class branch
        x = self.conv2(x)  # 16*16*32
        x = self.conv3(x)  # 8*8*64
        x = self.conv4(x)  # 4*4*128
        x = self.conv5(x)  # 1*1*256
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        cls = self.fc2(x)
        return dmap, cls

############################################################
#  Deep Tree Network (DTN)
############################################################

class DTN(tf.keras.models.Model):
    def __init__(self, filters):
        super(DTN, self).__init__()
        
        TRU_PARAMETERS = {
        "alpha": 1e-3,
        "beta": 1e-2,
        "mu_update_rate": 1e-3,
        }
        
        layer = [1, 2, 4, 8, 16]
        self.conv1 = Conv(filters, 5, apply_batchnorm=False)
        # CRU
        self.cru0 = CRU(filters)
        self.cru1 = CRU(filters)
        self.cru2 = CRU(filters)
        self.cru3 = CRU(filters)
        self.cru4 = CRU(filters)
        self.cru5 = CRU(filters)
        self.cru6 = CRU(filters)
        # TRU
        alpha = TRU_PARAMETERS['alpha']
        beta = TRU_PARAMETERS['beta']
        self.tru0 = TRU(filters, '1', alpha, beta)
        self.tru1 = TRU(filters, '2', alpha, beta)
        self.tru2 = TRU(filters, '3', alpha, beta)
        self.tru3 = TRU(filters, '4', alpha, beta)
        self.tru4 = TRU(filters, '5', alpha, beta)
        self.tru5 = TRU(filters, '6', alpha, beta)
        self.tru6 = TRU(filters, '7', alpha, beta)
        # SFL
        self.sfl0 = SFL(filters)
        self.sfl1 = SFL(filters)
        self.sfl2 = SFL(filters)
        self.sfl3 = SFL(filters)
        self.sfl4 = SFL(filters)
        self.sfl5 = SFL(filters)
        self.sfl6 = SFL(filters)
        self.sfl7 = SFL(filters)

    @tf.function
    def call(self, x, label, training):
        if training:
            mask_spoof = label
            mask_live = 1 - label
        else:
            mask_spoof = tf.ones_like(label)
            mask_live = tf.zeros_like(label)
        ''' Tree Level 1 '''
        x = self.conv1(x, training)
        x_cru0 = self.cru0(x)
        x_tru0, route_value0, tru0_loss = self.tru0(x_cru0, mask_spoof, training)

        ''' Tree Level 2 '''
        x_cru00 = self.cru1(x_cru0, training)
        x_cru01 = self.cru2(x_cru0, training)
        x_tru00, route_value00, tru00_loss = self.tru1(x_cru00, x_tru0[0], training)
        x_tru01, route_value01, tru01_loss = self.tru2(x_cru01, x_tru0[1], training)

        ''' Tree Level 3 '''
        x_cru000 = self.cru3(x_cru00, training)
        x_cru001 = self.cru4(x_cru00, training)
        x_cru010 = self.cru5(x_cru01, training)
        x_cru011 = self.cru6(x_cru01, training)
        x_tru000, route_value000, tru000_loss = self.tru3(x_cru000, x_tru00[0], training)
        x_tru001, route_value001, tru001_loss = self.tru4(x_cru001, x_tru00[1], training)
        x_tru010, route_value010, tru010_loss = self.tru5(x_cru010, x_tru01[0], training)
        x_tru011, route_value011, tru011_loss = self.tru6(x_cru011, x_tru01[1], training)

        ''' Tree Level 4 '''
        map0, cls0 = self.sfl0(x_cru000, training)
        map1, cls1 = self.sfl1(x_cru000, training)
        map2, cls2 = self.sfl2(x_cru001, training)
        map3, cls3 = self.sfl3(x_cru001, training)
        map4, cls4 = self.sfl4(x_cru010, training)
        map5, cls5 = self.sfl5(x_cru010, training)
        map6, cls6 = self.sfl6(x_cru011, training)
        map7, cls7 = self.sfl7(x_cru011, training)
        ''' Output '''
        maps = [map0, map1, map2, map3, map4, map5, map6, map7]
        clss = [cls0, cls1, cls2, cls3, cls4, cls5, cls6, cls7]
        route_value = [route_value0, route_value00, route_value01,
                       route_value000, route_value001, route_value010, route_value011]
        x_tru0000 = tf.concat([x_tru000[0], mask_live], axis=1)
        x_tru0001 = tf.concat([x_tru000[1], mask_live], axis=1)
        x_tru0010 = tf.concat([x_tru001[0], mask_live], axis=1)
        x_tru0011 = tf.concat([x_tru001[1], mask_live], axis=1)
        x_tru0100 = tf.concat([x_tru010[0], mask_live], axis=1)
        x_tru0101 = tf.concat([x_tru010[1], mask_live], axis=1)
        x_tru0110 = tf.concat([x_tru011[0], mask_live], axis=1)
        x_tru0111 = tf.concat([x_tru011[1], mask_live], axis=1)
        leaf_node_mask = [x_tru0000, x_tru0001, x_tru0010, x_tru0011, x_tru0100, x_tru0101, x_tru0110, x_tru0111]

        return maps, clss, route_value, leaf_node_mask
