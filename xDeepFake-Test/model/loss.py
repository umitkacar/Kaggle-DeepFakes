
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

def l1_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.reduce_mean(tf.reshape(tf.abs(x-y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(tf.abs(x-y))
    return loss


def l2_loss(x, y, mask=None):
    xshape = x.shape
    if mask is None:
        loss = tf.reduce_mean(tf.reshape(tf.square(x-y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(tf.square(x-y))
    return loss


def leaf_l1_loss(xlist, y, masklist):
    loss_list = []
    xshape = xlist[0].shape
    for x, mask in zip(xlist, masklist):
        loss = tf.reduce_mean(tf.reshape(tf.abs(x-y), [xshape[0], -1]), axis=1)
        # tag of spoof
        tag = tf.reduce_sum(mask[:, 0])
        tag = tag / (tag + 1e-8)
        # spoof
        spoof_loss = tf.reduce_sum(loss * mask[:, 0]) / (tf.reduce_sum(mask[:, 0]) + 1e-8)

        # live
        live_loss = tf.reduce_sum(loss * mask[:, 1]) / (tf.reduce_sum(mask[:, 1]) + 1e-8)

        total_loss = (spoof_loss + live_loss)/2
        loss_list.append(total_loss*tag)
    loss = tf.reduce_mean(loss_list)
    return loss


def leaf_l2_loss(xlist, y, masklist):
    loss_list = []
    for x, mask in zip(xlist, masklist):
        xshape = x.shape
        print(x.shape, y.shape, mask.shape)
        input()
        # spoof
        spoof_loss = tf.reduce_mean(tf.reshape(tf.square(x-y), [xshape[0], -1]), axis=1)
        spoof_loss = tf.reduce_sum(loss * mask[:, 0]) / (tf.reduce_sum(mask[:, 0]) + 1e-8)
        # live
        live_loss = tf.reduce_mean(tf.reshape(tf.square(x - y), [xshape[0], -1]), axis=1)
        live_loss = tf.reduce_sum(loss * mask[:, 1]) / (tf.reduce_sum(mask[:, 1]) + 1e-8)
        loss = spoof_loss + live_loss
        loss_list.append(loss)

    return loss

def leaf_l1_score(xlist, masklist, ch=None):
    loss_list = []
    xshape = xlist[0].shape
    scores = []
    for x, mask in zip(xlist, masklist):
        if ch is not None:
            score = tf.reduce_mean(tf.reshape(tf.abs(x[:, :, :, ch]), [xshape[0], -1]), axis=1)
        else:
            score = tf.reduce_mean(tf.reshape(tf.abs(x), [xshape[0], -1]), axis=1)
        spoof_score = score * mask[:, 0]
        scores.append(spoof_score)
    loss = np.sum(np.stack(scores, axis=1), axis=1)
    return loss
