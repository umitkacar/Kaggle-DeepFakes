"""
Loss functions for Deep Tree Network.

Implements specialized loss functions for depth map prediction and
tree-based classification with leaf node routing.
"""

from typing import List

import tensorflow as tf
import numpy as np


def l1_loss(x: tf.Tensor, y: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
    """
    Compute L1 (MAE) loss between tensors.

    Args:
        x: Predicted tensor
        y: Ground truth tensor
        mask: Optional mask tensor

    Returns:
        L1 loss value
    """
    x_shape = x.shape
    if mask is not None:
        loss = tf.reduce_mean(tf.reshape(tf.abs(x - y), [x_shape[0], -1]), axis=1, keepdims=True)
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def l2_loss(x: tf.Tensor, y: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
    """
    Compute L2 (MSE) loss between tensors.

    Args:
        x: Predicted tensor
        y: Ground truth tensor
        mask: Optional mask tensor

    Returns:
        L2 loss value
    """
    x_shape = x.shape
    if mask is not None:
        loss = tf.reduce_mean(
            tf.reshape(tf.square(x - y), [x_shape[0], -1]), axis=1, keepdims=True
        )
        loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.reduce_mean(tf.square(x - y))
    return loss


def leaf_l1_loss(
    x_list: List[tf.Tensor],
    y: tf.Tensor,
    mask_list: List[tf.Tensor],
) -> tf.Tensor:
    """
    Compute L1 loss for leaf nodes in the tree structure.

    This loss function is specialized for the Deep Tree Network architecture,
    computing separate losses for spoof and live samples at each leaf node.

    Args:
        x_list: List of predicted tensors from leaf nodes
        y: Ground truth tensor
        mask_list: List of mask tensors for routing

    Returns:
        Averaged leaf node L1 loss
    """
    loss_list = []
    x_shape = x_list[0].shape

    for x, mask in zip(x_list, mask_list):
        # Compute pixel-wise L1 loss
        loss = tf.reduce_mean(tf.reshape(tf.abs(x - y), [x_shape[0], -1]), axis=1)

        # Tag of spoof (1 if any spoof samples, 0 otherwise)
        tag = tf.reduce_sum(mask[:, 0])
        tag = tag / (tag + 1e-8)

        # Spoof loss: loss for fake samples
        spoof_loss = tf.reduce_sum(loss * mask[:, 0]) / (tf.reduce_sum(mask[:, 0]) + 1e-8)

        # Live loss: loss for real samples
        live_loss = tf.reduce_sum(loss * mask[:, 1]) / (tf.reduce_sum(mask[:, 1]) + 1e-8)

        # Average of both
        total_loss = (spoof_loss + live_loss) / 2
        loss_list.append(total_loss * tag)

    # Average across all leaf nodes
    return tf.reduce_mean(loss_list)


def leaf_l2_loss(
    x_list: List[tf.Tensor],
    y: tf.Tensor,
    mask_list: List[tf.Tensor],
) -> tf.Tensor:
    """
    Compute L2 loss for leaf nodes in the tree structure.

    Args:
        x_list: List of predicted tensors from leaf nodes
        y: Ground truth tensor
        mask_list: List of mask tensors for routing

    Returns:
        Averaged leaf node L2 loss
    """
    loss_list = []

    for x, mask in zip(x_list, mask_list):
        x_shape = x.shape

        # Spoof loss
        spoof_loss = tf.reduce_mean(
            tf.reshape(tf.square(x - y), [x_shape[0], -1]), axis=1
        )
        spoof_loss = tf.reduce_sum(spoof_loss * mask[:, 0]) / (
            tf.reduce_sum(mask[:, 0]) + 1e-8
        )

        # Live loss
        live_loss = tf.reduce_mean(tf.reshape(tf.square(x - y), [x_shape[0], -1]), axis=1)
        live_loss = tf.reduce_sum(live_loss * mask[:, 1]) / (tf.reduce_sum(mask[:, 1]) + 1e-8)

        loss = spoof_loss + live_loss
        loss_list.append(loss)

    return tf.reduce_mean(loss_list)


def leaf_l1_score(
    x_list: List[tf.Tensor],
    mask_list: List[tf.Tensor],
    ch: int = None,
) -> np.ndarray:
    """
    Compute prediction scores for leaf nodes.

    Args:
        x_list: List of predicted tensors from leaf nodes
        mask_list: List of mask tensors for routing
        ch: Optional channel index to compute score on

    Returns:
        Array of scores per sample
    """
    scores = []
    x_shape = x_list[0].shape

    for x, mask in zip(x_list, mask_list):
        if ch is not None:
            score = tf.reduce_mean(
                tf.reshape(tf.abs(x[:, :, :, ch]), [x_shape[0], -1]), axis=1
            )
        else:
            score = tf.reduce_mean(tf.reshape(tf.abs(x), [x_shape[0], -1]), axis=1)

        spoof_score = score * mask[:, 0]
        scores.append(spoof_score)

    # Sum scores across all leaf nodes
    return np.sum(np.stack(scores, axis=1), axis=1)


class ErrorMetric:
    """
    Error metric tracker for training and validation.

    Tracks running averages of metrics during training.
    """

    def __init__(self):
        """Initialize metric tracker."""
        self.value = 0.0
        self.value_val = 0.0
        self.step = 0
        self.step_val = 0

    def __call__(self, update: float, val: int = 0) -> float:
        """
        Update and return metric value.

        Args:
            update: Value to add
            val: If 1, update validation metric; else training metric

        Returns:
            Current average value
        """
        if val == 1:
            self.value_val += update
            self.step_val += 1
            return self.value_val / self.step_val
        else:
            self.value += update
            self.step += 1
            return self.value / self.step

    def reset(self) -> None:
        """Reset all metric values."""
        self.value = 0.0
        self.value_val = 0.0
        self.step = 0
        self.step_val = 0
