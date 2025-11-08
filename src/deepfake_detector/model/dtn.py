"""
Deep Tree Network (DTN) implementation.

Main model architecture for deepfake detection using hierarchical tree routing.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from deepfake_detector.core.config import Settings
from deepfake_detector.model.components import SFL, TRU
from deepfake_detector.model.layers import Conv, CRU
from deepfake_detector.model.loss import ErrorMetric, leaf_l1_loss


class DTN(tf.keras.Model):
    """
    Deep Tree Network for DeepFake Detection.

    Hierarchical tree structure with routing units for feature classification.
    """

    def __init__(self, filters: int, config: Settings):
        super().__init__()
        self.config = config

        # Initial convolution
        self.conv1 = Conv(filters, 5, apply_batchnorm=False)

        # CRU (Convolutional Routing Units)
        self.cru0 = CRU(filters)
        self.cru1 = CRU(filters)
        self.cru2 = CRU(filters)
        self.cru3 = CRU(filters)
        self.cru4 = CRU(filters)
        self.cru5 = CRU(filters)
        self.cru6 = CRU(filters)

        # TRU (Tree Routing Units)
        alpha = config.model.tru_parameters.alpha
        beta = config.model.tru_parameters.beta
        self.tru0 = TRU(filters, "1", alpha, beta)
        self.tru1 = TRU(filters, "2", alpha, beta)
        self.tru2 = TRU(filters, "3", alpha, beta)
        self.tru3 = TRU(filters, "4", alpha, beta)
        self.tru4 = TRU(filters, "5", alpha, beta)
        self.tru5 = TRU(filters, "6", alpha, beta)
        self.tru6 = TRU(filters, "7", alpha, beta)

        # SFL (Supervised Feature Learning) - 8 leaf nodes
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
        """
        Forward pass through the network.

        Args:
            x: Input images
            label: Ground truth labels
            training: Training mode flag

        Returns:
            Depth maps, classifications, routing values, and leaf masks
        """
        if training:
            mask_spoof = label
            mask_live = 1 - label
        else:
            mask_spoof = tf.ones_like(label)
            mask_live = tf.zeros_like(label)

        # Tree Level 1
        x = self.conv1(x, training)
        x_cru0 = self.cru0(x, training=training)
        x_tru0, route_value0, tru0_loss = self.tru0(x_cru0, mask_spoof, training)

        # Tree Level 2
        x_cru00 = self.cru1(x_cru0, training)
        x_cru01 = self.cru2(x_cru0, training)
        x_tru00, route_value00, tru00_loss = self.tru1(x_cru00, x_tru0[0], training)
        x_tru01, route_value01, tru01_loss = self.tru2(x_cru01, x_tru0[1], training)

        # Tree Level 3
        x_cru000 = self.cru3(x_cru00, training)
        x_cru001 = self.cru4(x_cru00, training)
        x_cru010 = self.cru5(x_cru01, training)
        x_cru011 = self.cru6(x_cru01, training)
        x_tru000, route_value000, tru000_loss = self.tru3(x_cru000, x_tru00[0], training)
        x_tru001, route_value001, tru001_loss = self.tru4(x_cru001, x_tru00[1], training)
        x_tru010, route_value010, tru010_loss = self.tru5(x_cru010, x_tru01[0], training)
        x_tru011, route_value011, tru011_loss = self.tru6(x_cru011, x_tru01[1], training)

        # Tree Level 4 (Leaf Nodes)
        map0, cls0 = self.sfl0(x_cru000, training)
        map1, cls1 = self.sfl1(x_cru000, training)
        map2, cls2 = self.sfl2(x_cru001, training)
        map3, cls3 = self.sfl3(x_cru001, training)
        map4, cls4 = self.sfl4(x_cru010, training)
        map5, cls5 = self.sfl5(x_cru010, training)
        map6, cls6 = self.sfl6(x_cru011, training)
        map7, cls7 = self.sfl7(x_cru011, training)

        # Output
        maps = [map0, map1, map2, map3, map4, map5, map6, map7]
        clss = [cls0, cls1, cls2, cls3, cls4, cls5, cls6, cls7]
        route_values = [
            route_value0,
            route_value00,
            route_value01,
            route_value000,
            route_value001,
            route_value010,
            route_value011,
        ]

        # Leaf node masks
        x_tru0000 = tf.concat([x_tru000[0], mask_live], axis=1)
        x_tru0001 = tf.concat([x_tru000[1], mask_live], axis=1)
        x_tru0010 = tf.concat([x_tru001[0], mask_live], axis=1)
        x_tru0011 = tf.concat([x_tru001[1], mask_live], axis=1)
        x_tru0100 = tf.concat([x_tru010[0], mask_live], axis=1)
        x_tru0101 = tf.concat([x_tru010[1], mask_live], axis=1)
        x_tru0110 = tf.concat([x_tru011[0], mask_live], axis=1)
        x_tru0111 = tf.concat([x_tru011[1], mask_live], axis=1)
        leaf_node_mask = [
            x_tru0000,
            x_tru0001,
            x_tru0010,
            x_tru0011,
            x_tru0100,
            x_tru0101,
            x_tru0110,
            x_tru0111,
        ]

        if training:
            # Routing and reconstruction losses
            route_loss = [
                tru0_loss[0],
                tru00_loss[0],
                tru01_loss[0],
                tru000_loss[0],
                tru001_loss[0],
                tru010_loss[0],
                tru011_loss[0],
            ]
            recon_loss = [
                tru0_loss[1],
                tru00_loss[1],
                tru01_loss[1],
                tru000_loss[1],
                tru001_loss[1],
                tru010_loss[1],
                tru011_loss[1],
            ]

            # Mu updates for TRU
            mu_update = [
                self.tru0.project.mu_of_visit + 0,
                self.tru1.project.mu_of_visit + 0,
                self.tru2.project.mu_of_visit + 0,
                self.tru3.project.mu_of_visit + 0,
                self.tru4.project.mu_of_visit + 0,
                self.tru5.project.mu_of_visit + 0,
                self.tru6.project.mu_of_visit + 0,
            ]

            eigenvalue = [
                self.tru0.project.eigenvalue,
                self.tru1.project.eigenvalue,
                self.tru2.project.eigenvalue,
                self.tru3.project.eigenvalue,
                self.tru4.project.eigenvalue,
                self.tru5.project.eigenvalue,
                self.tru6.project.eigenvalue,
            ]

            trace = [
                self.tru0.project.trace,
                self.tru1.project.trace,
                self.tru2.project.trace,
                self.tru3.project.trace,
                self.tru4.project.trace,
                self.tru5.project.trace,
                self.tru6.project.trace,
            ]

            return (
                maps,
                clss,
                route_values,
                leaf_node_mask,
                [route_loss, recon_loss],
                mu_update,
                eigenvalue,
                trace,
            )
        else:
            return maps, clss, route_values, leaf_node_mask


class DTNModel:
    """
    High-level DTN model wrapper for training and inference.
    """

    def __init__(self, config: Settings):
        self.config = config
        self.dtn = DTN(config.model.filters, config)
        self.optimizer = tf.keras.optimizers.Adam(config.training.learning_rate, beta_1=0.5)

        # Loss metrics
        self.depth_map_loss = ErrorMetric()
        self.class_loss = ErrorMetric()
        self.route_loss = ErrorMetric()
        self.uniq_loss = ErrorMetric()

        # Checkpoint manager
        self.last_epoch = 0
        self.checkpoint_manager = None

        logger.info("DTN Model initialized")

    def compile(self):
        """Setup checkpointing."""
        checkpoint_dir = self.config.log_dir
        checkpoint = tf.train.Checkpoint(dtn=self.dtn, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, str(checkpoint_dir), max_to_keep=10000
        )

        last_checkpoint = self.checkpoint_manager.latest_checkpoint
        if last_checkpoint:
            checkpoint.restore(last_checkpoint)
            self.last_epoch = int(last_checkpoint.split("-")[-1])
            logger.info(f"Restored from {last_checkpoint}")
        else:
            logger.info("Initializing from scratch")

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        if self.checkpoint_manager:
            self.checkpoint_manager.save(checkpoint_number=epoch)
            logger.info(f"Saved checkpoint for epoch {epoch}")


def plot_results(fname: str, result_list: List[tf.Tensor]):
    """Plot and save visualization of results."""
    column = []
    for count, fig in enumerate(result_list):
        shape = fig.shape
        fig = fig.numpy()
        row = []
        for idx in range(shape[0]):
            item = fig[idx, :, :, :]

            if item.shape[2] == 1:
                item = np.concatenate([item, item, item], axis=2)

            if item.shape[2] == 2:
                item = item[:, :, 1:2]
                item = np.concatenate([item, item, item], axis=2)

            if count == 1:
                item = cv2.resize(item, (100, 100))
            else:
                item = cv2.cvtColor(cv2.resize(item, (100, 100)), cv2.COLOR_RGB2BGR)

            row.append(item)

        row = np.concatenate(row, axis=1)
        column.append(row)

    column = np.concatenate(column, axis=0)
    img = np.uint8(column * 255)
    cv2.imwrite(fname, img)
