"""
Training orchestration for Deep Tree Network.

Handles the complete training loop with progress tracking and logging.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from deepfake_detector.core.config import Settings
from deepfake_detector.model.dtn import DTNModel, plot_results
from deepfake_detector.model.loss import leaf_l1_loss

console = Console()


class Trainer:
    """
    Training orchestrator for DTN model.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = DTNModel(settings)
        self.model.compile()

        logger.info("Trainer initialized")

    def train(self):
        """Run complete training loop."""
        logger.info("Starting training...")

        console.print("\n[bold cyan]🎓 Training Deep Tree Network[/bold cyan]\n")

        # Display configuration
        self.settings.display()

        # Note: Dataset loading would go here
        # For now, this is a skeleton showing the structure
        logger.warning("Dataset loading not implemented - add your dataset logic")

        console.print(
            "[yellow]⚠️  Training requires dataset implementation[/yellow]\n"
        )

    def train_one_step(self, data_batch, step: int, training: bool = True):
        """
        Execute one training step.

        Args:
            data_batch: Batch of (image, dmap, labels)
            step: Current step number
            training: Training mode flag

        Returns:
            Tuple of losses and metrics
        """
        image, dmap, labels = data_batch

        with tf.GradientTape() as tape:
            dmap_pred, cls_pred, route_value, leaf_node_mask, tru_loss, mu_update, eigenvalue, trace = (
                self.model.dtn(image, labels, True)
            )

            # Supervised loss
            depth_map_loss = leaf_l1_loss(
                dmap_pred, tf.image.resize(dmap, [32, 32]), leaf_node_mask
            )
            class_loss = leaf_l1_loss(cls_pred, labels, leaf_node_mask)
            supervised_loss = depth_map_loss + 0.001 * class_loss

            # Unsupervised tree loss
            route_loss = tf.reduce_mean(
                tf.stack(tru_loss[0], axis=0) * [1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
            )
            uniq_loss = tf.reduce_mean(
                tf.stack(tru_loss[1], axis=0) * [1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
            )
            eigenvalue_mean = np.mean(
                np.stack(eigenvalue, axis=0) * [1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
            )
            trace_mean = np.mean(
                np.stack(trace, axis=0) * [1.0, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
            )
            unsupervised_loss = 1 * route_loss + 0.001 * uniq_loss

            # Total loss
            if step > 10000000:  # Initially supervised only
                loss = supervised_loss + unsupervised_loss
            else:
                loss = supervised_loss

        if training:
            # Backpropagation
            gradients = tape.gradient(loss, self.model.dtn.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.dtn.trainable_variables)
            )

            # Update mu values
            mu_update_rate = self.settings.model.tru_parameters.mu_update_rate
            mu = [
                self.model.dtn.tru0.project.mu,
                self.model.dtn.tru1.project.mu,
                self.model.dtn.tru2.project.mu,
                self.model.dtn.tru3.project.mu,
                self.model.dtn.tru4.project.mu,
                self.model.dtn.tru5.project.mu,
                self.model.dtn.tru6.project.mu,
            ]
            for mu_var, mu_of_visit in zip(mu, mu_update):
                if step == 0:
                    update_mu = mu_of_visit
                else:
                    update_mu = mu_of_visit * mu_update_rate + mu_var * (
                        1 - mu_update_rate
                    )
                K.set_value(mu_var, update_mu)

        # Leaf counts
        spoof_counts = []
        for leaf in leaf_node_mask:
            spoof_count = tf.reduce_sum(leaf[:, 0]).numpy()
            spoof_counts.append(int(spoof_count))

        # Visualization data
        to_plot = [
            image[:, :, :, 0:3],
            image[:, :, :, 3:],
            dmap,
            dmap_pred[0],
            dmap_pred[1],
            dmap_pred[2],
            dmap_pred[3],
            dmap_pred[4],
            dmap_pred[5],
            dmap_pred[6],
            dmap_pred[7],
        ]

        return (
            depth_map_loss,
            class_loss,
            route_loss,
            uniq_loss,
            spoof_counts,
            eigenvalue_mean,
            trace_mean,
            to_plot,
        )
