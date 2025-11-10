"""
Prediction and inference for trained DTN models.

Provides high-level interface for running inference on images and videos.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger

from deepfake_detector.core.config import Settings
from deepfake_detector.model.dtn import DTNModel


class Predictor:
    """
    Predictor for running inference on images/videos.

    Handles model loading, preprocessing, and prediction.
    """

    def __init__(self, model_path: Union[str, Path], settings: Optional[Settings] = None):
        self.model_path = Path(model_path)
        self.settings = settings or Settings()
        self.model = DTNModel(self.settings)
        self.model.compile()

        logger.info(f"Predictor initialized with model: {self.model_path}")

    def preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize to model input size
        image = cv2.resize(image, (self.settings.model.image_size, self.settings.model.image_size))

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Normalize
        image_rgb = image_rgb / 255.0
        image_hsv = image_hsv / 255.0

        # Concatenate RGB and HSV channels
        image = np.concatenate([image_rgb, image_hsv], axis=2)

        # Add batch dimension
        image = np.expand_dims(image, axis=0).astype(np.float32)

        return image

    def predict(
        self, input_path: Union[str, Path], threshold: float = 0.5
    ) -> Dict[str, Union[bool, float]]:
        """
        Predict if input is a deepfake.

        Args:
            input_path: Path to image or video file
            threshold: Classification threshold

        Returns:
            Dictionary with prediction results
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Preprocess
        image = self.preprocess_image(input_path)

        # Create dummy label for inference
        label = tf.ones((1, 1), dtype=tf.float32)

        # Run inference
        maps, clss, route_values, leaf_node_mask = self.model.dtn(image, label, training=False)

        # Aggregate predictions from leaf nodes
        # Average classification scores weighted by routing
        scores = []
        for cls, mask in zip(clss, leaf_node_mask):
            score = tf.reduce_mean(cls * mask[:, 0]).numpy()
            scores.append(score)

        final_score = np.mean([s for s in scores if not np.isnan(s)])

        # Apply sigmoid for probability
        confidence = 1 / (1 + np.exp(-final_score))

        is_fake = confidence > threshold

        result = {
            "is_fake": bool(is_fake),
            "confidence": float(confidence),
            "score": float(final_score),
            "leaf_scores": [float(s) for s in scores],
        }

        logger.info(f"Prediction: {'FAKE' if is_fake else 'REAL'} (confidence: {confidence:.2%})")

        return result

    def visualize(
        self,
        input_path: Union[str, Path],
        result: Dict,
        output_path: Union[str, Path],
    ):
        """
        Create visualization of prediction.

        Args:
            input_path: Path to input image
            result: Prediction result dictionary
            output_path: Path to save visualization
        """
        # Load original image
        image = cv2.imread(str(input_path))

        # Add prediction text
        text = f"{'FAKE' if result['is_fake'] else 'REAL'} ({result['confidence']:.2%})"
        color = (0, 0, 255) if result["is_fake"] else (0, 255, 0)

        cv2.putText(
            image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

        # Save visualization
        cv2.imwrite(str(output_path), image)
        logger.info(f"Visualization saved to: {output_path}")
