"""
Inference Module for PlantVillage Tomato Disease Classification (MobileNetV2).
Executes actual model inference using the trained checkpoint and class mapping.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
import torch
import torch.nn.functional as F

from models.classification.architecture.mobilenetv2 import load_trained_mobilenetv2
from models.classification.config.classification_config import (
    CHECKPOINT_PATH,
    CLASS_MAPPING_PATH,
    NUM_CLASSES
)
from preprocessing.classification_preprocessing import (
    preprocess_classification_image,
    load_image_as_pil
)


class TomatoDiseaseClassifier:
    """
    Inference engine for tomato leaf disease classification using
    the verified MobileNetV2 model trained on the PlantVillage tomato dataset.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = CHECKPOINT_PATH,
        class_mapping_path: Union[str, Path] = CLASS_MAPPING_PATH,
        device: torch.device = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.class_mapping_path = Path(class_mapping_path)

        # Device selection: GPU if available, else CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load class mappings
        self.class_to_idx, self.idx_to_class = self._load_class_mapping()

        # Load trained model checkpoint
        self.model = self._load_model()

    def _load_class_mapping(self) -> tuple[Dict[str, int], Dict[int, str]]:
        """Loads and verifies the 10-class mapping from JSON."""
        if not self.class_mapping_path.is_file():
            raise FileNotFoundError(
                f"Class mapping file not found at: {self.class_mapping_path.resolve()}"
            )

        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)

        if len(class_to_idx) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} classes in class mapping, but found {len(class_to_idx)}."
            )

        # Invert mapping: index -> class name
        idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
        return class_to_idx, idx_to_class

    def _load_model(self) -> torch.nn.Module:
        """Loads the trained MobileNetV2 model with checkpoint weights."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained checkpoint not found at: {self.checkpoint_path.resolve()}"
            )

        model = load_trained_mobilenetv2(
            checkpoint_path=self.checkpoint_path,
            num_classes=NUM_CLASSES,
            device=self.device
        )
        return model

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Runs actual model inference on an input tomato leaf image.

        Args:
            image_input: File path, bytes, PIL Image, NumPy array, or file-like object.

        Returns:
            Dict containing:
                - 'predicted_class': String name of the predicted tomato condition/disease
                - 'class_index': Integer index (0-9)
                - 'confidence': Prediction probability (0.0 to 1.0)
                - 'confidence_percentage': Prediction probability as percentage string (e.g. '95.42%')
                - 'all_probabilities': Dict mapping each class name to its probability
        """
        # Preprocess input image using the exact verified pipeline
        input_tensor = preprocess_classification_image(image_input, device=self.device)

        # Forward pass without gradient tracking
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        top_prob, top_idx = torch.topk(probabilities, k=1)
        pred_idx = top_idx.item()
        confidence = top_prob.item()
        pred_class = self.idx_to_class[pred_idx]

        # Extract all probabilities mapped to class names
        all_probs = {
            self.idx_to_class[i]: float(probabilities[i].item())
            for i in range(len(self.idx_to_class))
        }

        return {
            "predicted_class": pred_class,
            "class_index": pred_idx,
            "confidence": float(confidence),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "all_probabilities": all_probs
        }
