"""
Integrated Surveillance Pipeline for Crop Disease and Pest Surveillance System.
Orchestrates:
  1. Model 1: MobileNetV2 (Tomato Disease Classification)
  2. Model 2: YOLO11n (Tomato Foliar Disease & Mite Lesion Detection)
  3. Model 3: ResNet18 U-Net (Tomato Disease Lesion Segmentation)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch

from inference.classification import TomatoDiseaseClassifier
from inference.detection import TomatoObjectDetector
from inference.segmentation import TomatoDiseaseSegmenter


class CropSurveillancePipeline:
    """
    Unified multi-task surveillance pipeline for tomato disease diagnostics.
    Provides modular execution for classification, detection, segmentation,
    and unified multi-model analysis.
    """

    def __init__(
        self,
        load_classification: bool = True,
        load_detection: bool = True,
        load_segmentation: bool = True,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.device = device
        self.classifier: Optional[TomatoDiseaseClassifier] = None
        self.detector: Optional[TomatoObjectDetector] = None
        self.segmenter: Optional[TomatoDiseaseSegmenter] = None

        if load_classification:
            self.classifier = TomatoDiseaseClassifier(device=device)

        if load_detection:
            self.detector = TomatoObjectDetector(device=device)

        if load_segmentation:
            self.segmenter = TomatoDiseaseSegmenter(device=device)

    def classify_image(self, image_input: Any) -> Dict[str, Any]:
        """Runs Model 1: MobileNetV2 Disease Classification."""
        if self.classifier is None:
            self.classifier = TomatoDiseaseClassifier(device=self.device)
        return self.classifier.predict(image_input)

    def detect_objects(
        self,
        image_input: Any,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """Runs Model 2: YOLO11n Object & Lesion Detection."""
        if self.detector is None:
            self.detector = TomatoObjectDetector(device=self.device)
        return self.detector.predict(
            image_input,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

    def segment_disease(
        self,
        image_input: Any,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Runs Model 3: ResNet18 U-Net Disease Lesion Segmentation."""
        if self.segmenter is None:
            self.segmenter = TomatoDiseaseSegmenter(device=self.device)
        return self.segmenter.predict(image_input, threshold=threshold)

    def analyze_image(
        self,
        image_input: Any,
        detection_conf: float = 0.25,
        segmentation_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Executes unified surveillance analysis using all three models:
          1. Classification: Whole-leaf disease diagnosis and probability distribution.
          2. Detection: Localized disease spots with bounding boxes and detection confidence.
          3. Segmentation: Exact pixel-level disease lesion mask and overlay.
        """
        classification_result = self.classify_image(image_input)
        detection_result = self.detect_objects(image_input, conf_threshold=detection_conf)
        segmentation_result = self.segment_disease(image_input, threshold=segmentation_threshold)

        return {
            "classification": classification_result,
            "detection": detection_result,
            "segmentation": segmentation_result
        }
