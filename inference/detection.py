"""
Inference module for Model 2: YOLO11n Tomato Object Detection.
Loads the trained checkpoint and executes object detection with bounding-box extraction.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from models.detection.config.detection_config import (
    CHECKPOINT_PATH,
    NUM_CLASSES,
    CONF_THRESHOLD_DEFAULT,
    IOU_THRESHOLD_DEFAULT,
    CLASS_NAMES
)
from preprocessing.detection_preprocessing import load_detection_image


class TomatoObjectDetector:
    """
    Inference engine for tomato leaf disease detection using
    the verified YOLO11n checkpoint trained on the tomato detection dataset.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = CHECKPOINT_PATH,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Detection checkpoint not found at: {self.checkpoint_path.resolve()}"
            )

        # Device resolution
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        # Load YOLO model
        self.model = YOLO(str(self.checkpoint_path))
        self.model.to(self.device)

        # Verify class count and mapping
        self.names = self.model.names
        if len(self.names) != NUM_CLASSES:
            raise ValueError(
                f"Model checkpoint expected {NUM_CLASSES} classes, but found {len(self.names)}."
            )

    def predict(
        self,
        image_input: Any,
        conf_threshold: float = CONF_THRESHOLD_DEFAULT,
        iou_threshold: float = IOU_THRESHOLD_DEFAULT
    ) -> Dict[str, Any]:
        """
        Runs object detection inference on an input image.

        Args:
            image_input: File path, bytes, PIL image, or numpy array.
            conf_threshold: Confidence filtering threshold (default 0.25).
            iou_threshold: NMS IoU threshold (default 0.45).

        Returns:
            Dict containing:
                - 'num_detections': Total bounding boxes detected.
                - 'detections': List of dicts for each detection:
                    {'box': [x1, y1, x2, y2], 'class_id': int, 'class_name': str, 'confidence': float}
                - 'inference_time_ms': Inference latency in milliseconds.
                - 'device_used': Execution device.
                - 'annotated_image': PIL Image with bounding boxes drawn.
        """
        pil_image = load_detection_image(image_input)

        start_time = time.perf_counter()
        results = self.model.predict(
            source=pil_image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Dict[str, Any]] = []
        annotated_image: Optional[Image.Image] = None

        if results and len(results) > 0:
            result = results[0]

            # Generate annotated image using Ultralytics plotting
            # plot() returns BGR numpy array
            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[..., ::-1]  # Convert BGR to RGB
            annotated_image = Image.fromarray(annotated_rgb)

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    xyxy = [round(float(c), 2) for c in b.xyxy[0].tolist()]

                    detections.append({
                        "box_xyxy": xyxy,
                        "class_id": cls_id,
                        "class_name": self.names.get(cls_id, f"class_{cls_id}"),
                        "confidence": round(conf, 4),
                        "confidence_percentage": f"{conf * 100:.2f}%"
                    })
        else:
            annotated_image = pil_image.copy()

        return {
            "num_detections": len(detections),
            "detections": detections,
            "inference_time_ms": round(inference_time_ms, 2),
            "device_used": self.device,
            "annotated_image": annotated_image
        }
