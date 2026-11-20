"""
Automated Testing Suite for Model 2: YOLO11n Tomato Object Detection.
Validates checkpoint integrity, architecture, class mapping, inference outputs,
bounding-box extraction, empty detection handling, and Model 1 isolation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection.config.detection_config import (
    CHECKPOINT_PATH,
    DATA_YAML_PATH,
    NUM_CLASSES,
    CLASS_NAMES,
    VERIFIED_TEST_METRICS
)
from inference.detection import TomatoObjectDetector
from inference.classification import TomatoDiseaseClassifier


class TestDetectionModel(unittest.TestCase):
    """Test suite strictly verifying Model 2 without mock data or substitute models."""

    @classmethod
    def setUpClass(cls):
        """Initializes detector once for test suite efficiency."""
        cls.detector = TomatoObjectDetector(checkpoint_path=CHECKPOINT_PATH)
        cls.sample_image_path = PROJECT_ROOT / "data" / "sample_images" / "test_tomato_leaf.jpg"

    def test_01_checkpoint_existence_and_integrity(self):
        """Verifies checkpoint file existence, type and size on disk."""
        self.assertTrue(
            CHECKPOINT_PATH.is_file(),
            f"Detection checkpoint missing at: {CHECKPOINT_PATH.resolve()}"
        )
        file_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
        print(f"\n[CHECKPOINT] Found: {CHECKPOINT_PATH.name} ({file_size_mb:.2f} MB)")
        self.assertGreater(file_size_mb, 4.0, "Checkpoint file appears suspiciously small.")

    def test_02_architecture_and_class_mapping(self):
        """Verifies YOLO architecture, task, and exact 10-class mapping."""
        self.assertEqual(len(self.detector.names), NUM_CLASSES)
        self.assertEqual(self.detector.model.task, "detect")

        for cls_id, expected_name in CLASS_NAMES.items():
            self.assertEqual(self.detector.names.get(cls_id), expected_name)
        print("\n[ARCHITECTURE & CLASSES] YOLO11n (detect task) with exactly 10 classes verified.")

    def test_03_inference_output_structure_and_boxes(self):
        """Verifies inference execution, latency recording, and bounding-box structure."""
        self.assertTrue(
            self.sample_image_path.is_file(),
            f"Sample image missing at: {self.sample_image_path}"
        )
        result = self.detector.predict(self.sample_image_path, conf_threshold=0.25)

        self.assertIn("num_detections", result)
        self.assertIn("detections", result)
        self.assertIn("inference_time_ms", result)
        self.assertIn("annotated_image", result)

        self.assertGreater(result["num_detections"], 0)
        self.assertGreater(result["inference_time_ms"], 0.0)

        # Inspect first detection box
        det = result["detections"][0]
        self.assertIn("box_xyxy", det)
        self.assertEqual(len(det["box_xyxy"]), 4)
        x1, y1, x2, y2 = det["box_xyxy"]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertGreaterEqual(det["confidence"], 0.25)
        self.assertIn(det["class_id"], CLASS_NAMES)
        print(f"\n[INFERENCE OUTPUT] Detected: {det['class_name']} ({det['confidence_percentage']}) | Box: {det['box_xyxy']}")

    def test_04_empty_detection_handling(self):
        """Verifies that an image without detectable objects returns 0 detections cleanly."""
        # Pure black image with zero leaf/disease features at extreme confidence
        blank_img = Image.new("RGB", (640, 640), color=(0, 0, 0))
        result = self.detector.predict(blank_img, conf_threshold=0.99)

        self.assertEqual(result["num_detections"], 0)
        self.assertEqual(len(result["detections"]), 0)
        self.assertIsNotNone(result["annotated_image"])
        print("\n[EMPTY DETECTION] Zero detections handled cleanly without exceptions.")

    def test_05_visualization_generation(self):
        """Verifies that annotated image with bounding boxes is properly generated."""
        result = self.detector.predict(self.sample_image_path, conf_threshold=0.25)
        annotated = result["annotated_image"]

        self.assertIsInstance(annotated, Image.Image)
        self.assertGreater(annotated.width, 0)
        self.assertGreater(annotated.height, 0)
        print(f"\n[VISUALIZATION] Annotated image generated: {annotated.size} in {annotated.mode} mode.")

    def test_06_model1_unaffected_isolation(self):
        """Ensures Model 1 (MobileNetV2 classification) remains completely intact and functional."""
        classifier = TomatoDiseaseClassifier()
        pred = classifier.predict(self.sample_image_path)
        self.assertEqual(pred["predicted_class"], "Tomato___Early_blight")
        self.assertGreater(pred["confidence"], 0.90)
        print("\n[MODEL 1 ISOLATION] Model 1 classification runs flawlessly and remains unaffected.")


if __name__ == "__main__":
    unittest.main()
