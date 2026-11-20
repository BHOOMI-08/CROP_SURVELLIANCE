"""
Automated Integration Testing for Multi-Model Surveillance Pipeline.
Verifies simultaneous execution of Model 1 (Classification), Model 2 (Detection),
and Model 3 (Segmentation).
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import CropSurveillancePipeline


class TestPipelineIntegration(unittest.TestCase):
    """Verifies that the unified pipeline accurately orchestrates all three models."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = CropSurveillancePipeline()
        cls.sample_image = PROJECT_ROOT / "data" / "sample_images" / "test_tomato_leaf.jpg"

    def test_01_classification_via_pipeline(self):
        """Verifies Model 1 execution through the unified pipeline interface."""
        result = self.pipeline.classify_image(self.sample_image)
        self.assertEqual(result["predicted_class"], "Tomato___Early_blight")
        self.assertGreater(result["confidence"], 0.90)

    def test_02_detection_via_pipeline(self):
        """Verifies Model 2 execution through the unified pipeline interface."""
        result = self.pipeline.detect_objects(self.sample_image, conf_threshold=0.25)
        self.assertGreater(result["num_detections"], 0)
        self.assertEqual(result["detections"][0]["class_name"], "Early_blight")

    def test_03_segmentation_via_pipeline(self):
        """Verifies Model 3 execution through the unified pipeline interface."""
        result = self.pipeline.segment_disease(self.sample_image, threshold=0.50)
        self.assertGreater(result["diseased_pixels"], 0)
        self.assertGreater(result["diseased_area_percentage"], 0.0)
        self.assertEqual(result["binary_mask"].shape, (256, 256))

    def test_04_combined_three_model_analysis(self):
        """Verifies simultaneous execution of all three models in combined analysis mode."""
        combined = self.pipeline.analyze_image(self.sample_image)
        self.assertIn("classification", combined)
        self.assertIn("detection", combined)
        self.assertIn("segmentation", combined)

        self.assertEqual(combined["classification"]["predicted_class"], "Tomato___Early_blight")
        self.assertEqual(combined["detection"]["detections"][0]["class_name"], "Early_blight")
        self.assertGreater(combined["segmentation"]["diseased_pixels"], 0)
        print("\n[PIPELINE INTEGRATION] Complete 3-model combined inference verified successfully.")


if __name__ == "__main__":
    unittest.main()
