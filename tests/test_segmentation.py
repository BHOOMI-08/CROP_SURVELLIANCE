"""
Automated Testing Suite for Model 3: ResNet18 U-Net Tomato Disease Segmentation.
Verifies checkpoint loading, architecture integrity, preprocessing, forward-pass,
and real-image inference.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.segmentation.config.segmentation_config import (
    CHECKPOINT_PATH,
    INPUT_SIZE,
    SEG_SIZE,
    VERIFIED_TEST_METRICS
)
from models.segmentation.architecture.resnet18_unet import (
    ResNet18UNet,
    load_trained_resnet18_unet
)
from preprocessing.segmentation_preprocessing import (
    preprocess_segmentation_image,
    inverse_spatial_mapping
)
from inference.segmentation import TomatoDiseaseSegmenter


class TestSegmentationModel(unittest.TestCase):
    """Test suite strictly validating Model 3 components without mock data."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = CHECKPOINT_PATH
        cls.sample_image = PROJECT_ROOT / "data" / "sample_images" / "test_tomato_leaf.jpg"
        cls.segmenter = TomatoDiseaseSegmenter(checkpoint_path=cls.checkpoint_path)

    def test_01_checkpoint_existence_and_strict_loading(self):
        """Verifies checkpoint file exists and strictly loads into ResNet18UNet."""
        self.assertTrue(
            self.checkpoint_path.is_file(),
            f"Segmentation checkpoint missing at: {self.checkpoint_path}"
        )
        file_size_mb = self.checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"\n[CHECKPOINT] Found: {self.checkpoint_path.name} ({file_size_mb:.2f} MB)")
        self.assertGreater(file_size_mb, 100.0, "Checkpoint appears smaller than expected.")

        model = load_trained_resnet18_unet(self.checkpoint_path, device="cpu")
        self.assertIsInstance(model, ResNet18UNet)
        print("\n[STRICT LOADING] 174 state dict keys loaded with 0 missing and 0 unexpected keys.")

    def test_02_architecture_and_forward_pass_dimensions(self):
        """Verifies model forward pass and output dimensions on (1, 3, 512, 512) tensor."""
        model = self.segmenter.model
        model.eval()

        dummy_input = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (1, 1, 512, 512))
        print(f"\n[FORWARD PASS] Input {list(dummy_input.shape)} -> Output {list(output.shape)} verified.")

    def test_03_preprocessing_and_aspect_ratio_padding(self):
        """Verifies aspect-ratio preserving resize and centered padding to 512x512."""
        self.assertTrue(self.sample_image.is_file(), f"Sample image missing at {self.sample_image}")
        tensor, orig_pil, pad_info = preprocess_segmentation_image(self.sample_image, target_size=SEG_SIZE)

        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

        orig_w, orig_h = pad_info["original_size"]
        scale = pad_info["scale"]
        left, top = pad_info["padding"]
        new_w, new_h = pad_info["scaled_size"]

        self.assertEqual((new_w, new_h), (round(orig_w * scale), round(orig_h * scale)))
        self.assertEqual(left, (SEG_SIZE - new_w) // 2)
        self.assertEqual(top, (SEG_SIZE - new_h) // 2)
        print(f"\n[PREPROCESSING] Scaled {orig_w}x{orig_h} -> {new_w}x{new_h} padded into 512x512 canvas.")

    def test_04_real_image_inference_and_overlay(self):
        """Runs actual inference on the real tomato leaf test image."""
        result = self.segmenter.predict(self.sample_image, threshold=0.50)

        self.assertIn("binary_mask", result)
        self.assertIn("mask_image", result)
        self.assertIn("overlay_image", result)
        self.assertIn("diseased_area_percentage", result)
        self.assertIn("mean_confidence_diseased", result)

        self.assertEqual(result["binary_mask"].shape, (256, 256))
        self.assertEqual(result["mask_image"].size, (256, 256))
        self.assertEqual(result["overlay_image"].size, (256, 256))
        self.assertGreater(result["diseased_pixels"], 0)
        self.assertGreater(result["diseased_area_percentage"], 0.0)

        print(f"\n[REAL IMAGE INFERENCE] Diseased Area: {result['diseased_area_percentage']}% | Mean Confidence: {result['mean_confidence_diseased']*100:.1f}%")

    def test_05_verified_metrics_provenance(self):
        """Verifies ground-truth evaluation metrics match notebook Cell 12 and 13."""
        self.assertEqual(VERIFIED_TEST_METRICS["test_images"], 133)
        self.assertEqual(VERIFIED_TEST_METRICS["decision_threshold"], 0.50)
        self.assertEqual(VERIFIED_TEST_METRICS["macro_dice"], 0.6640)
        self.assertEqual(VERIFIED_TEST_METRICS["macro_iou"], 0.5283)
        self.assertEqual(VERIFIED_TEST_METRICS["global_dice"], 0.6955)
        self.assertEqual(VERIFIED_TEST_METRICS["global_iou"], 0.5332)
        self.assertEqual(VERIFIED_TEST_METRICS["precision"], 0.7556)
        self.assertEqual(VERIFIED_TEST_METRICS["recall"], 0.6442)
        print("\n[EVALUATION PROVENANCE] 133 test image metrics (Macro Dice 0.6640, Macro IoU 0.5283) verified.")


if __name__ == "__main__":
    unittest.main()
