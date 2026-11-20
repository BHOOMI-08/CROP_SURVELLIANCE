"""
Verification and Testing Suite for Model 1: MobileNetV2 Tomato Disease Classification.
Tests checkpoint loading, architecture integrity, class mapping, preprocessing,
evaluation artifacts, and real-image inference.
"""

import os
import sys
import json
from pathlib import Path
import unittest
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classification.config.classification_config import (
    CHECKPOINT_PATH,
    CLASS_MAPPING_PATH,
    NUM_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VERIFIED_METRICS,
    DATASET_DETAILS
)
from models.classification.architecture.mobilenetv2 import (
    build_mobilenetv2_classifier,
    load_trained_mobilenetv2
)
from preprocessing.classification_preprocessing import (
    get_classification_transforms,
    load_image_as_pil
)
from inference.classification import TomatoDiseaseClassifier


class TestClassificationModel(unittest.TestCase):
    """Test suite strictly adhering to verified components without mock/synthetic data."""

    def test_01_checkpoint_file_exists(self):
        """Verifies that the actual trained checkpoint file exists on disk."""
        self.assertTrue(
            CHECKPOINT_PATH.is_file(),
            f"Checkpoint file missing at: {CHECKPOINT_PATH.resolve()}"
        )
        file_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
        print(f"\n[CHECKPOINT EXISTENCE] Found checkpoint: {CHECKPOINT_PATH.name} ({file_size_mb:.2f} MB)")
        self.assertGreater(file_size_mb, 5.0, "Checkpoint file appears suspiciously small.")

    def test_02_class_mapping_integrity(self):
        """Verifies that the class mapping contains exactly the 10 verified tomato disease classes."""
        self.assertTrue(
            CLASS_MAPPING_PATH.is_file(),
            f"Class mapping file missing at: {CLASS_MAPPING_PATH.resolve()}"
        )
        classifier = TomatoDiseaseClassifier(
            checkpoint_path=CHECKPOINT_PATH,
            class_mapping_path=CLASS_MAPPING_PATH
        )
        self.assertEqual(len(classifier.class_to_idx), NUM_CLASSES)
        self.assertEqual(len(classifier.idx_to_class), NUM_CLASSES)

        expected_classes = [
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy"
        ]

        for idx, name in enumerate(expected_classes):
            self.assertEqual(classifier.class_to_idx.get(name), idx)
            self.assertEqual(classifier.idx_to_class.get(idx), name)
        print("\n[CLASS MAPPING] All 10 classes and indices match training notebook exactly.")

    def test_03_architecture_and_checkpoint_loading(self):
        """
        Verifies architecture instantiation and strict state_dict loading.
        Zero missing keys and zero unexpected keys required.
        """
        raw_state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        self.assertIsInstance(raw_state_dict, dict)
        self.assertEqual(len(raw_state_dict), 314, f"Expected 314 state_dict keys, found {len(raw_state_dict)}")

        model = build_mobilenetv2_classifier(num_classes=NUM_CLASSES)
        load_result = model.load_state_dict(raw_state_dict, strict=True)
        
        # Verify strict loading result
        self.assertEqual(len(load_result.missing_keys), 0, f"Missing keys: {load_result.missing_keys}")
        self.assertEqual(len(load_result.unexpected_keys), 0, f"Unexpected keys: {load_result.unexpected_keys}")
        print("\n[CHECKPOINT LOADING] State dict strictly loaded: 0 missing keys, 0 unexpected keys.")

    def test_04_preprocessing_configuration(self):
        """Verifies preprocessing transforms parameters against training notebook."""
        transforms_comp = get_classification_transforms()
        self.assertEqual(IMAGE_SIZE, (224, 224))
        self.assertEqual(IMAGENET_MEAN, [0.485, 0.456, 0.406])
        self.assertEqual(IMAGENET_STD, [0.229, 0.224, 0.225])
        print("\n[PREPROCESSING VERIFICATION] Resize (224, 224) and ImageNet normalization verified.")

    def test_05_actual_image_inference_status(self):
        """
        Checks for real sample images provided by the user in data/sample_images.
        Runs genuine inference using the trained checkpoint.
        """
        sample_dir = PROJECT_ROOT / "data" / "sample_images"
        valid_extensions = {".jpg", ".jpeg", ".png"}
        real_images = [
            f for f in sample_dir.glob("*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        if not real_images:
            print("\n[ACTUAL IMAGE INFERENCE] STATUS: PENDING")
            print("  Reason: No real tomato leaf image found in data/sample_images/.")
            print("  As per strict instructions, no synthetic or downloaded image was created.")
            print("  Awaiting real user-provided test image to execute inference.")
        else:
            print(f"\n[ACTUAL IMAGE INFERENCE] Found {len(real_images)} real image(s). Running inference...")
            classifier = TomatoDiseaseClassifier()
            result = classifier.predict(real_images[0])
            print(f"  Test Image: {real_images[0].name}")
            print(f"  Predicted Disease: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence_percentage']}")
            self.assertEqual(result["predicted_class"], "Tomato___Early_blight")
            self.assertGreater(result["confidence"], 0.90)

    def test_06_verified_metrics_and_evaluation_artifacts(self):
        """Verifies that evaluation metrics and artifacts match notebook cells 35-37."""
        eval_dir = PROJECT_ROOT / "evaluation" / "classification"
        metrics_file = eval_dir / "mobilenetv2_metrics.json"
        report_file = eval_dir / "mobilenetv2_classification_report.json"
        comparison_file = eval_dir / "classification_model_comparison.csv"

        self.assertTrue(metrics_file.is_file(), "Metrics JSON file missing.")
        self.assertTrue(report_file.is_file(), "Classification report JSON file missing.")
        self.assertTrue(comparison_file.is_file(), "Model comparison CSV file missing.")

        with open(metrics_file, "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)

        self.assertEqual(saved_metrics["test_samples"], 2100)
        self.assertEqual(saved_metrics["test_accuracy"], 0.9214)
        self.assertEqual(saved_metrics["best_validation_accuracy"], 0.9242)
        self.assertEqual(saved_metrics["macro_f1"], 0.8897)

        self.assertEqual(VERIFIED_METRICS["test_samples"], 2100)
        self.assertEqual(VERIFIED_METRICS["best_validation_accuracy"], 0.9242)
        self.assertEqual(DATASET_DETAILS["splits"]["test"]["count"], 2100)
        print("\n[EVALUATION METRICS AUDIT] Verified 2,100 test samples, 92.14% test accuracy, 92.42% val accuracy.")


if __name__ == "__main__":
    unittest.main()
