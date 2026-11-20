"""
Configuration parameters for PlantVillage Tomato Disease Classification (MobileNetV2).
Derived directly and strictly from the verified training notebook: Plant_Village_FINAL.ipynb.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASSIFICATION_DIR = PROJECT_ROOT / "models" / "classification"

# Model Artifact Paths
CHECKPOINT_PATH = CLASSIFICATION_DIR / "checkpoints" / "best_mobilenetv2.pth"
CLASS_MAPPING_PATH = CLASSIFICATION_DIR / "class_mapping.json"

# Preprocessing Constants (Verified from Cell 11 of training notebook)
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Architecture Constants (Verified from Cell 32 of training notebook)
NUM_CLASSES = 10
ARCH_BACKBONE = "mobilenet_v2"
LAST_CHANNEL = 1280
DROPOUT_RATE = 0.2

# Dataset Split Constants (Verified from Cell 5 & 6 of training notebook)
DATASET_DETAILS = {
    "name": "PlantVillage Tomato Subset",
    "total_raw_images": 14006,
    "corrupted_images": 0,
    "duplicate_images_excluded": 11,
    "total_unique_images": 13995,
    "splits": {
        "train": {"count": 9796, "percentage": 70.0},
        "validation": {"count": 2099, "percentage": 15.0},
        "test": {"count": 2100, "percentage": 15.0}
    },
    "leakage_verification": {
        "train_val_overlap": 0,
        "train_test_overlap": 0,
        "val_test_overlap": 0,
        "deduplication_method": "MD5 Hash"
    }
}

# Verified Evaluation Metrics (from Plant_Village_FINAL.ipynb Cells 35, 36 & 37)
VERIFIED_METRICS = {
    "model": "MobileNetV2",
    "pretrained_weights": "ImageNet (Frozen Backbone)",
    "training_epochs": 10,
    "batch_size": 32,
    "optimizer": "Adam (lr=0.001, weight_decay=1e-4)",
    "loss": "Weighted CrossEntropyLoss",
    "best_validation_accuracy": 0.9242,
    "best_validation_epoch": 10,
    "test_samples": 2100,
    "test_accuracy": 0.9214,
    "macro_precision": 0.8839,
    "macro_recall": 0.9007,
    "macro_f1": 0.8897
}
