"""
Configuration parameters for Model 2: YOLO11n Tomato Object Detection.
Derived directly and strictly from the verified training notebook (yolooomodel2.ipynb)
and the trained checkpoint (best_tomato_detection.pt).
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DETECTION_DIR = PROJECT_ROOT / "models" / "detection"

# Artifact Paths
CHECKPOINT_PATH = DETECTION_DIR / "checkpoints" / "best_tomato_detection.pt"
DATA_YAML_PATH = DETECTION_DIR / "config" / "data.yaml"

# Model Hyperparameters & Architecture Specifications
MODEL_ARCH = "yolo11n"
TASK = "detect"
NUM_CLASSES = 10
INPUT_SIZE = (640, 640)
CONF_THRESHOLD_DEFAULT = 0.25
IOU_THRESHOLD_DEFAULT = 0.45

# 10 Verified Class Names (from checkpoint metadata and data.yaml)
CLASS_NAMES = {
    0: "Bacterial_spot",
    1: "Early_blight",
    2: "Late_blight",
    3: "Leaf_Mold",
    4: "Septoria_leaf_spot",
    5: "Spider_mites Two-spotted_spider_mite",
    6: "Target_Spot",
    7: "Tomato_Yellow_Leaf_Curl_Virus",
    8: "Tomato__healthy",
    9: "Tomato_mosaic_virus"
}

# Dataset Provenance & Split Details (Verified from Cell 4 of yolooomodel2.ipynb)
DATASET_DETAILS = {
    "name": "tomato-leaf-detection-u6mou-iisg2 (Roboflow Universe v1)",
    "license": "CC BY 4.0",
    "splits": {
        "train": {"images": 6928, "labels": 6928},
        "valid": {"images": 1969, "labels": 1969},
        "test": {"images": 997, "labels": 997}
    },
    "total_images": 9894,
    "total_labels": 9894,
    "excluded_validation_duplicates": 1
}

# Ground-Truth Evaluation Metrics on Held-out Test Split (Verified from Cell 9)
VERIFIED_TEST_METRICS = {
    "model": "YOLO11n",
    "eval_split": "test",
    "test_images": 997,
    "test_instances": 998,
    "precision": 0.996,
    "recall": 0.992,
    "mAP50": 0.994,
    "mAP50_95": 0.946,
    "per_class_metrics": {
        "Bacterial_spot": {"instances": 103, "precision": 1.000, "recall": 0.974, "mAP50": 0.995, "mAP50_95": 0.966},
        "Early_blight": {"instances": 94, "precision": 0.992, "recall": 1.000, "mAP50": 0.995, "mAP50_95": 0.953},
        "Late_blight": {"instances": 103, "precision": 1.000, "recall": 0.995, "mAP50": 0.995, "mAP50_95": 0.952},
        "Leaf_Mold": {"instances": 101, "precision": 0.998, "recall": 1.000, "mAP50": 0.995, "mAP50_95": 0.928},
        "Septoria_leaf_spot": {"instances": 101, "precision": 0.995, "recall": 0.980, "mAP50": 0.995, "mAP50_95": 0.936},
        "Spider_mites Two-spotted_spider_mite": {"instances": 100, "precision": 1.000, "recall": 0.993, "mAP50": 0.995, "mAP50_95": 0.903},
        "Target_Spot": {"instances": 100, "precision": 0.994, "recall": 0.980, "mAP50": 0.985, "mAP50_95": 0.940},
        "Tomato_Yellow_Leaf_Curl_Virus": {"instances": 100, "precision": 0.988, "recall": 1.000, "mAP50": 0.995, "mAP50_95": 0.971},
        "Tomato__healthy": {"instances": 103, "precision": 0.998, "recall": 1.000, "mAP50": 0.995, "mAP50_95": 0.952},
        "Tomato_mosaic_virus": {"instances": 93, "precision": 0.997, "recall": 1.000, "mAP50": 0.995, "mAP50_95": 0.954}
    }
}
