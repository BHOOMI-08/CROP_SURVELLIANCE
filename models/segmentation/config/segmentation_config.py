"""
Configuration parameters for Model 3: ResNet18 U-Net Tomato Disease Segmentation.
Extracted strictly from the training notebook (segment_done.ipynb) and verified checkpoint.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEGMENTATION_DIR = PROJECT_ROOT / "models" / "segmentation"

# Checkpoint Path
CHECKPOINT_PATH = SEGMENTATION_DIR / "checkpoints" / "best_segmentation_resnet18_unet_512.pth"

# Model Hyperparameters
INPUT_SIZE = (512, 512)
SEG_SIZE = 512
THRESHOLD_DEFAULT = 0.50
MODEL_ARCH = "ResNet18UNet"
ENCODER = "resnet18"
OUT_CHANNELS = 1

# Dataset Splits & Provenance (Verified from Cell 1 of segment_done.ipynb)
DATASET_DETAILS = {
    "name": "PlantSeg Tomato Leaf Disease Dataset",
    "splits": {
        "train": {"images": 460, "masks": 460, "pairs": 460},
        "val": {"images": 74, "masks": 74, "pairs": 74},
        "test": {"images": 133, "masks": 133, "pairs": 133}
    },
    "total_matched_pairs": 667,
    "task": "Binary semantic segmentation of diseased foliar regions"
}

# Ground-Truth Evaluation Metrics (Verified from Cell 12 & 13 of segment_done.ipynb)
VERIFIED_TEST_METRICS = {
    "model": "ResNet18 U-Net",
    "checkpoint_epoch": 18,
    "best_validation_dice": 0.6795100271701813,
    "test_images": 133,
    "decision_threshold": 0.50,
    "macro_dice": 0.6640,
    "macro_iou": 0.5283,
    "global_dice": 0.6955,
    "global_iou": 0.5332,
    "precision": 0.7556,
    "recall": 0.6442
}
