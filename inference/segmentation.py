"""
Inference Module for Model 3: ResNet18 U-Net Tomato Disease Segmentation.
Executes actual model inference, applies sigmoid thresholding, and generates
binary masks and disease-region overlays.
"""

import time
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
from PIL import Image
import torch

from models.segmentation.architecture.resnet18_unet import (
    ResNet18UNet,
    load_trained_resnet18_unet
)
from models.segmentation.config.segmentation_config import (
    CHECKPOINT_PATH,
    THRESHOLD_DEFAULT,
    SEG_SIZE
)
from preprocessing.segmentation_preprocessing import (
    preprocess_segmentation_image,
    inverse_spatial_mapping
)


class TomatoDiseaseSegmenter:
    """
    Inference engine for tomato leaf disease segmentation using
    the verified ResNet18 U-Net checkpoint trained on the PlantSeg dataset.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = CHECKPOINT_PATH,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Segmentation checkpoint not found at: {self.checkpoint_path.resolve()}"
            )

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Load trained model strictly
        self.model = load_trained_resnet18_unet(
            checkpoint_path=self.checkpoint_path,
            device=self.device
        )

    def predict(
        self,
        image_input: Any,
        threshold: float = THRESHOLD_DEFAULT
    ) -> Dict[str, Any]:
        """
        Runs disease lesion segmentation on an input image.

        Args:
            image_input: File path, bytes, PIL Image, or NumPy array.
            threshold: Sigmoid probability threshold for binary mask (default: 0.50).

        Returns:
            Dict containing:
                - 'binary_mask': Boolean NumPy array of shape (orig_h, orig_w).
                - 'mask_image': PIL Image of binary mask (255 for disease, 0 for background).
                - 'overlay_image': PIL Image with red disease-region overlay on original image.
                - 'diseased_pixels': Number of positive diseased pixels.
                - 'total_pixels': Total pixels in original image.
                - 'diseased_area_percentage': Percentage of leaf area affected by disease.
                - 'mean_confidence_diseased': Average sigmoid confidence across positive pixels.
                - 'inference_time_ms': Inference time in milliseconds.
                - 'original_size': (width, height) of original image.
                - 'threshold_used': Threshold applied.
        """
        # Preprocessing: Aspect-ratio preserving resize & centered padding to (1, 3, 512, 512)
        tensor_batch, orig_pil, pad_info = preprocess_segmentation_image(
            image_input=image_input,
            target_size=SEG_SIZE,
            device=self.device
        )

        start_time = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor_batch)
            probs_512 = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Threshold 512x512 canvas prediction
        mask_canvas = (probs_512 >= threshold).astype(np.uint8)

        # Inverse spatial mapping back to original image dimensions
        full_mask = inverse_spatial_mapping(mask_canvas, pad_info)

        # Calculate statistics
        total_pixels = full_mask.size
        diseased_pixels = int(np.sum(full_mask))
        diseased_percentage = (diseased_pixels / total_pixels) * 100.0

        # Mean probability over the unpadded predicted region
        left, top = pad_info["padding"]
        new_w, new_h = pad_info["scaled_size"]
        cropped_probs = probs_512[top : top + new_h, left : left + new_w]
        cropped_mask = mask_canvas[top : top + new_h, left : left + new_w] > 0

        if np.any(cropped_mask):
            mean_conf = float(np.mean(cropped_probs[cropped_mask]))
        else:
            mean_conf = 0.0

        # Generate binary mask PIL Image
        mask_image = Image.fromarray((full_mask * 255).astype(np.uint8), mode="L")

        # Generate red overlay matching Cell 14 of segment_done.ipynb:
        # overlay[prediction] = 0.5 * overlay[prediction] + 0.5 * [255, 0, 0]
        orig_np = np.array(orig_pil).copy()
        overlay_np = orig_np.copy()
        if diseased_pixels > 0:
            red_color = np.array([255, 0, 0], dtype=np.float32)
            overlay_np[full_mask] = (
                0.5 * overlay_np[full_mask].astype(np.float32) + 0.5 * red_color
            ).astype(np.uint8)

        overlay_image = Image.fromarray(overlay_np)

        return {
            "binary_mask": full_mask,
            "mask_image": mask_image,
            "overlay_image": overlay_image,
            "diseased_pixels": diseased_pixels,
            "total_pixels": total_pixels,
            "diseased_area_percentage": round(diseased_percentage, 2),
            "mean_confidence_diseased": round(mean_conf, 4),
            "inference_time_ms": round(inference_time_ms, 2),
            "original_size": orig_pil.size,
            "threshold_used": threshold
        }
