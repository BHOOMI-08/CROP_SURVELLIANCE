"""
Preprocessing pipeline for Model 3: ResNet18 U-Net Tomato Disease Segmentation.
Replicates the exact preprocessing, aspect-ratio preservation, centered padding,
and inverse spatial mapping from Cells 8 and 14 of segment_done.ipynb.
"""

from io import BytesIO
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps
import torch

from models.segmentation.config.segmentation_config import SEG_SIZE


def load_raw_image(
    image_input: Union[str, Path, bytes, BytesIO, Image.Image, np.ndarray]
) -> Image.Image:
    """
    Loads an image from multiple sources, corrects EXIF orientation, and converts to RGB.

    Args:
        image_input: File path, bytes, PIL Image, or NumPy array.

    Returns:
        PIL.Image.Image: EXIF-corrected RGB image.
    """
    if image_input is None:
        raise ValueError("Image input cannot be None.")

    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path.resolve()}")
            with Image.open(path) as img:
                return ImageOps.exif_transpose(img).convert("RGB")

        elif isinstance(image_input, bytes):
            with Image.open(BytesIO(image_input)) as img:
                return ImageOps.exif_transpose(img).convert("RGB")

        elif isinstance(image_input, BytesIO):
            image_input.seek(0)
            with Image.open(image_input) as img:
                return ImageOps.exif_transpose(img).convert("RGB")

        elif isinstance(image_input, Image.Image):
            return ImageOps.exif_transpose(image_input).convert("RGB")

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("NumPy array is empty.")
            if image_input.ndim == 2:
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                return Image.fromarray(image_input).convert("RGB")
            else:
                raise ValueError(f"Unsupported array shape: {image_input.shape}")

        elif hasattr(image_input, "read"):
            content = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            with Image.open(BytesIO(content)) as img:
                return ImageOps.exif_transpose(img).convert("RGB")

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError, TypeError)):
            raise
        raise ValueError(f"Failed to load or validate image: {str(e)}") from e


def preprocess_segmentation_image(
    image_input: Union[str, Path, bytes, BytesIO, Image.Image, np.ndarray],
    target_size: int = SEG_SIZE,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, Image.Image, Dict[str, Any]]:
    """
    Applies exact aspect-ratio preserving resize and centered padding to 512x512.
    Scales pixel values to [0.0, 1.0].
    NOTE: Does NOT apply ImageNet normalization here; the model applies mean/std internally.

    Returns:
        tensor: (1, 3, 512, 512) float32 tensor in range [0, 1].
        original_pil: Original EXIF-corrected PIL Image.
        pad_info: Dict containing scaling and padding metadata for inverse mapping.
    """
    original_pil = load_raw_image(image_input)
    orig_w, orig_h = original_pil.size

    # Calculate uniform scale factor
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))

    # Bilinear resize of original image
    resized_pil = original_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Centered zero-padding canvas
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized_pil, (left, top))

    # Scale to [0.0, 1.0]
    canvas_array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(canvas_array.transpose(2, 0, 1)).unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    pad_info = {
        "original_size": (orig_w, orig_h),
        "scaled_size": (new_w, new_h),
        "scale": scale,
        "padding": (left, top),
        "target_size": target_size
    }

    return tensor, original_pil, pad_info


def inverse_spatial_mapping(
    mask_canvas: np.ndarray,
    pad_info: Dict[str, Any]
) -> np.ndarray:
    """
    Inversely transforms the 512x512 predicted binary mask back to the original image dimensions:
    1. Unpads the centered region.
    2. Resizes unpadded mask to original (width, height) using Nearest-Neighbor interpolation.

    Args:
        mask_canvas: (512, 512) binary boolean or uint8 mask array.
        pad_info: Metadata dictionary produced during preprocessing.

    Returns:
        np.ndarray: (orig_h, orig_w) boolean binary mask matching original image dimensions.
    """
    left, top = pad_info["padding"]
    new_w, new_h = pad_info["scaled_size"]
    orig_w, orig_h = pad_info["original_size"]

    # Crop out the unpadded region
    cropped_mask = mask_canvas[top : top + new_h, left : left + new_w]

    # Convert to PIL and resize to original resolution using NEAREST interpolation
    cropped_pil = Image.fromarray((cropped_mask * 255).astype(np.uint8))
    full_mask_pil = cropped_pil.resize((orig_w, orig_h), Image.Resampling.NEAREST)

    return np.array(full_mask_pil) > 127
