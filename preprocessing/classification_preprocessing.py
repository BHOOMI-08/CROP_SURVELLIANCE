"""
Preprocessing Pipeline for PlantVillage Tomato Disease Classification.
Matches the exact evaluation transforms used in Plant_Village_FINAL.ipynb.
"""

from io import BytesIO
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from models.classification.config.classification_config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD
)


def get_classification_transforms() -> transforms.Compose:
    """
    Returns the exact evaluation transform composition from Cell 11 of notebook:
    1. Resize to (224, 224)
    2. Convert to PyTorch Tensor [0, 1]
    3. Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def load_image_as_pil(
    image_input: Union[str, Path, bytes, BytesIO, Image.Image, np.ndarray]
) -> Image.Image:
    """
    Loads and validates an image from multiple possible input sources,
    ensuring it is converted to RGB mode.

    Args:
        image_input: File path, bytes, BytesIO, PIL Image, or NumPy array.

    Returns:
        PIL.Image.Image: Loaded image in RGB format.

    Raises:
        FileNotFoundError: If a provided file path does not exist.
        ValueError: If input is invalid, corrupted, or cannot be parsed.
    """
    if image_input is None:
        raise ValueError("Image input cannot be None.")

    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path.resolve()}")
            with Image.open(path) as img:
                return img.convert("RGB")

        elif isinstance(image_input, bytes):
            with Image.open(BytesIO(image_input)) as img:
                return img.convert("RGB")

        elif isinstance(image_input, BytesIO):
            image_input.seek(0)
            with Image.open(image_input) as img:
                return img.convert("RGB")

        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("NumPy array is empty.")
            # If image has 2 dimensions (grayscale), or 3 dimensions (BGR/RGB)
            if image_input.ndim == 2:
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                # Handle OpenCV BGR if necessary, or standard RGB
                return Image.fromarray(image_input).convert("RGB")
            else:
                raise ValueError(f"Unsupported array shape: {image_input.shape}")

        elif hasattr(image_input, "read"):
            # File-like object (e.g., Streamlit UploadedFile)
            content = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            with Image.open(BytesIO(content)) as img:
                return img.convert("RGB")

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError, TypeError)):
            raise
        raise ValueError(f"Failed to load or validate image: {str(e)}") from e


def preprocess_classification_image(
    image_input: Union[str, Path, bytes, BytesIO, Image.Image, np.ndarray],
    device: torch.device = None
) -> torch.Tensor:
    """
    Prepares a single image for MobileNetV2 inference:
    1. Loads and validates image as RGB.
    2. Applies Resize(224, 224), ToTensor(), Normalize().
    3. Adds batch dimension: shape (1, 3, 224, 224).
    4. Places tensor on the requested device.

    Args:
        image_input: Image source.
        device: Target torch device (CPU or CUDA).

    Returns:
        torch.Tensor: Preprocessed batch tensor of shape (1, 3, 224, 224).
    """
    pil_image = load_image_as_pil(image_input)
    transform = get_classification_transforms()
    tensor = transform(pil_image)
    tensor_batch = tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

    if device is not None:
        tensor_batch = tensor_batch.to(device)

    return tensor_batch
