"""
Preprocessing pipeline for Model 2: YOLO11n Tomato Object Detection.
Validates and formats input images for Ultralytics YOLO inference.
"""

from io import BytesIO
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


def load_detection_image(
    image_input: Union[str, Path, bytes, BytesIO, Image.Image, np.ndarray]
) -> Image.Image:
    """
    Validates and loads an image into PIL RGB format suitable for YOLO detection.

    Args:
        image_input: Image source (path, bytes, PIL image, or NumPy array).

    Returns:
        PIL.Image.Image: Loaded RGB image.

    Raises:
        FileNotFoundError: If a file path does not exist.
        ValueError: If image data is empty, corrupted, or unsupported.
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
                return img.convert("RGB")

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError, TypeError)):
            raise
        raise ValueError(f"Failed to load or validate image: {str(e)}") from e
