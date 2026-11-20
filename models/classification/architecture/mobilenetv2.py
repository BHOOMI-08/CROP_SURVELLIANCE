"""
MobileNetV2 Architecture Definition for Tomato Disease Classification.
Constructed to match the exact architecture trained in Plant_Village_FINAL.ipynb.
"""

from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


def build_mobilenetv2_classifier(num_classes: int = 10) -> nn.Module:
    """
    Builds the exact MobileNetV2 architecture used in Plant_Village_FINAL.ipynb.

    Args:
        num_classes: Number of target disease classes (default: 10).

    Returns:
        nn.Module: MobileNetV2 with custom linear classifier head.
    """
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_trained_mobilenetv2(
    checkpoint_path: Union[str, Path],
    num_classes: int = 10,
    device: torch.device = None
) -> nn.Module:
    """
    Builds the MobileNetV2 model and loads weights from the verified checkpoint.

    Args:
        checkpoint_path: Path to the trained checkpoint (.pth file).
        num_classes: Number of output classes (default: 10).
        device: torch device to place model on (defaults to CPU/CUDA if None).

    Returns:
        nn.Module: Model in evaluation mode with trained weights loaded.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found at: {checkpoint_path.resolve()}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_mobilenetv2_classifier(num_classes=num_classes)
    
    # Load state dict strictly from the trained checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # If checkpoint contains metadata wrapper, extract state_dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    load_result = model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    return model
