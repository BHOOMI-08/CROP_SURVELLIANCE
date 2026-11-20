"""
ResNet18 U-Net Architecture Definition for Tomato Leaf Disease Segmentation.
Extracted directly and strictly from Cell 10 of segment_done.ipynb.
"""

from pathlib import Path
from typing import Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """U-Net decoder block with bilinear upsampling, skip connection concatenation, and ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet18UNet(nn.Module):
    """
    U-Net segmentation model with ImageNet-pretrained ResNet18 encoder.
    Includes registered buffers for ImageNet mean and std to perform normalization internally.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        encoder = resnet18(weights=weights)

        # Encoder stem and stages
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        # Decoder stages
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.decoder0 = DecoderBlock(64, 64, 64)

        # Final segmentation head (1 output channel for binary mask logits)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # Built-in normalization buffers matching training notebook
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Internal normalization using registered buffers
        x = (x - self.mean) / self.std

        # Encoder forward pass
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder forward pass with skip connections
        x = self.decoder3(x4, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        x = self.decoder0(x, x0)

        # Interpolate back to original input resolution
        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return self.final_conv(x)


def load_trained_resnet18_unet(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None
) -> ResNet18UNet:
    """
    Instantiates ResNet18UNet and strictly loads weights from the verified checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Target execution device (CPU or CUDA).

    Returns:
        ResNet18UNet: Model in eval mode with trained weights strictly loaded.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Segmentation checkpoint not found at: {checkpoint_path.resolve()}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Initialize architecture without downloading external weights
    model = ResNet18UNet(pretrained=False)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Strict state_dict loading
    load_result = model.load_state_dict(state_dict, strict=True)
    if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
        raise RuntimeError(
            f"Strict checkpoint loading failed: "
            f"Missing={load_result.missing_keys}, Unexpected={load_result.unexpected_keys}"
        )

    model.to(device)
    model.eval()
    return model
