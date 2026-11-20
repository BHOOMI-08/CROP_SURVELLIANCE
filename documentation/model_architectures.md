# Model Architectures

This document details the exact neural network architectures utilized in the **Crop Disease and Pest Surveillance System**.

---

## Model 1: MobileNetV2 — PlantVillage Tomato Disease Classification

### 1. Architectural Overview
- **Base Architecture:** `torchvision.models.mobilenet_v2`
- **Initial Weights:** Pretrained on ImageNet-1K (`MobileNet_V2_Weights.DEFAULT`), with the convolutional backbone frozen during training.
- **Input Dimensions:** $(B, 3, 224, 224)$ where $B$ is the batch size.
- **Feature Extractor:** Inverted Residual Blocks (Sandglass design) utilizing depthwise separable convolutions:
  - 1x1 expansion convolution (increasing channel depth)
  - 3x3 depthwise convolution (spatial filtering with minimal parameters)
  - Batch normalization and ReLU6 activation ($\min(\max(0, x), 6)$)
  - 1x1 linear bottleneck projection (reducing dimensionality without non-linear activation)
  - Residual skip connections when input and output spatial dimensions match.
- **Output Channels from Backbone:** $1,280$ channels (`mobilenet.last_channel`).

### 2. Classifier Head Specification
The original 1,000-class ImageNet classifier was replaced with a custom head designed for 10 tomato disease classes:
```
mobilenet.classifier = Sequential(
    (0): Dropout(p=0.2, inplace=False),
    (1): Linear(in_features=1280, out_features=10, bias=True)
)
```
- **Trainable Parameters:** $12,810$ parameters in the classifier head ($10 \times 1280 + 10$).
- **Total State Dict Keys:** $314$ tensor keys in `best_mobilenetv2.pth`.
- **Checkpoint File:** `models/classification/checkpoints/best_mobilenetv2.pth` ($8.77\text{ MB}$).

---

## Model 2: YOLO11n — Tomato Disease & Lesion Object Detection

### 1. Architectural Overview
- **Base Architecture:** `ultralytics.nn.tasks.DetectionModel` (YOLO11 nano scale, `yolo11n.yaml`)
- **Framework & Version:** Ultralytics `8.4.215`, PyTorch `2.13.0`
- **Checkpoint File:** `models/detection/checkpoints/best_tomato_detection.pt` ($5.25\text{ MB}$)
- **Layer Complexity:**
  - Raw Graph: $182$ layers, $2,607,070$ parameters, $6.6\text{ GFLOPs}$
  - Fused Inference Mode: $101$ layers, $2,599,238$ parameters, $6.4\text{ GFLOPs}$
- **Strides:** $[8.0, 16.0, 32.0]$ across P3, P4, P5 feature pyramids.
- **Input Resolution:** $(B, 3, 640, 640)$

### 2. Architecture Modules
- **Backbone:**
  - Focus/Conv stem with $3\times 3$ kernel, stride $2$.
  - C3k2 modules (Cross Stage Partial with 2 convolutions, optimized kernel scale).
  - Spatial Pyramid Pooling - Fast (`SPPF`) with $5\times 5$ pooling kernel.
  - C2PSA (`Cross-Stage Partial with Polarized Self-Attention`) for enhanced spatial saliency.
- **Neck & Detection Head:**
  - Feature Pyramid Network (FPN) + Path Aggregation Network (PAN) with bidirectional feature flow.
  - Anchor-free decoupled detection head separating bounding-box regression (Distribution Focal Loss, `dfl=1.5`, `box=7.5`) from classification (`cls=0.5`).
  - Output Classes: Exactly $10$ classes ($nc=10$).

---

## Model 3: ResNet18 U-Net — Binary Tomato Foliar Lesion Segmentation

### 1. Architectural Overview
- **Base Architecture:** U-Net encoder-decoder with an ImageNet-pretrained ResNet18 backbone.
- **Checkpoint File:** `models/segmentation/checkpoints/best_segmentation_resnet18_unet_512.pth` ($165.03\text{ MB}$)
- **Checkpoint Structure:** Nested dictionary with `model_state_dict` containing $174$ tensor keys.
- **Total Parameters:** $14,404,225$ parameters.
- **Native Resolution:** $(B, 3, 512, 512)$

### 2. Architecture Modules
- **Encoder (ResNet18):**
  - Stem: $7\times 7$ Conv ($64$ filters, stride $2$) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool ($3\times 3$, stride $2$).
  - Layer 1: $2$ BasicBlocks ($64$ channels).
  - Layer 2: $2$ BasicBlocks ($128$ channels, stride $2$).
  - Layer 3: $2$ BasicBlocks ($256$ channels, stride $2$).
  - Layer 4 (Bottleneck): $2$ BasicBlocks ($512$ channels, stride $2$).
- **Decoder Blocks:**
  - `DecoderBlock 3`: Upsamples $512$ channels to $256$ using bilinear interpolation (`align_corners=False`), concatenates with Layer 3 skip ($256$ channels), followed by `ConvBlock(768, 256)`.
  - `DecoderBlock 2`: Upsamples $256$ channels, concatenates with Layer 2 skip ($128$ channels), followed by `ConvBlock(384, 128)`.
  - `DecoderBlock 1`: Upsamples $128$ channels, concatenates with Layer 1 skip ($64$ channels), followed by `ConvBlock(192, 64)`.
  - `DecoderBlock 0`: Upsamples $64$ channels, concatenates with stem skip ($x_0$, $64$ channels), followed by `ConvBlock(128, 64)`.
- **Segmentation Head:**
  - $3\times 3$ Conv ($64 \rightarrow 32$ filters, padding $1$) $\rightarrow$ ReLU $\rightarrow$ $1\times 1$ Conv ($32 \rightarrow 1$ filter).
- **Internal Normalization Buffers:**
  - Model registers `mean` ($[0.485, 0.456, 0.406]$) and `std` ($[0.229, 0.224, 0.225]$) as buffers to perform ImageNet standardization directly inside the forward pass.
