# Methodology

This document outlines the training protocol, preprocessing pipeline, and evaluation methodology implemented for all three surveillance models.

---

## 1. Model 1 Methodology: MobileNetV2 Classification

- **Preprocessing:** RGB mode, resize to $(224, 224)$, ToTensor, ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
- **Training Protocol:** 10 epochs, Adam optimizer (lr $= 0.001$, weight decay $= 10^{-4}$), batch size 32, class-weighted cross-entropy loss, frozen BatchNorm layers.
- **Evaluation Results (2,100 Test Samples):**
  - Test Accuracy: **$92.14\%$**
  - Macro Precision: **$88.39\%$**
  - Macro Recall: **$90.07\%$**
  - Macro F1-Score: **$88.97\%$**
  - Best Validation Accuracy: **$92.42\%$**

---

## 2. Model 2 Methodology: YOLO11n Object Detection

- **Preprocessing & Augmentation:** Input resolution $640 \times 640$, letterboxing, online HSV jitter, translation, scale jitter, mosaic augmentation for epochs 1–20 (closed for final 10 epochs via `close_mosaic=10`), RandAugment, random erasing ($p=0.4$).
- **Training Protocol:** 30 epochs on Tesla T4, AdamW (`lr0=0.01`, warmup to `lr0=0.000714`), CIoU / DFL box loss (`box=7.5`, `dfl=1.5`) and binary cross-entropy classification loss (`cls=0.5`).
- **Evaluation Results (997 Held-Out Test Images, 998 Instances):**
  - Box Precision: **$99.6\%$** ($0.996$)
  - Box Recall: **$99.2\%$** ($0.992$)
  - mAP@50: **$99.4\%$** ($0.994$)
  - mAP@50–95: **$94.6\%$** ($0.946$)

---

## 3. Model 3 Methodology: ResNet18 U-Net Binary Segmentation

### Preprocessing & Spatial Pipeline
- **EXIF Handling:** Automated EXIF rotation alignment using `PIL.ImageOps.exif_transpose()`.
- **Aspect-Ratio Preserving Resize:** Uniform scaling factor $s = \min(512/W, 512/H)$, resizing image with bilinear interpolation and mask with nearest-neighbor interpolation.
- **Centered Zero-Padding:** Pasted onto a $(512, 512)$ canvas with offsets $(\lfloor(512-W')/2\rfloor, \lfloor(512-H')/2\rfloor)$.
- **Internal Normalization:** Normalized within the forward pass using registered ImageNet `mean` and `std` buffers.
- **Inverse Spatial Mapping:** Crops unpadded canvas prediction and resizes using nearest-neighbor interpolation to recover original image dimensions.

### Training Protocol
- **Epochs & Schedule:** Up to 35 epochs with early stopping (`patience=7`) and `ReduceLROnPlateau` scheduler (factor $= 0.5$, patience $= 2$).
- **Best Validation Checkpoint:** Saved at **Epoch 18** with validation Dice $= \mathbf{0.6795}$ ($0.679510027$).
- **Loss Function:** Combined Binary Cross-Entropy with Logits + Soft Dice Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(z, y) + \left(1 - \frac{2 \sum \sigma(z) y + 1}{\sum \sigma(z) + \sum y + 1}\right)$$
- **Precision:** Mixed Precision (`torch.cuda.amp.autocast`) with GradScaler.

### Evaluation Protocol & Verified Results (133 Held-Out Test Images)
Evaluated at standard decision threshold $\tau = 0.50$ (Cell 13 of `segment_done.ipynb`):
- **Macro Dice:** **$0.6640$**
- **Macro IoU:** **$0.5283$**
- **Global Pixel-Level Dice:** **$0.6955$**
- **Global Pixel-Level IoU:** **$0.5332$**
- **Pixel Precision:** **$0.7556$** ($75.56\%$)
- **Pixel Recall:** **$0.6442$** ($64.42\%$)
- **Best Validation Dice:** **$0.6795$** ($67.95\%$ at epoch 18)
