# Limitations and Operational Boundaries

This document details known operational constraints, failure modes, and boundaries for all three models in the **Crop Disease and Pest Surveillance System**.

---

## 1. Model 1 Limitations: MobileNetV2 Disease Classification
1. **Single Leaf Laboratory Assumption:** Trained on isolated single leaf images with plain backgrounds (PlantVillage laboratory acquisition setup). It cannot parse cluttered whole-plant field scenes without prior localization.
2. **Species Boundary:** Specialized exclusively on Tomato (*Solanum lycopersicum*) foliage. Other crop leaves will be assigned to tomato classes based on highest logit.
3. **Symptom Confusion:** Necrotic spot diseases (`Early blight` and `Target Spot`) share visual overlap, resulting in minor class ambiguity.

---

## 2. Model 2 Limitations: YOLO11n Object Detection
1. **Scope of Detection Classes:**
   - Model 2 is trained on **foliar disease symptoms and healthy tomato leaves**, with exactly **one mite class** (`Spider_mites Two-spotted_spider_mite`).
   - It is **NOT** a general insect/pest detector. It cannot detect chewing beetles, whiteflies, aphids, caterpillars, or thrips, as none of these were included in the training labels. It should strictly be described as a **Tomato Foliar Disease & Mite Lesion Detector**.
2. **Bounding Box Overlaps in Heavy Clutter:**
   - In dense canopies with multiple intersecting leaves, non-maximum suppression (`iou_threshold=0.45`) may occasionally merge closely overlapping disease spots on adjoining leaves into a single bounding box.
3. **Resolution Sensitivity:**
   - Optimal detection occurs at $640 \times 640$ pixels. Extreme downscaling below $320 \times 320$ impairs small lesion detection (such as early Bacterial Spot speckles).

---

## 3. Model 3 Limitations: ResNet18 U-Net Foliar Lesion Segmentation
1. **Binary Segmentation Scope:**
   - Model 3 produces a **binary mask** distinguishing diseased foliar tissue from healthy tissue/background.
   - It does **NOT** classify individual disease types within the mask. The lesion category is provided by Model 1 and Model 2.
2. **Boundary Precision on Diffuse Chlorosis:**
   - Lesions with sharp necrotic boundaries (such as Bacterial Spot or Early Blight concentric rings) segment with high precision ($>75\%$).
   - Diffuse, gradual chlorotic halos (yellowing from viral infections or early spider mite stippling) have fuzzy transitions that may lead to lower recall under the fixed $0.50$ threshold.
3. **Resolution Padding Overhead:**
   - Images with extreme aspect ratios (e.g. $1:4$ or $4:1$) experience substantial zero-padding to reach the $512 \times 512$ canvas, reducing the effective pixel density of the leaf surface.
4. **Local Dataset Availability:**
   - The official 133 test image-mask pairs reside on the Kaggle training workspace (`/kaggle/input/datasets/bhoomi0805/segmentation-data/plantseg/test/`). Offline local verification was performed using the user-provided sample leaf and automated model structure/unit tests. Bulk on-device re-evaluation across all 133 pairs remains pending local test directory placement.
