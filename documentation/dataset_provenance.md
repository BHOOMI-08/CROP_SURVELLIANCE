# Dataset Provenance and Integrity

This document outlines the source, deduplication procedure, and leakage-safe data splitting methodology used for the three surveillance models.

---

## 1. Model 1 Dataset: PlantVillage Tomato Subset (Classification)
- **Source:** PlantVillage dataset (Tomato subset), processed via Kaggle environment (`Plant_Village_FINAL.ipynb`).
- **Raw Image Count:** $14,006$ images across 10 tomato leaf disease and healthy categories.
- **Corrupted Image Audit:** Verified with `PIL.Image.verify()`; $0$ corrupted files detected.
- **Deduplication Procedure:** Full-file MD5 hash computation (`hashlib.md5(f.read()).hexdigest()`) identified $11$ duplicate image copies.
- **Clean Unique Dataset Size:** $13,995$ unique images.
- **Partitions:**
  - Training ($70\%$): $9,796$ images
  - Validation ($15\%$): $2,099$ images
  - Testing ($15\%$): $2,100$ images
- **Leakage Verification:** $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$.

---

## 2. Model 2 Dataset: Roboflow Tomato Leaf Bounding-Box Dataset (Detection)
- **Source:** Roboflow Universe (`tomato-leaf-detection-u6mou-iisg2`, version 1, license CC BY 4.0).
- **Workspace:** `quynh-nguyen-tnu6q`
- **Project URL:** https://universe.roboflow.com/quynh-nguyen-tnu6q/tomato-leaf-detection-u6mou-iisg2/dataset/1
- **Domain Focus:** Bounding-box localized tomato foliar lesions and healthy foliage.
- **Annotations:** YOLO format bounding boxes normalized to $[0.0, 1.0]$: `class_id center_x center_y width height`.
- **Deduplication Audit:** A duplicate validation image (`Tomato___Leaf_Mold_original_329f2f31-ceca-402f-a007-21d64fe75f5e...`) was explicitly identified and excluded from the validation partition during preprocessing in Cell 4 of `yolooomodel2.ipynb`.
- **Partitions (Cleaned & Verified):**
  - **Training Partition:** $6,928$ images and $6,928$ bounding-box label files.
  - **Validation Partition:** $1,969$ images and $1,969$ bounding-box label files.
  - **Held-out Test Partition:** $997$ images and $997$ bounding-box label files ($998$ annotated instances).
  - **Total Dataset Size:** $9,894$ images and $9,894$ label files.
- **Classes ($nc=10$):** Exactly $10$ classes. No 97 or 102 classes exist in this model or dataset.

---

## 3. Model 3 Dataset: PlantSeg Tomato Disease Subset (Segmentation)
- **Source:** PlantSeg agricultural segmentation dataset (Tomato subset), processed via Kaggle (`segment_done.ipynb`).
- **Data Location:** `/kaggle/input/datasets/bhoomi0805/segmentation-data/plantseg`
- **Annotation Format:** Grayscale 8-bit PNG binary masks matching image spatial dimensions ($mask > 0$ denotes diseased leaf area).
- **Audit & EXIF Handling:** In Cell 6 and 7 of `segment_done.ipynb`, EXIF orientation flags ($274$) caused dimension mismatches between raw camera orientation and ground-truth masks. Applying `ImageOps.exif_transpose()` resolved all dimension mismatches, achieving $100\%$ spatial alignment across all partitions.
- **Partitions (Matched Image-Mask Pairs):**
  - **Training Partition:** $460$ image-mask pairs.
  - **Validation Partition:** $74$ image-mask pairs.
  - **Held-out Test Partition:** $133$ image-mask pairs.
  - **Total Matched Pairs:** $667$ pairs ($0$ missing masks, $0$ missing images).
