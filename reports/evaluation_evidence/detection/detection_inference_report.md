# Model 2 (YOLO11n Tomato Object Detection) Audit & Inference Report

**Project:** Crop Disease and Pest Surveillance System  
**Model 2:** YOLO11n Object Detection (`best_tomato_detection.pt`)  
**Task Type:** Bounding-box disease and foliar lesion localization  

---

## 1. Checkpoint & Architecture Verification

- **Checkpoint File:** `models/detection/checkpoints/best_tomato_detection.pt`
- **File Size:** $5.25\text{ MB}$ ($5,502,741\text{ bytes}$)
- **YOLO Version:** YOLO11 nano (`scale: n`, `yaml_file: yolo11n.yaml`)
- **Layers & Parameters:** $182$ layers ($101$ fused), $2,599,238$ parameters, $6.4\text{ GFLOPs}$
- **Ultralytics Version:** Built and validated on `8.4.215`
- **Task Type:** `detect` (`ultralytics.nn.tasks.DetectionModel`)
- **Classes ($nc=10$):**
  - `0: Bacterial_spot`
  - `1: Early_blight`
  - `2: Late_blight`
  - `3: Leaf_Mold`
  - `4: Septoria_leaf_spot`
  - `5: Spider_mites Two-spotted_spider_mite`
  - `6: Target_Spot`
  - `7: Tomato_Yellow_Leaf_Curl_Virus`
  - `8: Tomato__healthy`
  - `9: Tomato_mosaic_virus`
- **Class Count Investigation:** Verified strictly as $nc=10$. No 97 or 102 classes exist in the trained model, YAML, or training notebook.

---

## 2. Dataset Split Verification (from Notebook Cell 4)

- **Training Split:** $6,928$ images and $6,928$ label files
- **Validation Split:** $1,969$ images and $1,969$ label files ($1$ duplicate validation image removed)
- **Held-out Test Split:** $997$ images and $997$ label files ($998$ annotated instances)
- **Total Dataset:** $9,894$ images and $9,894$ label files

---

## 3. Evaluation Metrics Audit on Test Split (from Notebook Cell 9)

Evaluated across $997$ images ($998$ ground-truth instances) on the held-out test split:

| Metric | Previously Reported | Notebook Cell 9 Ground-Truth | Status |
| :--- | :---: | :---: | :---: |
| **Box Precision ($P$)** | $99.6\%$ | **$99.6\%$** ($0.996$) | **VERIFIED** |
| **Box Recall ($R$)** | $99.2\%$ | **$99.2\%$** ($0.992$) | **VERIFIED** |
| **mAP@50** | $99.4\%$ | **$99.4\%$** ($0.994$) | **VERIFIED** |
| **mAP@50–95** | $94.6\%$ | **$94.6\%$** ($0.946$) | **VERIFIED** |

---

## 4. Real-Image Inference Verification

Inference was executed on the available user-provided tomato leaf sample (`test_tomato_leaf.jpg`):

- **Input Sample:** `reports/evaluation_evidence/detection/test_input_leaf.jpg`
- **Annotated Output:** `reports/evaluation_evidence/detection/annotated_detection_result.jpg`
- **Structured JSON Result:** `reports/evaluation_evidence/detection/detection_prediction_result.json`
- **Inference Summary:**
  - **Detections:** $1$ bounding box detected.
  - **Class:** `Early_blight` (Class ID: `1`)
  - **Confidence:** **$89.72\%$** ($0.8972$)
  - **Bounding Box ($x_1, y_1, x_2, y_2$):** `[30.09, 5.39, 230.51, 239.77]`
  - **Inference Latency:** $\approx 202.5\text{ ms}$ (CPU)
  - **Cross-Model Alignment:** Matches Model 1 (MobileNetV2), which classified this exact leaf as `Tomato___Early_blight` with $93.94\%$ confidence.

---

## 5. Automated Unit Tests

All 6 automated tests in `tests/test_detection.py` and 3 tests in `tests/test_pipeline.py` passed with zero errors:
1. Checkpoint existence & integrity: **PASS**
2. Architecture & class mapping: **PASS**
3. Inference output structure & bounding boxes: **PASS**
4. Empty detection handling: **PASS**
5. Visualization generation: **PASS**
6. Model 1 isolation (unaffected): **PASS**
7. Pipeline integration: **PASS**
