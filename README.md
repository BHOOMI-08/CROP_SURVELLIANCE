# Crop Disease and Pest Surveillance System

A professional, modular precision agriculture deep learning framework for real-time tomato leaf disease surveillance, combining classification, object detection, and semantic lesion segmentation into a single unified Python and Streamlit application.

---

## 🚀 System Architecture: The Three Verified Models

The system is built strictly around three genuine, trained deep learning models:

```
                                  [ Uploaded Tomato Leaf Image ]
                                                │
                     ┌──────────────────────────┼──────────────────────────┐
                     ▼                          ▼                          ▼
           [ Model 1: MobileNetV2 ]    [ Model 2: YOLO11n ]     [ Model 3: ResNet18 U-Net ]
                     │                          │                          │
              Whole-Leaf Disease       Lesion Bounding Boxes      Exact Pixel-Level Lesion
            Diagnosis & Confidence     & Localized Detection       Mask & Disease Overlay
```

| Model | Task | Architecture | Checkpoint | Trained Resolution | Benchmark Highlights |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Model 1** | **Classification** | MobileNetV2 (`Linear(1280, 10)`) | `best_mobilenetv2.pth` ($8.77\text{ MB}$) | $224 \times 224$ | **$92.14\%$ Test Accuracy**, $88.97\%$ Macro F1 |
| **Model 2** | **Detection** | YOLO11n ($101$ fused layers) | `best_tomato_detection.pt` ($5.25\text{ MB}$) | $640 \times 640$ | **$99.6\%$ Precision**, $99.2\%$ Recall, **$99.4\%$ mAP@50** |
| **Model 3** | **Segmentation** | ResNet18 U-Net | `best_segmentation_resnet18_unet_512.pth` ($165.03\text{ MB}$) | $512 \times 512$ | **$0.6640$ Macro Dice**, $0.5283$ Macro IoU, **$0.6955$ Global Dice** |

---

## 📁 Project Directory Structure

```text
CropDiseaseSurveillance/
│
├── app.py                                # Integrated Streamlit Web Application
├── requirements.txt                      # Project Dependencies
├── README.md                             # Comprehensive Documentation
├── .gitignore                            # Version Control Exclusions
│
├── models/
│   ├── classification/                   # Model 1 Artifacts
│   │   ├── checkpoints/best_mobilenetv2.pth
│   │   ├── architecture/mobilenetv2.py
│   │   ├── config/classification_config.py
│   │   └── class_mapping.json
│   │
│   ├── detection/                        # Model 2 Artifacts
│   │   ├── checkpoints/best_tomato_detection.pt
│   │   ├── config/detection_config.py
│   │   └── config/data.yaml
│   │
│   └── segmentation/                     # Model 3 Artifacts
│       ├── checkpoints/best_segmentation_resnet18_unet_512.pth
│       ├── architecture/resnet18_unet.py
│       └── config/segmentation_config.py
│
├── inference/
│   ├── __init__.py
│   ├── classification.py                 # MobileNetV2 Inference Engine
│   ├── detection.py                      # YOLO11n Inference Engine
│   ├── segmentation.py                   # ResNet18 U-Net Inference Engine
│   └── pipeline.py                       # Unified 3-Model Orchestrator
│
├── preprocessing/
│   ├── __init__.py
│   ├── classification_preprocessing.py   # RGB, 224x224, ImageNet Normalization
│   ├── detection_preprocessing.py        # Aspect Ratio & Validation
│   └── segmentation_preprocessing.py     # EXIF, 512x512 Centered Padding, Inverse Unpad
│
├── evaluation/                           # Authentic Evaluation Artifacts & Reports
│   ├── classification/                   # Reports, Metrics JSON, Comparisons
│   ├── detection/                        # Metrics JSON, Dataset Summaries
│   └── segmentation/                     # Dice/IoU Reports, Dataset Summaries
│
├── notebooks/                            # Original Training Notebooks
│   ├── classification/Plant_Village_FINAL.ipynb
│   ├── detection/yolooomodel2.ipynb
│   └── segmentation/segment_done.ipynb
│
├── reports/
│   └── evaluation_evidence/              # Genuine Visual Predictions & JSON Results
│       ├── test_tomato_leaf.jpg
│       ├── classification_prediction_visual.png
│       ├── detection/annotated_detection_result.jpg
│       └── segmentation/segmentation_prediction_visual.png
│
├── data/
│   └── sample_images/test_tomato_leaf.jpg # Verified Real Tomato Leaf Test Image
│
├── documentation/                        # Comprehensive Technical Reports
│   ├── model_architectures.md
│   ├── dataset_provenance.md
│   ├── methodology.md
│   └── limitations.md
│
└── tests/                                # Automated Test Suites (100% Pass)
    ├── test_classification.py
    ├── test_detection.py
    ├── test_segmentation.py
    └── test_pipeline.py
```

---

## 🛠️ Installation & Setup

### 1. Clone & Environment Activation
Ensure Python 3.10+ is installed:
```bash
cd E:/Deep_learning(PBL)/CropDiseaseSurveillance
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```
Core packages: `torch`, `torchvision`, `ultralytics`, `streamlit`, `pillow`, `numpy`, `matplotlib`, `pandas`.

---

## 🧪 Automated Testing Suite

All tests verify real checkpoint loading, strict key matching, forward-pass integrity, and live image inference:

```bash
# 1. Run Model 1 Classification Tests (6 tests)
python tests/test_classification.py

# 2. Run Model 2 Detection Tests (6 tests)
python tests/test_detection.py

# 3. Run Model 3 Segmentation Tests (5 tests)
python tests/test_segmentation.py

# 4. Run Unified Multi-Model Pipeline Tests (4 tests)
python tests/test_pipeline.py
```
**Total:** 21 automated unit tests with **$100\%$ PASS rate**.

---

## 🖥️ Running the Streamlit Web Application

Launch the dark-themed surveillance dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The application supports:
- Uploading any tomato leaf image (`JPG`, `PNG`).
- One-click testing with the verified sample leaf.
- Viewing individual predictions for classification, detection, and segmentation.
- Exploring multi-model combined visualizations and ground-truth benchmark metrics.
