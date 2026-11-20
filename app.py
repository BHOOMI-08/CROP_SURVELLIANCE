"""
Crop Disease and Pest Surveillance System — Unified Streamlit Application.
Integrates all three genuine trained deep learning models:
  1. Classification: MobileNetV2 (PlantVillage tomato dataset)
  2. Detection: YOLO11n (Tomato disease & foliar lesion detection)
  3. Segmentation: ResNet18 U-Net (PlantSeg foliar lesion segmentation)
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import CropSurveillancePipeline
from models.classification.config.classification_config import VERIFIED_METRICS as CLF_METRICS
from models.detection.config.detection_config import VERIFIED_TEST_METRICS as DET_METRICS
from models.segmentation.config.segmentation_config import VERIFIED_TEST_METRICS as SEG_METRICS

# Streamlit Page Configuration
st.set_page_config(
    page_title="Crop Disease Surveillance System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark-Themed Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2ecc71;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #bdc3c7;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e272e;
        border: 1px solid #34495e;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-green { background-color: #27ae60; color: white; }
    .badge-blue { background-color: #2980b9; color: white; }
    .badge-purple { background-color: #8e44ad; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading verified deep learning models...")
def get_pipeline() -> CropSurveillancePipeline:
    """Instantiates and caches the unified surveillance pipeline."""
    return CropSurveillancePipeline(
        load_classification=True,
        load_detection=True,
        load_segmentation=True
    )


def main():
    # Header
    st.markdown('<div class="main-header">🌱 Precision Agriculture: Crop Disease Surveillance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Task Deep Learning Framework for Real-Time Tomato Leaf Pathology</div>', unsafe_allow_html=True)

    # Sidebar: Model status and controls
    with st.sidebar:
        st.header("⚙️ Surveillance Controls")
        
        st.subheader("Model Status")
        st.markdown("""
        - **Model 1:** <span class="status-badge badge-green">MobileNetV2</span>
        - **Model 2:** <span class="status-badge badge-blue">YOLO11n</span>
        - **Model 3:** <span class="status-badge badge-purple">ResNet18 U-Net</span>
        """, unsafe_allow_html=True)
        st.divider()

        st.subheader("Inference Settings")
        detection_conf = st.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
        seg_threshold = st.slider("Segmentation Mask Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
        
        st.divider()
        st.subheader("Input Selection")
        sample_img_path = PROJECT_ROOT / "data" / "sample_images" / "test_tomato_leaf.jpg"
        use_sample = False
        if sample_img_path.is_file():
            use_sample = st.checkbox("Use Verified Sample Leaf (Early Blight)", value=True)

        uploaded_file = st.file_uploader(
            "Or Upload Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of a tomato leaf for multi-model inspection."
        )

    # Resolve active image
    active_image = None
    image_name = ""

    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading uploaded image: {str(e)}")
            return
    elif use_sample and sample_img_path.is_file():
        active_image = Image.open(sample_img_path).convert("RGB")
        image_name = "test_tomato_leaf.jpg (Verified Sample)"

    if active_image is None:
        st.info("👆 Please upload a tomato leaf image or enable the verified sample leaf in the sidebar to begin surveillance.")
        return

    # Load Pipeline
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Failed to load surveillance pipeline: {str(e)}")
        return

    # Run Analysis Tabs
    tabs = st.tabs([
        "🔬 Combined Surveillance",
        "🏷️ Model 1: Classification",
        "🎯 Model 2: Detection",
        "🧩 Model 3: Segmentation",
        "📊 Model Provenance & Metrics"
    ])

    # 1. Combined Surveillance Tab
    with tabs[0]:
        st.subheader(f"Multi-Model Analysis: `{image_name}`")
        col_preview, col_clf, col_det = st.columns([1, 1, 1])

        with col_preview:
            st.image(active_image, caption=f"Input Image ({active_image.width}x{active_image.height})", use_container_width=True)

        with st.spinner("Executing combined multi-model inference..."):
            results = pipeline.analyze_image(
                active_image,
                detection_conf=detection_conf,
                segmentation_threshold=seg_threshold
            )

        clf_res = results["classification"]
        det_res = results["detection"]
        seg_res = results["segmentation"]

        with col_clf:
            st.markdown("#### 1. Classification (MobileNetV2)")
            st.markdown(f"**Predicted Diagnosis:** `{clf_res['predicted_class']}`")
            st.markdown(f"**Confidence:** `{clf_res['confidence_percentage']}`")
            st.progress(clf_res["confidence"])

            st.markdown("#### 3. Segmentation (ResNet18 U-Net)")
            st.markdown(f"**Affected Leaf Area:** `{seg_res['diseased_area_percentage']}%`")
            st.markdown(f"**Lesion Pixel Count:** `{seg_res['diseased_pixels']:,} / {seg_res['total_pixels']:,}`")
            st.markdown(f"**Mean Lesion Confidence:** `{seg_res['mean_confidence_diseased']*100:.1f}%`")

        with col_det:
            st.markdown("#### 2. Detection (YOLO11n)")
            st.markdown(f"**Lesions/Objects Found:** `{det_res['num_detections']}`")
            if det_res["num_detections"] > 0:
                for idx, d in enumerate(det_res["detections"]):
                    st.markdown(f"- **{d['class_name']}**: `{d['confidence_percentage']}` (Box: `{d['box_xyxy']}`)")
            else:
                st.info("No localized lesion bounding boxes met the detection confidence threshold.")

        st.divider()
        st.subheader("Visualizations")
        col_vis1, col_vis2, col_vis3 = st.columns(3)
        with col_vis1:
            st.image(active_image, caption="Original Input Leaf", use_container_width=True)
        with col_vis2:
            st.image(det_res["annotated_image"], caption=f"YOLO11n Detections ({det_res['num_detections']} localized)", use_container_width=True)
        with col_vis3:
            st.image(seg_res["overlay_image"], caption=f"U-Net Lesion Overlay ({seg_res['diseased_area_percentage']}% affected)", use_container_width=True)

    # 2. Classification Tab
    with tabs[1]:
        st.subheader("Model 1: MobileNetV2 Disease Classification")
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.image(active_image, caption=f"Classified Image: {image_name}", use_container_width=True)
            st.markdown(f"### Predicted: `{clf_res['predicted_class']}`")
            st.markdown(f"**Top Confidence:** `{clf_res['confidence_percentage']}`")

        with c2:
            st.markdown("#### 10-Class Probability Distribution")
            sorted_probs = sorted(clf_res["all_probabilities"].items(), key=lambda x: x[1], reverse=True)
            for cls_name, prob in sorted_probs:
                col_name, col_bar, col_val = st.columns([2, 3, 1])
                with col_name:
                    is_pred = (cls_name == clf_res["predicted_class"])
                    prefix = "👉 **" if is_pred else ""
                    suffix = "**" if is_pred else ""
                    st.markdown(f"{prefix}{cls_name.replace('Tomato___', '')}{suffix}")
                with col_bar:
                    st.progress(float(prob))
                with col_val:
                    st.markdown(f"`{prob*100:.2f}%`")

    # 3. Detection Tab
    with tabs[2]:
        st.subheader("Model 2: YOLO11n Object & Lesion Detection")
        d1, d2 = st.columns([1.2, 1])
        with d1:
            st.image(det_res["annotated_image"], caption="Annotated Detection Output", use_container_width=True)
        with d2:
            st.markdown(f"**Inference Latency:** `{det_res['inference_time_ms']} ms`")
            st.markdown(f"**Detections Identified:** `{det_res['num_detections']}`")
            if det_res["num_detections"] > 0:
                det_table = [
                    {
                        "Class": d["class_name"],
                        "Confidence": d["confidence_percentage"],
                        "Box [x1, y1, x2, y2]": str(d["box_xyxy"])
                    }
                    for d in det_res["detections"]
                ]
                st.table(det_table)
            else:
                st.warning("No disease spots or leaves detected above the selected confidence threshold.")

    # 4. Segmentation Tab
    with tabs[3]:
        st.subheader("Model 3: ResNet18 U-Net Disease Lesion Segmentation")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.image(active_image, caption="Original Input", use_container_width=True)
        with s2:
            st.image(seg_res["mask_image"], caption="Binary Segmentation Mask (White = Lesion)", use_container_width=True)
        with s3:
            st.image(seg_res["overlay_image"], caption="Red Disease Overlay", use_container_width=True)

        st.markdown(f"""
        - **Diseased Foliar Area:** `{seg_res['diseased_area_percentage']}%`
        - **Positive Mask Pixels:** `{seg_res['diseased_pixels']:,}` of `{seg_res['total_pixels']:,}` total pixels
        - **Mean Lesion Probability:** `{seg_res['mean_confidence_diseased']*100:.2f}%`
        - **Decision Threshold:** `{seg_res['threshold_used']}`
        - **Inference Latency:** `{seg_res['inference_time_ms']} ms`
        """)

    # 5. Provenance & Metrics Tab
    with tabs[4]:
        st.subheader("Authentic Training & Evaluation Benchmarks")
        st.markdown("All metrics below are extracted from the executed training notebooks on held-out test splits.")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("### Model 1: Classification")
            st.markdown("**Architecture:** MobileNetV2")
            st.markdown("**Dataset:** PlantVillage Tomato (Deduplicated)")
            st.markdown(f"- Test Accuracy: **{CLF_METRICS['test_accuracy']*100:.2f}%**")
            st.markdown(f"- Macro Precision: **{CLF_METRICS['macro_precision']*100:.2f}%**")
            st.markdown(f"- Macro Recall: **{CLF_METRICS['macro_recall']*100:.2f}%**")
            st.markdown(f"- Macro F1-Score: **{CLF_METRICS['macro_f1']*100:.2f}%**")
            st.markdown(f"- Test Samples: **{CLF_METRICS['test_samples']:,}**")

        with m2:
            st.markdown("### Model 2: Detection")
            st.markdown("**Architecture:** YOLO11n")
            st.markdown("**Dataset:** Roboflow Tomato Leaf Bounding Box")
            st.markdown(f"- Box Precision: **{DET_METRICS['precision']*100:.1f}%**")
            st.markdown(f"- Box Recall: **{DET_METRICS['recall']*100:.1f}%**")
            st.markdown(f"- mAP@50: **{DET_METRICS['mAP50']*100:.1f}%**")
            st.markdown(f"- mAP@50-95: **{DET_METRICS['mAP50_95']*100:.1f}%**")
            st.markdown(f"- Test Images: **{DET_METRICS['test_images']:,}**")

        with m3:
            st.markdown("### Model 3: Segmentation")
            st.markdown("**Architecture:** ResNet18 U-Net")
            st.markdown("**Dataset:** PlantSeg Tomato Subset")
            st.markdown(f"- Macro Dice: **{SEG_METRICS['macro_dice']:.4f}**")
            st.markdown(f"- Macro IoU: **{SEG_METRICS['macro_iou']:.4f}**")
            st.markdown(f"- Global Dice: **{SEG_METRICS['global_dice']:.4f}**")
            st.markdown(f"- Global IoU: **{SEG_METRICS['global_iou']:.4f}**")
            st.markdown(f"- Precision: **{SEG_METRICS['precision']:.4f}**")
            st.markdown(f"- Recall: **{SEG_METRICS['recall']:.4f}**")
            st.markdown(f"- Test Images: **{SEG_METRICS['test_images']:,}**")


if __name__ == "__main__":
    main()
