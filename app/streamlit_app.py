import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import io
from src.inference import predict, CLASS_LABELS, CLASS_COLORS

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title = "PCB Defect Detector",
    layout     = "wide"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'best.pt'
)

# ─────────────────────────────────────────────
# Cache model — loads once, reused for all uploads
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("PCB Defect Detector")
    st.caption("YOLOv8s — DeepPCB Benchmark")
    st.markdown("---")

    st.markdown("#### Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value = 0.10,
        max_value = 0.90,
        value     = 0.25,
        step      = 0.05,
        help      = "Lower value detects more defects but may include "
                    "false positives. Higher value only shows strong detections."
    )
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value = 0.10,
        max_value = 0.90,
        value     = 0.45,
        step      = 0.05,
        help      = "Controls how aggressively overlapping boxes are merged."
    )

    st.markdown("---")
    st.markdown("#### Model Performance")
    c1, c2 = st.columns(2)
    c1.metric("mAP@50",    "95.30%")
    c2.metric("Precision", "94.16%")
    c1.metric("Recall",    "90.58%")
    c2.metric("mAP@50-95", "56.81%")
    st.caption("Trained on DeepPCB · 1500 images · 6 classes")

    st.markdown("---")
    st.markdown("#### Defect Classes")
    for cls, color in CLASS_COLORS.items():
        hex_color = '#%02x%02x%02x' % color
        st.markdown(
            f'<span style="color:{hex_color}; font-weight:600;">'
            f'&#9632;</span>&nbsp; {cls.replace("_", " ").title()}',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "**Dataset:** [DeepPCB](https://github.com/tangsanli5201/DeepPCB)  \n"
        "**Model:** YOLOv8s (Ultralytics)  \n"
        "**Notebook:** [View on Kaggle](https://www.kaggle.com/ahsanneural)  \n"
        "**Author:** [Muhammad Ahsan](https://www.kaggle.com/ahsanneural)"
    )

# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────
st.title("PCB Defect Detection")
st.markdown(
    "Upload a PCB board image to automatically detect and classify "
    "manufacturing defects. Model trained on the "
    "[DeepPCB benchmark dataset](https://github.com/tangsanli5201/DeepPCB) "
    "achieving **95.3% mAP@50** across 6 defect types."
)
st.markdown("---")

# ─────────────────────────────────────────────
# Preload model
# ─────────────────────────────────────────────
with st.spinner("Loading model..."):
    load_model()

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a PCB Image",
    type = ["jpg", "jpeg", "png"],
    help = "Upload any PCB board image in JPG or PNG format."
)

# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Running defect detection..."):
        output = predict(
            image_path = tmp_path,
            model_path = MODEL_PATH,
            conf       = conf_threshold,
            iou        = iou_threshold,
            device     = "cpu"
        )
    os.unlink(tmp_path)

    total = output["count"]

    if total == 0:
        st.success("No defects detected above the current confidence threshold.")
    else:
        st.error(f"{total} defect(s) detected on this PCB board.")

    st.markdown("---")

    col_img, col_stats = st.columns([3, 2], gap="large")

    with col_img:
        st.markdown("#### Detection Result")
        st.image(
            output["annotated"],
            caption          = f"{total} defect(s) detected  |  "
                               f"conf: {conf_threshold}  |  iou: {iou_threshold}",
            use_column_width = True
        )

        with st.expander("Show original image"):
            uploaded_file.seek(0)
            st.image(uploaded_file.read(), use_column_width=True)

        pil_img = Image.fromarray(output["annotated"])
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        st.download_button(
            label     = "Download Annotated Image",
            data      = buf.getvalue(),
            file_name = "pcb_defect_result.png",
            mime      = "image/png"
        )

    with col_stats:
        st.markdown("#### Defect Summary")

        per_class   = output["per_class"]
        class_names = [c.replace("_", " ").title() for c in CLASS_LABELS]
        counts      = [per_class[c] for c in CLASS_LABELS]
        colors_norm = [tuple(v/255 for v in CLASS_COLORS[c]) for c in CLASS_LABELS]

        fig, ax = plt.subplots(figsize=(5, 3.8))
        bars = ax.barh(class_names, counts, color=colors_norm,
                       edgecolor="#222233", height=0.55)
        ax.set_xlabel("Count", fontsize=9, color="white")
        ax.set_title("Defects per Class", fontsize=10,
                     fontweight="bold", color="white")
        ax.set_xlim(0, max(counts) + 1 if max(counts) > 0 else 3)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        str(val), va="center", fontsize=9,
                        fontweight="bold", color="white")
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.spines[:].set_color("#333344")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if total > 0:
            worst_class = max(per_class, key=per_class.get)
            avg_conf    = sum(
                d["confidence"] for d in output["detections"]
            ) / total
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Total Defects",  total)
            c2.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
            st.info(
                f"Most frequent: "
                f"**{worst_class.replace('_', ' ').title()}** "
                f"({per_class[worst_class]}x)"
            )

    if output["detections"]:
        st.markdown("---")
        st.markdown("#### Detailed Detection Table")
        rows = []
        for i, det in enumerate(output["detections"], 1):
            x1, y1, x2, y2 = det["bbox"]
            rows.append({
                "#"          : i,
                "Class"      : det["class"].replace("_", " ").title(),
                "Confidence" : f"{det['confidence']*100:.1f}%",
                "X1"         : int(x1),
                "Y1"         : int(y1),
                "X2"         : int(x2),
                "Y2"         : int(y2),
                "Width px"   : int(x2 - x1),
                "Height px"  : int(y2 - y1),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "Download Results as CSV",
            data      = csv,
            file_name = "pcb_defect_results.csv",
            mime      = "text/csv"
        )

else:
    st.info(
        "Upload a PCB image above to begin detection.\n\n"
        "Detectable defect types: "
        "Open Circuit, Short Circuit, Mouse Bite, Spur, Copper Spill, Pin Hole"
    )
    st.markdown("---")
    st.markdown("#### How It Works")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Upload**  \nDrop any PCB image in JPG or PNG format")
    c2.markdown("**2. Detect**  \nYOLOv8s scans and localizes each defect with bounding boxes")
    c3.markdown("**3. Analyze**  \nReview annotated image, class breakdown, and exportable results table")
