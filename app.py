import io
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Object Detection Studio", page_icon="🔎", layout="wide")
st.title("🔎 Object Detection Studio")
st.caption("Upload an image, choose a YOLO model, and inspect detected objects.")

MODEL_OPTIONS = {
    "YOLOv8 Nano (fast)": "yolov8n.pt",
    "YOLOv8 Small (balanced)": "yolov8s.pt",
    "YOLOv8 Medium (heavier)": "yolov8m.pt",
}

@st.cache_resource(show_spinner="Loading detection model…")
def load_model(weights_name):
    return YOLO(weights_name)

with st.sidebar:
    st.header("Detection settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_threshold = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
    st.info("The pretrained YOLO model detects common objects from the COCO categories.")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "webp"],
    help="Choose a photo from your phone or computer."
)

if uploaded_file is None:
    st.markdown("""
    ### How to use
    1. Upload a photo.
    2. Choose a YOLO model and adjust the confidence threshold.
    3. Run detection to view bounding boxes, confidence scores and inference time.
    """)
    st.stop()

try:
    original_image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

st.image(original_image, caption="Uploaded image", use_container_width=True)

if st.button("Run object detection", type="primary", use_container_width=True):
    try:
        model = load_model(MODEL_OPTIONS[model_label])
        image_array = np.asarray(original_image)

        start_time = time.perf_counter()
        predictions = model.predict(
            source=image_array, conf=confidence, iou=iou_threshold, verbose=False
        )
        elapsed = time.perf_counter() - start_time

        result = predictions[0]
        annotated_rgb = result.plot()[:, :, ::-1]

        rows = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                coords = box.xyxy[0].tolist()
                rows.append({
                    "Object": result.names[class_id],
                    "Confidence": round(float(box.conf.item()), 4),
                    "X1": round(coords[0], 1),
                    "Y1": round(coords[1], 1),
                    "X2": round(coords[2], 1),
                    "Y2": round(coords[3], 1),
                })

        detections = pd.DataFrame(
            rows, columns=["Object", "Confidence", "X1", "Y1", "X2", "Y2"]
        )

        st.subheader("Detection result")
        left, right = st.columns(2)
        with left:
            st.image(annotated_rgb, caption="Detected objects", use_container_width=True)
        with right:
            st.metric("Objects detected", len(detections))
            st.metric("Inference time", f"{elapsed:.3f} seconds")
            st.dataframe(detections, use_container_width=True, hide_index=True)
            st.download_button(
                "Download detection report (CSV)",
                data=detections.to_csv(index=False).encode("utf-8"),
                file_name="detection_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        image_buffer = io.BytesIO()
        Image.fromarray(annotated_rgb).save(image_buffer, format="PNG")
        st.download_button(
            "Download detected image",
            data=image_buffer.getvalue(),
            file_name="detected_output.png",
            mime="image/png",
            use_container_width=True,
        )

        if detections.empty:
            st.warning("No objects passed this confidence threshold. Try lowering it.")
    except Exception as exc:
        st.error("Detection could not be completed.")
        st.exception(exc)

st.divider()
st.caption(
    "Educational demonstration. Inference time is measured on the Streamlit server; "
    "it is not an accuracy or mAP score."
)
