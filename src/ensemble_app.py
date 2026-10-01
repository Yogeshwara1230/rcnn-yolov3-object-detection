"""Single-output Streamlit application using Faster R-CNN + YOLOv3."""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detectors import FasterRCNNDetector, YOLOv3Detector, annotate
from fusion import fuse_detections


st.set_page_config(page_title="R-CNN + YOLOv3 Object Detector", layout="wide")
st.title("R-CNN + YOLOv3 Object Detection")
st.write("Upload one image. Both models process the same image and their detections are fused into one final result.")

threshold = st.sidebar.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)
iou_threshold = st.sidebar.slider("Fusion IoU threshold", 0.10, 0.90, 0.50, 0.05)

@st.cache_resource
def load_models(threshold_value: float):
    rcnn = FasterRCNNDetector(threshold_value)
    yolo = YOLOv3Detector(
        cfg=str(ROOT / "models/yolov3/yolov3.cfg"),
        weights=str(ROOT / "models/yolov3/yolov3.weights"),
        names=str(ROOT / "models/yolov3/coco.names"),
        threshold=threshold_value,
    )
    return rcnn, yolo


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    data = uploaded.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    with st.spinner("Running Faster R-CNN + YOLOv3 and fusing detections..."):
        rcnn, yolo = load_models(threshold)
        rcnn_results = rcnn.predict(frame)
        yolo_results = yolo.predict(frame)
        final_results = fuse_detections(
            rcnn_results, yolo_results, iou_threshold=iou_threshold
        )

    result = annotate(frame, final_results)

    st.subheader("Final Fused Detection")
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.write(f"Final objects detected: **{len(final_results)}**")

    st.subheader("Detected Objects")
    if final_results:
        for name, score, box in final_results:
            st.write(f"**{name}** — {score:.2%} — box={box}")
    else:
        st.info("No objects met the confidence threshold.")

    with st.expander("Model details"):
        st.write(f"Faster R-CNN detections: {len(rcnn_results)}")
        st.write(f"YOLOv3 detections: {len(yolo_results)}")
        st.write(f"Fused detections: {len(final_results)}")
else:
    st.info("Upload a JPG, JPEG, or PNG image to start detection.")
