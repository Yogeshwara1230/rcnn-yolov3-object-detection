"""Streamlit app for offline image/video object detection with R-CNN + YOLOv3."""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st

from .detectors import FasterRCNNDetector, YOLOv3Detector, annotate


st.set_page_config(page_title="R-CNN + YOLOv3 Object Detection", layout="wide")
st.title("Object Detection and Recognition using R-CNN + YOLOv3")
st.write(
    "Upload an image or video. The system runs Faster R-CNN and YOLOv3 and shows "
    "the detected object names, bounding boxes, and confidence scores."
)

source = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"],
)
threshold = st.sidebar.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05)
run_both = st.sidebar.checkbox("Run both models", value=True)

@st.cache_resource
def load_rcnn(threshold_value: float):
    return FasterRCNNDetector(threshold_value)

@st.cache_resource
def load_yolo(threshold_value: float):
    return YOLOv3Detector(threshold=threshold_value)

if source is not None:
    suffix = Path(source.name).suffix.lower()
    data = source.read()

    if suffix in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not decode the uploaded image.")
            st.stop()

        if run_both:
            with st.spinner("Running Faster R-CNN and YOLOv3..."):
                rcnn_res = load_rcnn(threshold).predict(frame)
                yolo_res = load_yolo(threshold).predict(frame)

            left = annotate(frame, rcnn_res)
            right = annotate(frame, yolo_res)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Faster R-CNN")
                st.image(
                    cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                st.write(f"Objects detected: {len(rcnn_res)}")

            with col2:
                st.subheader("YOLOv3")
                st.image(
                    cv2.cvtColor(right, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                st.write(f"Objects detected: {len(yolo_res)}")
        else:
            st.info("Enable 'Run both models' to process the same image with both detectors.")

    else:
        st.warning(
            "Video upload is supported. The current app processes the video offline "
            "and displays the annotated result after processing."
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(data)
            inpath = f.name

        detector_rcnn = load_rcnn(threshold)
        detector_yolo = load_yolo(threshold)

        cap = cv2.VideoCapture(inpath)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        outpath = inpath + "_out.mp4"
        writer = cv2.VideoWriter(
            outpath,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        progress = st.progress(0.0)
        status = st.empty()
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        processed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rcnn_res = detector_rcnn.predict(frame)
            yolo_res = detector_yolo.predict(frame)

            # For video output, show both model results side-by-side when possible.
            left = annotate(frame, rcnn_res)
            right = annotate(frame, yolo_res)
            combined = cv2.hconcat([left, right])
            writer.write(combined)

            processed += 1
            if total:
                progress.progress(min(processed / total, 1.0))
            status.write(f"Processed {processed} frames")

        cap.release()
        writer.release()

        st.success("Video processing completed.")
        st.video(outpath)
