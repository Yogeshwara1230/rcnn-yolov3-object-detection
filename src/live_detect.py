"""Live webcam object detection using Faster R-CNN + YOLOv3 fusion.

Run from the project root:
    python -m src.live_detect

Press Q or ESC to quit.
"""

import argparse
from pathlib import Path

import cv2

from .detectors import FasterRCNNDetector, YOLOv3Detector, annotate
from .fusion import fuse_detections


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live fused R-CNN + YOLOv3 object detection"
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.50, help="Fusion IoU threshold")
    args = parser.parse_args()

    print("Loading Faster R-CNN...")
    rcnn = FasterRCNNDetector(args.threshold)

    print("Loading YOLOv3...")
    yolo = YOLOv3Detector(
        cfg=str(ROOT / "models/yolov3/yolov3.cfg"),
        weights=str(ROOT / "models/yolov3/yolov3.weights"),
        names=str(ROOT / "models/yolov3/coco.names"),
        threshold=args.threshold,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check webcam permissions or try --camera 1."
        )

    print("Live fused detection started. Press Q or ESC to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break

        rcnn_results = rcnn.predict(frame)
        yolo_results = yolo.predict(frame)

        final_results = fuse_detections(
            rcnn_results,
            yolo_results,
            iou_threshold=args.iou,
        )

        out = annotate(frame, final_results)
        cv2.putText(
            out,
            "R-CNN + YOLOv3 FUSION - LIVE",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"Objects: {len(final_results)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Live Object Detection - R-CNN + YOLOv3", out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
