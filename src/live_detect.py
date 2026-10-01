""""Live webcam object detection using Faster R-CNN or YOLOv3.

Run from the project root:
    python -m src.live_detect --model yolo
    python -m src.live_detect --model rcnn
    python -m src.live_detect --model both

Press Q or ESC to quit.
"""
import argparse
from pathlib import Path

import cv2

from .detectors import FasterRCNNDetector, YOLOv3Detector, annotate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live R-CNN + YOLOv3 object detection")
    parser.add_argument("--model", choices=["rcnn", "yolo", "both"], default="yolo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    rcnn = FasterRCNNDetector(args.threshold) if args.model in {"rcnn", "both"} else None
    yolo = (
        YOLOv3Detector(
            cfg=str(ROOT / "models/yolov3/yolov3.cfg"),
            weights=str(ROOT / "models/yolov3/yolov3.weights"),
            names=str(ROOT / "models/yolov3/coco.names"),
            threshold=args.threshold,
        )
        if args.model in {"yolo", "both"}
        else None
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check webcam permissions or try --camera 1."
        )

    print("Live detection started. Press Q or ESC to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break

        if args.model == "both":
            left = annotate(frame, rcnn.predict(frame))
            right = annotate(frame, yolo.predict(frame))
            h = max(left.shape[0], right.shape[0])
            left = cv2.resize(left, (left.shape[1], h))
            right = cv2.resize(right, (right.shape[1], h))
            out = cv2.hconcat([left, right])
            cv2.putText(
                out, "Faster R-CNN", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            cv2.putText(
                out, "YOLOv3", (right.shape[1] + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
        elif args.model == "rcnn":
            out = annotate(frame, rcnn.predict(frame))
            cv2.putText(
                out, "Faster R-CNN - LIVE", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
        else:
            out = annotate(frame, yolo.predict(frame))
            cv2.putText(
                out, "YOLOv3 - LIVE", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )

        cv2.imshow("Live Object Detection - R-CNN + YOLOv3", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"