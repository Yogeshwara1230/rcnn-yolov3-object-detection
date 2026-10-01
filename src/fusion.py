"""Fuse detections from Faster R-CNN and YOLOv3 into one final result."""

from typing import List, Tuple

import numpy as np

Detection = Tuple[str, float, Tuple[int, int, int, int]]


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def fuse_detections(
    rcnn_detections: List[Detection],
    yolo_detections: List[Detection],
    iou_threshold: float = 0.50,
) -> List[Detection]:
    """Combine detections from both models.

    When both models detect the same class with overlapping boxes, retain one
    detection using the higher confidence and the average box coordinates.
    Unmatched detections from either model are retained.
    """
    candidates = [
        (name, score, box, "rcnn")
        for name, score, box in rcnn_detections
    ] + [
        (name, score, box, "yolo")
        for name, score, box in yolo_detections
    ]

    candidates.sort(key=lambda x: x[1], reverse=True)
    used = np.zeros(len(candidates), dtype=bool)
    final: List[Detection] = []

    for i, (name, score, box, source) in enumerate(candidates):
        if used[i]:
            continue

        used[i] = True
        matched = [(box, score)]

        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue

            other_name, other_score, other_box, _ = candidates[j]

            if other_name == name and _iou(box, other_box) >= iou_threshold:
                used[j] = True
                matched.append((other_box, other_score))

        if len(matched) > 1:
            xs1 = [b[0] for b, _ in matched]
            ys1 = [b[1] for b, _ in matched]
            xs2 = [b[2] for b, _ in matched]
            ys2 = [b[3] for b, _ in matched]
            fused_box = (
                int(round(sum(xs1) / len(xs1))),
                int(round(sum(ys1) / len(ys1))),
                int(round(sum(xs2) / len(xs2))),
                int(round(sum(ys2) / len(ys2))),
            )
            fused_score = float(sum(s for _, s in matched) / len(matched))
        else:
            fused_box = box
            fused_score = float(score)

        final.append((name, fused_score, fused_box))

    final.sort(key=lambda x: x[1], reverse=True)
    return final
