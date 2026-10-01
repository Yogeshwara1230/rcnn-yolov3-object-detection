# Project Plan

## Problem statement
Given an image or video, automatically identify visible road/scene objects and show their names and locations using bounding boxes.

## Models
1. Faster R-CNN — two-stage region-based detector representing the R-CNN family.
2. YOLOv3 — one-stage detector.

## Input
- Image: JPG/PNG
- Video: MP4/AVI/MOV/MKV
- Future dataset input: nuScenes camera images and labels

## Output
- Object class/name
- Confidence score
- Bounding box coordinates
- Annotated image/video

## Evaluation
For a labeled test set:
- Precision
- Recall
- mAP@0.5 / mAP@0.5:0.95
- Inference time
- FPS

Do not fill these metrics with sample values; calculate them from actual ground-truth annotations and predictions.
