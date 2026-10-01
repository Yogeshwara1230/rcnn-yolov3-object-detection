# Object Detection using R-CNN and YOLOv3

A student-friendly project for detecting and naming objects in **images and videos** using two deep-learning detectors:

- **R-CNN family:** Faster R-CNN (the practical R-CNN implementation used here)
- **YOLO:** YOLOv3
- **Dataset reference:** nuScenes metadata supplied with the project

## What the project does

1. Accepts an image or video.
2. Pre-processes the input.
3. Runs Faster R-CNN or YOLOv3.
4. Finds objects such as car, person, bus, truck, bicycle, etc.
5. Draws a bounding box around each object.
6. Displays the object name and confidence score.
7. Saves the annotated image/video.
8. Provides a side-by-side comparison when an image is processed by both models.

> Important: `v1.0-test_meta.tgz` contains nuScenes metadata/annotations and map files, but not the camera image files themselves. Therefore this project is immediately usable with your own images/videos; the nuScenes metadata parser is included for later dataset preparation when the corresponding nuScenes sensor files are available.

## Project architecture

```text
Image / Video
      |
      v
Pre-processing
      |
      +-------------------+
      |                   |
      v                   v
Faster R-CNN           YOLOv3
      |                   |
      +---------+---------+
                v
        Object Detection
                |
       Class + Box + Score
                |
                v
     Annotated Image / Video
                |
                v
       Results / Comparison
```

## 1. Install

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 2. Get YOLOv3 model files

Run:

```bash
python src/download_yolov3.py
```

This downloads the standard YOLOv3 COCO configuration, weights, and class names into `models/yolov3/`.

Faster R-CNN weights are downloaded automatically by torchvision on first use.

## 3. Start the application

```bash
streamlit run src/app.py
```

Then open the local Streamlit address shown in the terminal.

## 4. Command-line image detection

```bash
python src/detect.py --model rcnn --source path/to/image.jpg --output outputs/rcnn.jpg
python src/detect.py --model yolov3 --source path/to/image.jpg --output outputs/yolov3.jpg
```

## 5. Video detection

```bash
python src/detect.py --model yolov3 --source path/to/video.mp4 --output outputs/result.mp4
```

## 6. NuScenes metadata

The supplied archive can be extracted into `data/nuscenes_meta/`. The file `src/nuscenes_metadata.py` reads categories, scenes, samples and annotations and prints a summary.

```bash
python src/nuscenes_metadata.py --root data/nuscenes_meta/v1.0-test
```

To train/evaluate specifically on nuScenes camera images, download the corresponding nuScenes sensor data and connect its image paths to the metadata records. The supplied metadata archive alone does not contain those image files.

## Suggested report title

**Object Detection and Recognition in Images and Videos Using R-CNN and YOLOv3**

## Suggested objectives

- Detect objects in images and videos.
- Identify the name/class of each detected object.
- Localize objects using bounding boxes.
- Compare Faster R-CNN and YOLOv3 detection outputs.
- Study confidence scores and processing speed.
- Prepare the pipeline for nuScenes-based traffic/autonomous-driving analysis.

## Important evaluation note

Precision, recall and mAP require ground-truth labels. Do not report invented values. When a labeled test set is available, use `evaluate.py` and calculate the metrics from the actual predictions and annotations.

## Live Webcam Detection

The project now supports real-time webcam detection using OpenCV.

From the project root:

```bash
python src/live_detect.py --model yolo
```

For Faster R-CNN:

```bash
python src/live_detect.py --model rcnn
```

To show both detectors side-by-side:

```bash
python src/live_detect.py --model both
```

Press **Q** or **Esc** to stop the live detection window. If the default camera does not open, try another index, for example `--camera 1`.

The live output shows the **object name, confidence score, and bounding box** for detected objects such as cars, buses, trucks, people, bicycles, and other supported COCO classes.
