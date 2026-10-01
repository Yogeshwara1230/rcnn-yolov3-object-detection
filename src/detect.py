import argparse, time
from pathlib import Path
import cv2
from detectors import FasterRCNNDetector, YOLOv3Detector, annotate

def get_detector(name, threshold):
    return FasterRCNNDetector(threshold) if name=='rcnn' else YOLOv3Detector(threshold=threshold)

def process_image(detector, source, output):
    frame=cv2.imread(source)
    if frame is None: raise ValueError(f'Cannot read image: {source}')
    t=time.perf_counter(); det=detector.predict(frame); dt=time.perf_counter()-t
    out=annotate(frame,det); cv2.imwrite(output,out)
    print(f'detections={len(det)}, inference={dt:.3f}s, output={output}')
    for x in det: print(x[0], f'{x[1]:.3f}', x[2])

def process_video(detector, source, output):
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise ValueError(f'Cannot open video: {source}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    n=0; total=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        det=detector.predict(frame); total+=len(det); n+=1
        writer.write(annotate(frame,det))
    cap.release(); writer.release()
    print(f'frames={n}, detections={total}, output={output}')

ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['rcnn','yolov3'],required=True); ap.add_argument('--source',required=True); ap.add_argument('--output',required=True); ap.add_argument('--threshold',type=float,default=.5)
a=ap.parse_args(); Path(a.output).parent.mkdir(parents=True,exist_ok=True)
d=get_detector(a.model,a.threshold)
ext=Path(a.source).suffix.lower()
if ext in {'.jpg','.jpeg','.png','.bmp','.webp'}: process_image(d,a.source,a.output)
else: process_video(d,a.source,a.output)
