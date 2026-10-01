from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from .classes import COCO_CLASSES

class FasterRCNNDetector:
    def __init__(self, threshold=0.50):
        self.threshold = threshold
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).eval()
        self.preprocess = weights.transforms()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.preprocess(torch.from_numpy(rgb).permute(2,0,1)).to(self.device)
        out = self.model([image])[0]
        results = []
        for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
            s = float(score)
            if s < self.threshold:
                continue
            label_i = int(label)
            if 1 <= label_i <= len(COCO_CLASSES):
                name = COCO_CLASSES[label_i - 1]
            else:
                name = str(label_i)
            x1,y1,x2,y2 = [int(v) for v in box.cpu().tolist()]
            results.append((name, s, (x1,y1,x2,y2)))
        return results

class YOLOv3Detector:
    def __init__(self, cfg='models/yolov3/yolov3.cfg', weights='models/yolov3/yolov3.weights', names='models/yolov3/coco.names', threshold=0.50, nms=0.40):
        self.threshold = threshold
        self.nms = nms
        for p in (cfg, weights, names):
            if not Path(p).exists():
                raise FileNotFoundError(f'Missing YOLOv3 file: {p}. Run python src/download_yolov3.py')
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.names = [x.strip() for x in Path(names).read_text().splitlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def predict(self, frame_bgr):
        h,w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, (416,416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        boxes=[]; scores=[]; class_ids=[]
        for output in outputs:
            for det in output:
                scores_all = det[5:]
                class_id = int(np.argmax(scores_all))
                score = float(scores_all[class_id])
                if score < self.threshold:
                    continue
                cx,cy,bw,bh = det[:4] * np.array([w,h,w,h])
                x=int(cx-bw/2); y=int(cy-bh/2)
                boxes.append([x,y,int(bw),int(bh)])
                scores.append(score); class_ids.append(class_id)
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.nms)
        results=[]
        if len(idxs):
            for i in np.array(idxs).flatten():
                x,y,bw,bh=boxes[i]
                name=self.names[class_ids[i]] if class_ids[i] < len(self.names) else str(class_ids[i])
                results.append((name, float(scores[i]), (x,y,x+bw,y+bh)))
        return results

def annotate(frame, detections):
    out=frame.copy()
    for name,score,(x1,y1,x2,y2) in detections:
        x1=max(0,x1); y1=max(0,y1); x2=min(out.shape[1]-1,x2); y2=min(out.shape[0]-1,y2)
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        text=f'{name} {score:.2f}'
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
        cv2.rectangle(out,(x1,y1-max(24,th+8)),(x1+tw+6,y1),(0,255,0),-1)
        cv2.putText(out,text,(x1+3,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA)
    return out
