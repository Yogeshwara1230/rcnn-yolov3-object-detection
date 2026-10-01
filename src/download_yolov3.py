from pathlib import Path
import requests

BASE=Path('models/yolov3'); BASE.mkdir(parents=True, exist_ok=True)
FILES={
 'yolov3.cfg':'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
 'yolov3.weights':'https://pjreddie.com/media/files/yolov3.weights',
 'coco.names':'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
}
for name,url in FILES.items():
    path=BASE/name
    if path.exists():
        print('exists:',path); continue
    print('downloading:',name)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path,'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
    print('saved:',path)
print('YOLOv3 files ready.')
