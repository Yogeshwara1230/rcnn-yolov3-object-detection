import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import tempfile
import cv2
import streamlit as st
from detectors import FasterRCNNDetector, YOLOv3Detector, annotate

st.set_page_config(page_title='R-CNN + YOLOv3 Object Detection', layout='wide')
st.title('Object Detection and Recognition using R-CNN and YOLOv3')
st.write('Use an image, video, or your live webcam. The system detects objects, gives their names and confidence scores, and draws bounding boxes.')
model=st.sidebar.selectbox('Detection model',['R-CNN (Faster R-CNN)','YOLOv3','Compare both (image only)'])
source_mode=st.sidebar.radio('Input source',['Upload file','Live webcam'])
threshold=st.sidebar.slider('Confidence threshold',0.10,0.95,0.50,0.05)
source=st.file_uploader('Upload image or video',type=['jpg','jpeg','png','mp4','avi','mov','mkv'])
@st.cache_resource
def load_rcnn(): return FasterRCNNDetector()
@st.cache_resource
def load_yolo(): return YOLOv3Detector()
if source_mode == 'Live webcam':
    st.subheader('Live webcam detection')
    st.info('For true continuous live detection, run: python src/live_detect.py --model yolo (or --model rcnn / --model both) from the project folder. The browser-based Streamlit app cannot continuously stream webcam frames with OpenCV alone.')
    st.code('python src/live_detect.py --model yolo', language='bash')

if source_mode == 'Upload file' and source:
    suffix=Path(source.name).suffix.lower()
    data=source.read()
    if suffix in {'.jpg','.jpeg','.png'}:
        import numpy as np
        frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if model.startswith('R-CNN') or model.startswith('Compare'):
            d=load_rcnn(); res=annotate(frame,d.predict(frame)); st.subheader('R-CNN result'); st.image(cv2.cvtColor(res,cv2.COLOR_BGR2RGB),use_container_width=True)
        if model.startswith('YOLO') or model.startswith('Compare'):
            d=load_yolo(); res=annotate(frame,d.predict(frame)); st.subheader('YOLOv3 result'); st.image(cv2.cvtColor(res,cv2.COLOR_BGR2RGB),use_container_width=True)
    else:
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f: f.write(data); inpath=f.name
        outpath=inpath+'_out.mp4'; cap=cv2.VideoCapture(inpath)
        fps=cap.get(cv2.CAP_PROP_FPS) or 25; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer=cv2.VideoWriter(outpath,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)); detector=load_rcnn() if model.startswith('R-CNN') else load_yolo(); count=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            writer.write(annotate(frame,detector.predict(frame))); count+=1
            if count % 10 == 0: st.write(f'Processed {count} frames...')
        cap.release(); writer.release(); st.video(outpath)
