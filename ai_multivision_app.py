# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os

# App Config 
st.set_page_config(page_title="AI Vision App", layout="wide")
st.title(" AI Vision Processing Application")

# Load custom style if available
if os.path.exists("static/style.css"):
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Model Loading 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO models
try:
    license_plate_model = YOLO("license_plate_detector.pt")
    yolo_model = YOLO("yolov8n.pt")
    yolo_face_model = YOLO("yolov8n-face.pt")
except Exception as e:
    st.error(f"Error loading YOLO models: {e}")

# OCR reader
ocr_reader = easyocr.Reader(['en'])

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion Model Definition
class EmotionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)
    
    def forward(self, x):
        return self.model(x)

# Load pretrained emotion model
emotion_model = EmotionResNet().to(device)
try:
    emotion_model.load_state_dict(torch.load("emotion_resnet18.pth", map_location=device))
    emotion_model.eval()
except Exception as e:
    st.error(f"Error loading emotion model: {e}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Utility Functions 
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def blur_faces(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_face_model(rgb, conf=0.3)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for (x1, y1, x2, y2) in boxes:
        roi = image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred

    return image

def run_object_detection(image):
    results = yolo_model(image)
    return results[0].plot()

def detect_license_plate(image):
    image_up = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray_up = cv2.cvtColor(image_up, cv2.COLOR_BGR2GRAY)

    results = yolo_model(image_up, conf=0.25, iou=0.3)
    vehicle_boxes = [box for box, cls in zip(results[0].boxes.xyxy.cpu().numpy().astype(int), results[0].boxes.cls.cpu().numpy()) if int(cls) in [2, 3, 5, 7]]

    plate_texts = []
    output_img = image_up.copy()

    for (x1, y1, x2, y2) in vehicle_boxes:
        car_crop = image_up[y1:y2, x1:x2]
        gray_car = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_car, 100, 200)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 2 < w/h < 6 and 80 < w < 400 and 25 < h < 150:
                plate_roi = gray_car[y:y+h, x:x+w]
                result = ocr_reader.readtext(plate_roi)
                for (_, text, confidence) in result:
                    if confidence > 0.5 and len(text.strip()) >= 4:
                        cv2.rectangle(output_img, (x1+x, y1+y), (x1+x+w, y1+y+h), (0, 255, 0), 2)
                        cv2.putText(output_img, text, (x1+x, y1+y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        plate_texts.append(((x1+x, y1+y, w, h), text))

    return output_img, plate_texts

def detect_emotions(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_img = image.copy()
    
    faces_yolo = yolo_face_model(rgb, conf=0.3)[0].boxes.xyxy.cpu().numpy().astype(int)
    faces_haar = face_cascade.detectMultiScale(gray, 1.1, 5)

    faces = set()
    for (x1, y1, x2, y2) in faces_yolo:
        faces.add((x1, y1, x2, y2))

    for (x, y, w, h) in faces_haar:
        box2 = (x, y, x + w, y + h)
        if not any(iou(box2, b) > 0.5 for b in faces):
            faces.add(box2)

    results = []

    for (x1, y1, x2, y2) in faces:
        if (x2 - x1 < 30) or (y2 - y1 < 30):
            continue

        face_crop = rgb[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) / 255.0
        tensor = torch.tensor(face_gray).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            output = emotion_model(tensor)
            prediction = torch.argmax(output, dim=1).item()
            emotion = emotion_labels[prediction]
            results.append(((x1, y1, x2-x1, y2-y1), emotion))
            
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return output_img, results

def iou(box1, box2):
    xa, ya, xb, yb = box1
    xc, yc, xd, yd = box2
    xi1, yi1 = max(xa, xc), max(ya, yc)
    xi2, yi2 = min(xb, xd), min(yb, yd)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (xb - xa) * (yb - ya)
    box2_area = (xd - xc) * (yd - yc)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Streamlit UI 
st.title("MultiVision AI Streamlit App")
tabs = st.tabs(["Upload", "YOLO Detection", "License Plate", "Emotion Detection", "Blur Faces"])

with tabs[0]:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image_np = load_image(uploaded)
        st.image(image_np, caption="Original Image", use_container_width=True)
        st.session_state['image'] = image_np

with tabs[1]:
    if 'image' in st.session_state:
        result_img = run_object_detection(st.session_state['image'].copy())
        st.image(result_img, caption="YOLOv8 Object Detection", use_container_width=True)

with tabs[2]:
    if 'image' in st.session_state:
        output_img, plates = detect_license_plate(st.session_state['image'].copy())
        st.image(output_img, caption="Detected License Plates", use_container_width=True)
        for (x, y, w, h), text in plates:
            st.write(f"Detected Plate: `{text}` at position ({x},{y},{w},{h})")

with tabs[3]:
    if 'image' in st.session_state:
        result_img, emotions = detect_emotions(st.session_state['image'].copy())
        st.image(result_img, caption="Emotion Detection Result", use_container_width=True)
        for (box, emotion) in emotions:
            st.write(f"Face at {box}: Emotion → **{emotion}**")

with tabs[4]:
    if 'image' in st.session_state:
        blurred_img = blur_faces(st.session_state['image'].copy())
        st.image(blurred_img, caption="Faces Blurred for Privacy", use_container_width=True)

