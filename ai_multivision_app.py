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
import matplotlib.pyplot as plt

# App configuration
st.set_page_config(page_title="AI Vision App", layout="wide")
st.title("🔍 AI Vision Processing App")
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Load pre-trained models
yolo_model = YOLO("yolov8n.pt")
yolo_face_model = YOLO("yolov8n-face.pt")  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ocr_reader = easyocr.Reader(['en'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simplified CNN architecture for emotion classification
class EmotionCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)  
        self.conv2 = torch.nn.Conv2d(32, 64, 3) 
        self.pool = torch.nn.MaxPool2d(2)       
        self.fc1 = torch.nn.Linear(64 * 10 * 10, 128) 
        self.fc2 = torch.nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x))) 
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 10 * 10)  # FIXED shape
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)
    
# Load pretrained ResNet emotion model
class ResNetEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 7)
    def forward(self, x):
        return self.model(x)
    
# Initialize dummy emotion model (to be replaced with a trained model)

emotion_model = ResNetEmotion().to(device)
emotion_model.load_state_dict(torch.load("emotion_resnet18.pth", map_location=device))
emotion_model.eval()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Utility: Load and convert uploaded image
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

# Blur faces using Haar cascade
def detect_faces_and_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi, (99, 99), 30)
        image[y:y+h, x:x+w] = blur
    return image

# Perform object detection using YOLOv8
def run_yolo(image):
    results = yolo_model(image)
    img_with_boxes = results[0].plot()
    return img_with_boxes
#
#  Detect and recognize license plates using contour heuristics + OCR
def detect_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h)
        if 2 < ratio < 6 and 100 < w < 500 and 30 < h < 150:
            plate_candidates.append((x, y, w, h))

    result_img = image.copy()
    plate_texts = []

    for (x, y, w, h) in plate_candidates:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plate_img = gray[y:y + h, x:x + w]
        result = ocr_reader.readtext(plate_img)

        for (bbox, text, confidence) in result:
            if confidence > 0.5 and len(text.strip()) >= 3:
                plate_texts.append((x, y, w, h, text))
                cv2.putText(result_img, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return plate_texts

# Emotion detection from facial regions using CNN
def detect_emotion(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_face_model(rgb)
    faces = results[0].boxes.xyxy.cpu().numpy().astype(int)  # [x1, y1, x2, y2]
    
    output_img = image.copy()
    emotions = []

    for (x1, y1, x2, y2) in faces:
        w = x2 - x1
        h = y2 - y1

        if w < 40 or h < 40:
            continue  # Skip

        face = rgb[y1:y2, x1:x2]
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        face = face / 255.0

        face_tensor = torch.tensor(face).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = emotion_model(face_tensor)
            pred = torch.argmax(output, dim=1).item()
            emotion = emotion_labels[pred]
            emotions.append(((x1, y1, w, h), emotion))

            # Result
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return output_img, emotions
# Streamlit UI layout with task-specific tabs
st.title("🧠 MultiVision AI Streamlit App")
tabs = st.tabs(["Upload", "YOLOv8 Detection", "License Plate OCR", "Emotion Detection", "Blur Faces"])

with tabs[0]:
    uploaded = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        image_np = load_image(uploaded)
        st.image(image_np, caption="Original Image", use_container_width=True)
        st.session_state['image'] = image_np

with tabs[1]:
    if 'image' in st.session_state:
        yolo_result = run_yolo(st.session_state['image'].copy())
        st.image(yolo_result, caption="YOLOv8 Object Detection", use_container_width=True)

with tabs[2]:
    if 'image' in st.session_state:
        plates = detect_license_plate(st.session_state['image'].copy())
        st.image(st.session_state['image'], caption="Detected Plates", use_container_width=True)
        for (x, y, w, h, text) in plates:
            st.write(f"🔤 Text at ({x},{y},{w},{h}): `{text}`")

with tabs[3]:
    if 'image' in st.session_state:
        result_img, emotion_results = detect_emotion(st.session_state['image'].copy())
        st.image(result_img, caption="Emotion Detection Result", use_container_width=True)
        for (box, emotion) in emotion_results:
            st.write(f"🙂 Face at {box}: Emotion → **{emotion}**")

with tabs[4]:
    if 'image' in st.session_state:
        blurred = detect_faces_and_blur(st.session_state['image'].copy())
        st.image(blurred, caption="Faces Blurred for Privacy", use_container_width=True)
