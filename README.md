# 🧠 AI_Multivision_App – Streamlit-based AI Vision Toolkit
A lightweight AI computer vision app built with **Streamlit** that combines multiple real-world vision tasks: **YOLOv8 object detection**, **license plate OCR**, **emotion detection**, and **face blurring** — all wrapped in a user-friendly web interface.
![demo](./images/demo2.png)

---

## 📌 Table of Contents

1. [✨ Project Overview](#-project-overview)  
2. [🚀 Features](#-features)  
3. [🗂️ Project Structure](#-project-structure)
4. [🧰 Tech Stack](#-tech-stack)
5. [⚙️ Installation](#-installation)
6. [✅ Feature Details](#-feature-details)
7. [🧠 How It Works)](#-how-it-works)
8. [🧪 Known Issuess](#-known-issues)
9. [📈 Future Enhancements](#-future-enhancements)  
10. [📄 License](#-license)
11. [🤝 Contributing](#-contributing)
12. [📬 Contact](#-contact)

---

## ✨ Project Overview

### 1. This app is designed to:
- Leverage **YOLOv8** for detecting general objects and faces.
- Perform **OCR** on **license plates** of vehicles with contour-based heuristics.
- Recognize **facial emotions** using a fine-tuned **ResNet18 CNN**.
- Provide **face anonymization** via Gaussian blur.

### 2. Accessible via Streamlit for demo or educational purposes.

---

## 🚀 Features

| Feature Name                         | Description                                                | Technique/Model Used                                  |
|--------------------------------------|------------------------------------------------------------|--------------------------------------------------------
| **YOLOv8 Detection**              | Real-time object detection for general scenes                         | ``YOLOv8n.pt``                 |   
| **License Plate OCR**   | Detect & extract text from Vietnamese plates                           | YOLOv8 + EasyOCR + Contour Heuristics                      |
| **Emotion Detection**                    | Predict facial emotion from detected faces                               | ResNet18 (trained on FER2013)           |
| **Face Blurring**                | 	Auto-detect and anonymize faces for privacy              | YOLOv8-Face + Gaussian Blur        |


---
## 🗂️ Project Structure
```
├── ai_multivision_app.py           # Main Streamlit app logic
├── train_emotion_model.py          # Script to train ResNet emotion model
├── emotion_resnet18.pth            # Trained model weights for emotion detection
├── yolov8n.pt                      # YOLOv8 base model
├── yolov8n-face.pt                 # YOLOv8 face detector
├── license_plate_detector.pt       # (Optional) Custom license plate YOLOv8 model
├── static/
├── requirements.txt                # Python dependencies
└── README.md
└── LICENSE

```
---

## 🧰 Tech Stack

| Purpose                  | Libraries Used                                        |
|--------------------------|-------------------------------------------------------|
| **Detection**        | ``	ultralytics`` (YOLOv8)                            |
| **OCR**       | ``easyocr``, ``OpenCV``, ``contours``                                 |
| **Emotion Detection**              | ``PyTorch``, `FER2013`, `ResNet18`  |
| **Face Blurring**       | ``cv2``, ``YOLO`` ``haar cascade``                                    |
| **Web Interface**    | ``Streamlit``, ``Pillow``, ``matplotlib``  |



---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/paht2005/ai-voice-assistant-suite.git
cd ai-voice-assistant-suite

# Install dependencies
pip install -r requirements-ai.txt

# Run Flask web app
python flask_app.py


```
Then open your browser: http://127.0.0.1:5000

---

## ✅ Feature Details

### 1. Voice Transcription
- Uses Whisper model to transcribe audio files or mic input.
- Auto language detection & punctuation recovery.
### 2. TTS Answering
- Enter any text → generate voice using Coqui TTS model.
- Output saved as ``static/tts_output.wav``.
### 3. Voice Cloning
- Upload a 3–5 sec voice sample (.wav).
- Type any text → generates response in that speaker's voice.
### 4. Emotion Detection (CNN)
- Trained using RAVDESS dataset.
- Input ``.wav`` → predicts 1 of 8 emotions.
- Model: ``CNN + MFCC`` → ``emotion_cnn.pth``
### 5. Document Q&A
- Upload voice question (.wav)
- Uses Whisper to transcribe → SentenceTransformer + ChromaDB to retrieve doc context → Falcon or LLM to answer.
### 6. Podcast Summarization
- Upload long ``.wav`` podcast → splits into chunks → summarizes using BART-based model.
- Summary returned as paragraph.

---
## 🛠 How It Works (Behind the Scenes)
Each feature operates through a dedicated **audio or NLP processing pipeline**:

### 1. Voice Transcription
- **Input:** ``.wav`` file recorded from the user
- **Process:**
  - The **Whisper** model converts the audio waveform into a **log-Mel spectrogram**.
  - A multilingual decoder processes the spectrogram and generates the corresponding **text transcription**.
- **Output:** Clean, normalized text

### 2. Text-to-Speech (TTS) Answering
- **Input:** Text string generated or typed by the user
- **Process:**
  - A **Tacotron2** model (or similar) converts text to a **mel-spectrogram**.
  - A **vocoder** (e.g., HiFi-GAN, WaveGlow) synthesizes audio from the spectrogram.
  - Optionally, the voice output is adjusted using a **cloned speaker embedding**.
- **Output:** ``tts_output.wav`` file saved in ``/static/`` directory

### 3. Voice Cloning
- **Input:** A reference voice ``.wav`` file + target text
- **Process:**
  - Extract a **speaker embedding** from the input voice.
  - Use a multi-speaker TTS model to synthesize speech that matches the **tone and identity** of the reference speaker.
- **Output:** Synthetic speech in the cloned voice

### 4. Emotion Detection (CNN)
- **Input:** A short audio segment 
- **Process:**
  - Extract **MFCC (Mel-Frequency Cepstral Coefficients)** features.
  - Feed the MFCC vector into a **2-layer CNN** or classifier.
  - Predict one of several emotion classes: e.g., ``angry``, ``happy``, ``sad``, etc.
- **Output:** Detected emotion label (among 8 predefined classes)

### 5. Document Q&A (RAG)
- **Input:** Spoken question from the user
- **Process:**
  - **Whisper** transcribes the spoken query into text.
  - The query is embedded using **SentenceTransformer**.
  - A **vector search** is performed using **ChromaDB** to find relevant documents.
  - A **language model (LLM)** generates the final answer using the retrieved context.
- **Output:** Answer in natural language (text)

### 6. Podcast Summarization
- **Input:** Long-form ``.wav`` file (e.g., podcast, recorded lecture)
- **Process:**
  - **Whisper** transcribes the full audio into text.
  - The transcript is **chunked** into manageable segments.
  - Each segment is summarized using a model like **BART** or **DistilBART**.
- **Output:** Final summary as paragraph or structured bullet points.
--- 
## 🧪 Known Issues

| Issue                         | Cause                       | Solution                                                                |
|-------------------------------|-----------------------------|-------------------------------------------------------------------------
| **❗ Whisper FP16 Warningn**  | No GPU                      | Ignore or use GPU for speed                                            |   
| **❌ ``punkt`` not found**    | NLTK missing tokenizer      | Run ``nltk.download('punkt')``                                         |
| **❌ Audio shape mismatchg**  | CNN flatten mismatch        | Use dynamic flatten in CNN                                             |
| **❌ ffmpeg not found**       | Whisper depends on it       | [Install ffmpeg](https://ffmpeg.org/download.html) & add to PATH       |

--- 
## 🧭 Future Work
-  Add real-time streaming voice interface
- WebSocket for fast speech interaction
- Export as REST API (for mobile use)
- Integrate multi-lingual support (Vietnamese, etc.)
- User auth system for personalized interaction
---
## 📄 License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.


---
## 🤝 Contributing
I welcome contributions to improve this project!
Feel free to fork, pull request, or open issues. Ideas welcome!


--- 
## 📬 Contact
- Contact for work: **Nguyễn Công Phát** – congphatnguyen.work@gmail.com
- [Github](https://github.com/paht2005) 
