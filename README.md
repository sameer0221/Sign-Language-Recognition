# ✋ Sign Language Recognition

A **real-time hand gesture recognition system** built with **Python**, **OpenCV**, and **TensorFlow/Keras** that detects and classifies sign language hand gestures (digits 0–9). This project trains a convolutional neural network (CNN) using webcam-captured images to recognize hand signs and can be extended for broader gesture recognition tasks. :contentReference[oaicite:0]{index=0}

---

## 🚀 Features

- 📸 **Real-time gesture capture** using webcam
- 🧠 **CNN model training** for accurate classification of hand gestures
- 📊 Simple folder-based dataset setup
- 🛠️ Ready-to-run Python scripts for training and inference
- 📦 Expandable for sign language letters and custom gestures

---

## 📁 Repository Structure

Sign-Language-Recognition/
├── create_gesture_data.py # Capture gesture images via webcam
├── DataFlair_trainCNN.py # Train the CNN model
├── model_for_gesture.py # Load & test the trained model
├── dataset/ # Collected gesture images
├── requirements.txt # Python dependencies
└── README.md

---

## 🧠 How It Works

1. **Capture images** of hand signs for each class (digit) using the webcam.  
2. Store captured images under class-named folders.  
3. Train the CNN model using these images.  
4. Use the trained model to classify live hand gestures.  

---

## 🛠️ Getting Started

### 🧾 Prerequisites

Install Python 3.7+ and required libraries:

pip install -r requirements.txt
Make sure you have:

Python

OpenCV

TensorFlow / Keras

NumPy
📸 Step 1 — Collect Gesture Data

Run the data collection script:

python create_gesture_data.py


A window will open allowing you to capture images for each gesture class. Save enough images per class (50–200 recommended).

🧪 Step 2 — Train the Model

Train your CNN model on the captured dataset:

python DataFlair_trainCNN.py


This will generate a saved model file (e.g., model.h5) that can be used for prediction.

▶️ Step 3 — Recognize Gestures

After training, run the inference script:

python model_for_gesture.py


This launches the webcam and displays detected gestures in real-time.

📈 Results

Once trained, the model can recognize numeric sign gestures (0–9) with reasonable accuracy. You can expand this system to include:

Alphabet gestures (A–Z)

Sentence formation

Integration with Speech/Text output

🧩 Contributions

Contributions, suggestions and improvements are very welcome!
To contribute:
Fork the repository ⭐

Create a new branch (git checkout -b feature/xyz)

Commit your changes

Open a Pull Request
