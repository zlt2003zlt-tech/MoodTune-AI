
# 😊 MoodTune AI

## AI-Based Facial Emotion Recognition

MoodTune AI is an AI-powered facial expression recognition application developed using Python, CNN, OpenCV, and MediaPipe.

Users can upload a facial image and receive a predicted facial expression from seven categories.

## Features

- Facial image upload
- Face detection using MediaPipe
- Face cropping and preprocessing
- Grayscale image conversion (48x48)
- CNN-based facial expression classification
- Emotion prediction probability visualization

## Emotion Categories

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

## Technologies Used

- Python 3.13.2
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- OpenCV
- MediaPipe
- Streamlit
- NumPy
- Pillow

## Model Information

- Model: Baseline CNN
- Dataset: FER2013
- Input Image Size: 48x48
- Color Mode: Grayscale
- Output: 7 emotion classes

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
MoodTune-AI/
├── app.py
├── emotion_model.keras
├── class_names.json
├── blaze_face_short_range.tflite
├── requirements.txt
└── README.md
```

## Disclaimer

The predicted expressions are model classifications and do not represent a definitive measurement of a person's actual emotional state.

## Developer

MoodTune AI - Machine Learning Project
