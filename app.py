
import os
import json
import cv2
import numpy as np
import streamlit as st

from PIL import Image
from tensorflow import keras

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MoodTune AI",
    page_icon="😊",
    layout="wide"
)

# ==========================================
# EMOTION INFORMATION
# ==========================================

EMOTIONS = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise"
}

# ==========================================
# LOAD EXISTING TRAINED MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "emotion_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


@st.cache_resource
def load_emotion_model():

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if model.input_shape != (None, 48, 48, 1):
        raise ValueError("Unexpected model input shape.")

    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output and class names mismatch.")

    return model, class_names


try:
    model, class_names = load_emotion_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# FACE DETECTION
# ==========================================

@st.cache_resource
def load_face_detector():

    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")

    return detector


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image_array = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    detector = load_face_detector()

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    # Select largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    face = gray[y:y+h, x:x+w]

    face_resized = cv2.resize(
        face,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    # Same normalization as training
    face_normalized = (
        face_resized.astype("float32") / 255.0
    )

    # Model input: (1, 48, 48, 1)
    model_input = np.expand_dims(
        face_normalized,
        axis=(0, -1)
    )

    return model_input, face_resized


# ==========================================
# USER INTERFACE
# ==========================================

st.title("😊 MoodTune AI")

st.subheader(
    "AI-Based Facial Emotion Recognition"
)

st.write(
    "Upload a face image and let MoodTune AI "
    "analyze its facial expression."
)

st.info(
    "This application uses a CNN model trained "
    "specifically for seven facial expression classes."
)

st.caption(
    "Predictions represent model classifications, "
    "not a definitive measurement of a person's feelings."
)

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        model_input, processed_face = preprocess_image(image)

        if model_input is None:

            st.warning(
                "No face detected. Please upload a clear, "
                "front-facing face image."
            )

        else:

            with col2:
                st.image(
                    processed_face,
                    caption="Processed Face (48 x 48)",
                    clamp=True,
                    use_container_width=True
                )

            # ==================================
            # PREDICTION
            # ==================================

            with st.spinner("Analyzing facial expression..."):

                probabilities = model.predict(
                    model_input,
                    verbose=0
                )[0]

            predicted_index = int(np.argmax(probabilities))

            predicted_emotion = class_names[predicted_index]

            confidence = float(
                probabilities[predicted_index] * 100
            )

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Expression: "
                f"{EMOTIONS.get(predicted_emotion, predicted_emotion.title())}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # ==================================
            # ALL EMOTION PROBABILITIES
            # ==================================

            st.subheader("Emotion Probability Distribution")

            for i, emotion in enumerate(class_names):

                probability = float(probabilities[i])

                st.write(
                    f"**{EMOTIONS.get(emotion, emotion.title())}** "
                    f"- {probability * 100:.2f}%"
                )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

            st.caption(
                "Probabilities are model output scores, "
                "not calibrated measures of actual emotion."
            )

    except Exception as e:

        st.error(
            f"Unable to process this image: {e}"
        )

st.divider()

st.caption(
    "MoodTune AI | Machine Learning Final Project"
)

st.caption(
    "Model: Baseline CNN | Input: 48x48 Grayscale"
)
