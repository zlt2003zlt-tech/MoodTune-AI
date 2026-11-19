
import os
import json
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

from PIL import Image, ImageOps
from tensorflow import keras


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MoodTune AI",
    page_icon="😊",
    layout="wide"
)


# ==========================================
# 2. EMOTION INFORMATION
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
# 3. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "emotion_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "blaze_face_short_range.tflite"
)


# ==========================================
# 4. LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_emotion_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"CNN model not found: {MODEL_PATH}"
        )

    if not os.path.exists(CLASS_PATH):
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_PATH}"
        )

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(
        CLASS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_names = json.load(f)

    if model.input_shape != (None, 48, 48, 1):
        raise ValueError(
            f"Unexpected model input: {model.input_shape}"
        )

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output and class names mismatch."
        )

    return model, class_names


try:

    model, class_names = load_emotion_model()

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# 5. LOAD MEDIAPIPE FACE DETECTOR
# ==========================================

@st.cache_resource
def load_face_detector():

    if not os.path.exists(FACE_MODEL_PATH):
        raise FileNotFoundError(
            "MediaPipe face model not found. "
            "Please download blaze_face_short_range.tflite."
        )

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=FACE_MODEL_PATH
        ),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=0.5
    )

    detector = FaceDetector.create_from_options(
        options
    )

    return detector


# ==========================================
# 6. FACE DETECTION AND PREPROCESSING
# ==========================================

def preprocess_image(image):

    # Convert PIL image to RGB
    image_array = np.array(
        image.convert("RGB")
    )

    height, width = image_array.shape[:2]

    detector = load_face_detector()

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_array
    )

    # Detect faces
    detection_result = detector.detect(
        mp_image
    )

    # Check face detection
    if not detection_result.detections:

        return None, None, None, None

    # Select largest detected face
    detection = max(
        detection_result.detections,
        key=lambda d: (
            d.bounding_box.width *
            d.bounding_box.height
        )
    )

    # Get face bounding box
    bbox = detection.bounding_box

    x = bbox.origin_x
    y = bbox.origin_y
    w = bbox.width
    h = bbox.height

    # Keep coordinates inside image
    x = max(0, x)
    y = max(0, y)

    w = min(w, width - x)
    h = min(h, height - y)

    if w <= 0 or h <= 0:
        return None, None, None, None

    # ======================================
    # Expand face region
    # ======================================

    center_x = x + w // 2
    center_y = y + h // 2

    # Add margin around the entire face
    side = int(max(w, h) * 1.5)

    half_side = side // 2

    x1 = center_x - half_side
    y1 = center_y - half_side

    x2 = x1 + side
    y2 = y1 + side

    crop_box = (x1, y1, x2, y2)

    # ======================================
    # Add padding outside image boundaries
    # ======================================

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)

    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    padded_gray = cv2.copyMakeBorder(
        gray,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE
    )

    # Adjust coordinates
    x1 += pad_left
    y1 += pad_top

    x2 += pad_left
    y2 += pad_top

    # ======================================
    # Crop entire face region
    # ======================================

    face = padded_gray[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None, None, None, None

    # ======================================
    # FER2013 preprocessing
    # ======================================

    face_resized = cv2.resize(
        face,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    # Normalize pixels
    face_normalized = (
        face_resized.astype("float32") / 255.0
    )

    # CNN input shape: (1, 48, 48, 1)
    model_input = face_normalized.reshape(
        1, 48, 48, 1
    )

    return (
        model_input,
        face_resized,
        (x, y, w, h),
        crop_box
    )


# ==========================================
# 7. APPLICATION HEADER
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
# 8. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 9. IMAGE PROCESSING
# ==========================================

if uploaded_file is not None:

    try:

        image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original Uploaded Image",
                use_container_width=True
            )

        # Face detection and preprocessing
        (
            model_input,
            processed_face,
            face_box,
            crop_box
        ) = preprocess_image(image)

        # ==================================
        # 10. FACE DETECTION RESULT
        # ==================================

        if model_input is None:

            st.warning(
                "No face detected. Please upload "
                "a clear face image."
            )

            st.info(
                "Use a well-lit, front-facing photo "
                "with the entire face visible."
            )

        else:

            x, y, w, h = face_box

            cx1, cy1, cx2, cy2 = crop_box

            detected_image = np.array(
                image
            ).copy()

            image_height, image_width = (
                detected_image.shape[:2]
            )

            # Green: MediaPipe detected face
            cv2.rectangle(
                detected_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # Red: Expanded crop region
            cx1 = max(0, cx1)
            cy1 = max(0, cy1)

            cx2 = min(image_width, cx2)
            cy2 = min(image_height, cy2)

            cv2.rectangle(
                detected_image,
                (cx1, cy1),
                (cx2, cy2),
                (255, 0, 0),
                2
            )

            with col2:

                st.image(
                    detected_image,
                    caption=(
                        "MediaPipe Face Detection "
                        "(Green: Face, Red: Crop)"
                    ),
                    use_container_width=True
                )

                st.image(
                    processed_face,
                    caption="CNN Input (48 x 48 Grayscale)",
                    clamp=True,
                    use_container_width=True
                )

            st.success(
                "Face detected successfully!"
            )

            # ==================================
            # 11. CNN PREDICTION
            # ==================================

            with st.spinner(
                "Analyzing facial expression..."
            ):

                probabilities = model.predict(
                    model_input,
                    verbose=0
                )[0]

            # Highest probability
            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_emotion = (
                class_names[predicted_index]
            )

            confidence = float(
                probabilities[predicted_index] * 100
            )

            # ==================================
            # 12. PREDICTION RESULT
            # ==================================

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            emotion_name = EMOTIONS.get(
                predicted_emotion,
                predicted_emotion.title()
            )

            st.success(
                f"Predicted Expression: {emotion_name}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # ==================================
            # 13. EMOTION PROBABILITIES
            # ==================================

            st.divider()

            st.subheader(
                "Emotion Probability Distribution"
            )

            for i, emotion in enumerate(class_names):

                probability = float(
                    probabilities[i]
                )

                emotion_name = EMOTIONS.get(
                    emotion,
                    emotion.title()
                )

                st.write(
                    f"**{emotion_name}** "
                    f"- {probability * 100:.2f}%"
                )

                st.progress(
                    min(
                        max(probability, 0.0),
                        1.0
                    )
                )

            st.caption(
                "Probabilities are model output scores, "
                "not calibrated measures of actual emotion."
            )

    except Exception as e:

        st.error(
            f"Unable to process this image: {e}"
        )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "MoodTune AI | Machine Learning Final Project"
)

st.caption(
    "Model: Baseline CNN | Input: 48x48 Grayscale"
)

st.caption(
    "MediaPipe Face Detection | FER2013 Preprocessing"
)
