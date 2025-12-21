import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
from remedies import REMEDIES



# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Crop Disease Detection",
    layout="centered"
)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/crop_disease_model.h5")

model = load_model()

# ----------------- CLASS NAMES -----------------
DATASET_PATH = "dataset/Plant_Village/PlantVillage"
class_names = sorted(os.listdir(DATASET_PATH))

# ----------------- IMAGE PREPROCESS -----------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def estimate_severity(pil_img):
    # Convert PIL → OpenCV (RGB → BGR)
    img = np.array(pil_img)
    img = cv2.resize(img, (224, 224))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Color range for infected regions (yellow/brown)
    lower = np.array([10, 40, 40])
    upper = np.array([30, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Calculate severity
    infected_area = cv2.countNonZero(mask)
    total_area = mask.shape[0] * mask.shape[1]
    severity = (infected_area / total_area) * 100

    # Severity level
    if severity < 30:
        level = "Mild"
    elif severity < 60:
        level = "Moderate"
    else:
        level = "Severe"

    return severity, level, mask

# ----------------- UI -----------------
st.title("🌱 AI-Based Crop Disease Detection")
st.write("Upload a leaf image to detect disease and get treatment recommendations.")

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    if st.button("🔍 Detect Disease"):
        with st.spinner("Analyzing leaf image..."):
            input_img = preprocess_image(image)
            predictions = model.predict(input_img)
            pred_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            disease = class_names[pred_index]

        st.success("Prediction complete!")

        st.subheader("🦠 Disease Detected")
        st.write(f"**{disease}**")

        st.subheader("📊 Confidence")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}%")

       
        # -------- SEVERITY ESTIMATION --------
        severity, level, mask = estimate_severity(image)

        st.subheader("⚠️ Disease Severity")
        st.write(f"**{severity:.2f}% — {level}**")

        # Visuals
        col1, col2 = st.columns(2)

        with col1:
            st.caption("Original Image")
            st.image(image, width=400)

        with col2:
            st.caption("Infected Area (Estimated)")
            st.image(mask, width=400)
            # -------- REMEDY RECOMMENDATIONS --------
            st.subheader("💊 Recommended Remedies")

            if disease in REMEDIES:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🌿 Organic Remedies")
                    for r in REMEDIES[disease]["organic"]:
                        st.write("•", r)

                with col2:
                    st.markdown("### 🧪 Chemical Remedies")
                    for r in REMEDIES[disease]["chemical"]:
                        st.write("•", r)
            else:
                st.warning("No remedy information available for this disease.")


