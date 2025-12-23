from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
from remedies import REMEDIES
import base64
from io import BytesIO

# ----------------- INIT APP -----------------
app = FastAPI(title="Crop Disease Detection API")

# ----------------- LOAD MODEL -----------------
model = tf.keras.models.load_model("models/crop_disease_model.h5")

# ----------------- CLASS NAMES -----------------
DATASET_PATH = "dataset/Plant_Village/PlantVillage"
class_names = sorted(os.listdir(DATASET_PATH))

# ----------------- CONSTANTS -----------------
ALLOWED_TYPES = ["image/jpeg", "image/png" ,"image/jpg"]
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# ----------------- IMAGE PREPROCESS -----------------
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------- SEVERITY ESTIMATION -----------------
def estimate_severity(pil_img):
    img = np.array(pil_img)
    img = cv2.resize(img, (224, 224))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 40, 40])
    upper = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    infected_area = cv2.countNonZero(mask)
    total_area = mask.shape[0] * mask.shape[1]
    severity = (infected_area / total_area) * 100

    if severity < 30:
        level = "Mild"
    elif severity < 60:
        level = "Moderate"
    else:
        level = "Severe"

    # Encode mask as Base64
    _, buffer = cv2.imencode(".png", mask)
    mask_base64 = base64.b64encode(buffer).decode("utf-8")

    return severity, level, mask_base64

# ----------------- ROOT ENDPOINT -----------------
@app.get("/")
def root():
    return {"message": "Crop Disease Detection API is running"}

# ----------------- PREDICT ENDPOINT -----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read file
    contents = await file.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    # Load image
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Prediction
    input_img = preprocess_image(image)
    preds = model.predict(input_img)
    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)
    disease = class_names[pred_index]

    # Severity + mask
    severity, level, mask_b64 = estimate_severity(image)

    # Remedies
    remedies = REMEDIES.get(disease, {})

    return {
        "disease": disease,
        "confidence": round(confidence, 2),
        "severity_percent": round(severity, 2),
        "severity_level": level,
        "mask": mask_b64,
        "remedies": remedies
    }
