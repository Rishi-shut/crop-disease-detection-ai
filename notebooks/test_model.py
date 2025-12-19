import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = load_model("models/crop_disease_model.h5")

# Load class labels
DATASET_PATH = "dataset/Plant_Village/PlantVillage"
class_names = sorted(os.listdir(DATASET_PATH))

# Load and preprocess image
img_path = ""
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print("Predicted Disease:", predicted_class)
print(f"Confidence: {confidence:.2f}%")
