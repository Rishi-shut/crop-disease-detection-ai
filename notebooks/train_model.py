import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "dataset/Plant_Village/PlantVillage"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4

# ==============================
# DATA GENERATORS
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_data.num_classes
print("Detected classes:", NUM_CLASSES)

# ==============================
# MODEL
# ==============================

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 2
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 3
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Feature aggregation
    GlobalAveragePooling2D(),

    # Classifier
    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(NUM_CLASSES, activation="softmax")
])

model.summary()

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# CALLBACKS
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==============================
# TRAIN
# ==============================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ==============================
# SAVE MODEL
# ==============================

model.save("models/crop_disease_custom_cnn.h5")
print("Model saved successfully!")