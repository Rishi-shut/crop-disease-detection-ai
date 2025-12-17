import os

DATASET_PATH = "dataset/Plant_Village/PlantVillage"

# List all class folders
classes = sorted(os.listdir(DATASET_PATH))

print("Number of classes (diseases):", len(classes))
print("\nFirst 10 classes:")
for cls in classes[:10]:
    print(" -",cls)

# Count total images
total_images = 0
for cls in classes:
    class_path = os.path.join(DATASET_PATH, cls)
    images = os.listdir(class_path)
    total_images += len(images)

print("\nTotal images in dataset:", total_images)
