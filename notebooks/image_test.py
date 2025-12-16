import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("images/leaf.png")

# Check if image loaded
if img is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize image
img_resized = cv2.resize(img_rgb, (224, 224))

# Display image
plt.imshow(img_resized)
plt.title("Leaf Image (Resized to 224x224)")
plt.axis("on")
plt.show()
