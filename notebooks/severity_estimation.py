import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("images/leaf2.jpg")
img = cv2.resize(img, (224, 224))

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for diseased (brown/yellow) regions
lower = np.array([10, 40, 40])
upper = np.array([30, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower, upper)

# Calculate areas
infected_area = cv2.countNonZero(mask)
total_area = img.shape[0] * img.shape[1]

severity = (infected_area / total_area) * 100

# Severity level
if severity < 30:
    level = "Mild"
elif severity < 60:
    level = "Moderate"
else:
    level = "Severe"

print(f"Severity: {severity:.2f}% ({level})")

# Show mask
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Infected Area Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()
