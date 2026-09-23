
# Q4: CNN Feature Extraction with Filters and Pooling
#Task1 : Implement Edge Detection Using Convolution
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loading grayscale image
image = cv2.imread(
    "sample.jpg",
    cv2.IMREAD_GRAYSCALE
)

# Check whether image loaded successfully or not
if image is None:
    raise FileNotFoundError(
        "sample.jpg was not found. "
        "Place an image named sample.jpg in the same folder."
    )
    
# Define Sobel filters
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)


sobel_y = np.array([
    [-1, -2, -1],
    [0,   0,  0],
    [1,   2,  1]
], dtype=np.float32)


# Apply Sobel filters
edge_x = cv2.filter2D(
    image,
    cv2.CV_32F,
    sobel_x
)

edge_y = cv2.filter2D(
    image,
    cv2.CV_32F,
    sobel_y
)

# Convert values for visualization
edge_x_display = cv2.convertScaleAbs(edge_x)
edge_y_display = cv2.convertScaleAbs(edge_y)


# Displaying three images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edge_x_display, cmap="gray")
plt.title("Sobel-X")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edge_y_display, cmap="gray")
plt.title("Sobel-Y")
plt.axis("off")

plt.tight_layout()
plt.show()
