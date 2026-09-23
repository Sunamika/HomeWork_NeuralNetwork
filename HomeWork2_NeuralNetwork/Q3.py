# Question 3: Convolution Operations with Different Parameters
import numpy as np
import tensorflow as tf


# 1. Define the following 5×5 input matrix:
input_matrix = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# 2. Define the following 3×3 kernel:
kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

# Reshape the input and kernel to match the expected dimensions
input_tensor = input_matrix.reshape(
    1, 5, 5, 1
)
kernel_tensor = kernel.reshape(
    3, 3, 1, 1
)

# Function for performing convolution
def perform_convolution(stride, padding):

    result = tf.nn.conv2d(
        input_tensor,
        kernel_tensor,
        strides=[1, stride, stride, 1],
        padding=padding
    )
    return result.numpy().squeeze()


# 3. Perform convolution operations with:
valid_s1 = perform_convolution(1, "VALID")
same_s1 = perform_convolution(1, "SAME")

valid_s2 = perform_convolution(2, "VALID")
same_s2 = perform_convolution(2, "SAME")


# 4. Print the output feature maps for each case.
print("Stride = 1, Padding = VALID")
print(valid_s1)

print("\nStride = 1, Padding = SAME")
print(same_s1)

print("\nStride = 2, Padding = VALID")
print(valid_s2)

print("\nStride = 2, Padding = SAME")
print(same_s2)