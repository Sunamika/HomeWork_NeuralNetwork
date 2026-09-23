# Q4: Implement Max Pooling and Average Pooling
#Task2 : Implement Max Pooling and Average Pooling
import numpy as np
import tensorflow as tf


# Make the random result reproducible
np.random.seed(42)

# random 4x4 matrix
input_matrix = np.random.randint(
    1,
    10,
    size=(4, 4)
).astype(np.float32)

print("Original 4x4 Matrix:")
print(input_matrix)


# Convert to TensorFlow format:[batch, height, width, channels]
input_tensor = input_matrix.reshape(
    1, 4, 4, 1
)

# Max Pooling
max_pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2
)

max_output = max_pool(input_tensor)

# Average Pooling
average_pool = tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=2
)

average_output = average_pool(input_tensor)


# Print results
print("\nMax Pooled Matrix:")
print(max_output.numpy().squeeze())

print("\nAverage Pooled Matrix:")
print(average_output.numpy().squeeze())