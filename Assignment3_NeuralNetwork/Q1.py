import numpy as np
input_matrix = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Define the filter matrix (3 x 3)
filter_matrix = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])
# Stride and padding
stride = 1
padding = 0

# Dimensions of input and filter matrices
input_height, input_width = input_matrix.shape
filter_height, filter_width = filter_matrix.shape

# Calculate output dimensions
output_height = ((input_height - filter_height + 2 * padding) // stride) + 1
output_width = ((input_width - filter_width + 2 * padding) // stride) + 1

# Create an empty output matrix
output = np.zeros((output_height, output_width), dtype=int)

# Slide the filter over the input matrix and compute the convolution
for i in range(output_height):
    for j in range(output_width):
        row_start = i * stride
        col_start = j * stride
        region = input_matrix[ 
            row_start:row_start + filter_height,
            col_start:col_start + filter_width
        ]
        output[i, j] = np.sum(region * filter_matrix)

print("Input Matrix:")
print(input_matrix)

print("\nFilter:")
print(filter_matrix)

print("\nOutput Feature Map:")
print(output)

print("\nOutput Shape:")
print(output.shape)