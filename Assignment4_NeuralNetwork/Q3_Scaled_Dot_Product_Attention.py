
import numpy as np

# Implement softmax function
def softmax(x):
    # Subtract row maximum to prevent overflow
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Implement scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):
    # Step 1: Compute the dot product of Q and K transpose
    scores = np.matmul(Q, K.T)

    # Step 2: Scale scores by the square root of key dimension
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply softmax to obtain attention weights
    attention_weights = softmax(scaled_scores)

    # Step 4: Multiply attention weights by V
    output = np.matmul(attention_weights, V)

    return attention_weights, output


# Given Input matrices
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])

K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])

V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])


weights, output = scaled_dot_product_attention(Q, K, V)

# Display results to four decimal places
np.set_printoptions(precision=4, suppress=True)

print("Attention Weights Matrix:")
print(weights)

print("\nFinal Output Matrix:")
print(output)
