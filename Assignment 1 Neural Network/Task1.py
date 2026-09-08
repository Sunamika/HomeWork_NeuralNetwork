import tensorflow as tf

# 1. Creating a random tensor of shape (4, 6)
tensor = tf.random.normal((4, 6))

print("Original Tensor:")
print(tensor)

# 2. Finding rank and shape
print("\nBefore Reshaping:")
print("Rank:", tf.rank(tensor).numpy())
print("Shape:", tf.shape(tensor).numpy())

# 3. Reshape tensor into (2, 3, 4)
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))

print("\nAfter Reshaping:")
print("Rank:", tf.rank(reshaped_tensor).numpy())
print("Shape:", tf.shape(reshaped_tensor).numpy())

# Transpose to (3, 2, 4)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])

print("\nAfter Transposing:")
print("Rank:", tf.rank(transposed_tensor).numpy())
print("Shape:", tf.shape(transposed_tensor).numpy())

# 4. Creating a smaller tensor with shape (1, 4)
small_tensor = tf.random.normal((1, 4))

print("\nSmall Tensor Shape:")
print(tf.shape(small_tensor).numpy())

# Adding using broadcasting
result = transposed_tensor + small_tensor

print("\nResult Shape After Broadcasting:")
print(tf.shape(result).numpy())