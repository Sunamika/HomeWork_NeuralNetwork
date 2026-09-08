import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# True values
y_true = tf.constant([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=tf.float32)

# Original predictions
y_pred1 = tf.constant([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7]
], dtype=tf.float32)

# Modified predictions
y_pred2 = tf.constant([
    [0.6, 0.2, 0.2],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
], dtype=tf.float32)

# Creating loss functions
mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()

# Loss calculations
mse_loss1 = mse(y_true, y_pred1)
mse_loss2 = mse(y_true, y_pred2)

cce_loss1 = cce(y_true, y_pred1)
cce_loss2 = cce(y_true, y_pred2)

# Printing results
print("Original Predictions:")
print("MSE Loss:", mse_loss1.numpy())
print("CCE Loss:", cce_loss1.numpy())

print("\nModified Predictions:")
print("MSE Loss:", mse_loss2.numpy())
print("CCE Loss:", cce_loss2.numpy())

# Preparing data for bar chart
loss_names = ["MSE", "Cross-Entropy"]

original_losses = [
    mse_loss1.numpy(),
    cce_loss1.numpy()
]

modified_losses = [
    mse_loss2.numpy(),
    cce_loss2.numpy()
]

x = np.arange(len(loss_names))
width = 0.35

# Creating bar chart
plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    original_losses,
    width,
    label="Original Predictions"
)

plt.bar(
    x + width / 2,
    modified_losses,
    width,
    label="Modified Predictions"
)

plt.xlabel("Loss Function")
plt.ylabel("Loss Value")
plt.title("MSE vs. Categorical Cross-Entropy")
plt.xticks(x, loss_names)
plt.legend()

plt.show()