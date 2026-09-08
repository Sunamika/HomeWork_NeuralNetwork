import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing pixel values to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Adam Model
adam_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

adam_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training Adam Model...")

adam_history = adam_model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    verbose=1
)


# SGD Model
sgd_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

sgd_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining SGD Model...")

sgd_history = sgd_model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    verbose=1
)


# Plot Training Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    adam_history.history["accuracy"],
    label="Adam"
)

plt.plot(
    sgd_history.history["accuracy"],
    label="SGD"
)

plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy: Adam vs. SGD")
plt.legend()
plt.show()


# Plot Validation Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    adam_history.history["val_accuracy"],
    label="Adam"
)

plt.plot(
    sgd_history.history["val_accuracy"],
    label="SGD"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy: Adam vs. SGD")
plt.legend()
plt.show()