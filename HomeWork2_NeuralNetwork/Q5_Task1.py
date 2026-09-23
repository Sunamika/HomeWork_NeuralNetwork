
# Q5: Implementing and Comparing CNN Architectures
#Task 1: Implement AlexNet Architecture

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
alexnet = Sequential([

    # First convolution
    Conv2D(
        96,
        kernel_size=(11, 11),
        strides=4,
        activation="relu",
        input_shape=(227, 227, 3)
    ),

    MaxPooling2D(
        pool_size=(3, 3),
        strides=2
    ),

    # Second convolution
    Conv2D(
        256,
        kernel_size=(5, 5),
        padding="same",
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(3, 3),
        strides=2
    ),

    # Third convolution
    Conv2D(
        384,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    # Fourth convolution
    Conv2D(
        384,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    # Fifth convolution
    Conv2D(
        256,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(3, 3),
        strides=2
    ),

    # Flatten features
    Flatten(),

    # Fully connected layer
    Dense(
        4096,
        activation="relu"
    ),

    Dropout(0.5),

    # Second fully connected layer
    Dense(
        4096,
        activation="relu"
    ),

    Dropout(0.5),

    # 10-class output
    Dense(
        10,
        activation="softmax"
    )
])

alexnet.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nAlexNet Model Summary:\n")

alexnet.summary()