# Q5 :  Implementing and Comparing CNN Architectures
# Task 2:  Implement a Residual Block and ResNet
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Add,
    Activation,
    Flatten,
    Dense
)

# Define residual block 
def residual_block(input_tensor, filters=64):

    # First convolution
    x = Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(input_tensor)

    # Second convolution
    x = Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        activation=None  # Activation is applied after the skip connection
    )(x)

    # Skip connection
    x = Add()([
        x,
        input_tensor
    ])

    # Activation after addition
    x = Activation("relu")(x)

    return x


# Define the ResNet-like model
inputs = Input(
    shape=(64, 64, 3)
)

x = Conv2D(
    64,
    kernel_size=(7, 7),
    strides=2,
    padding="same",
    activation="relu"
)(inputs)

x = residual_block( #First residual block
    x,
    filters=64
)

x = residual_block( #Second residual block
    x,
    filters=64
)

x = Flatten()(x)
x = Dense( #Dense layer
    128,
    activation="relu"
)(x)
outputs = Dense( #output layer
    10,
    activation="softmax"
)(x)
resnet_model = Model(
    inputs=inputs,
    outputs=outputs
)
resnet_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("\nResNet-like Model Summary:\n")

resnet_model.summary()