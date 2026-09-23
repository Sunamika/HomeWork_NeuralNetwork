# Question 2: Sentiment Classification Using RNN

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
# 1. Load the IMDB sentiment dataset
VOCAB_SIZE = 10000
MAX_LENGTH = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# 2. Preprocess the text data by tokenization and padding sequences.
x_train = pad_sequences(
    x_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

x_test = pad_sequences(
    x_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

# 3. Train an LSTM-based model to classify reviews as positive or negative.
model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128
    ),

    LSTM(64),

    Dropout(0.5),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.build((None, MAX_LENGTH))

model.summary()

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)
print("\nTest Accuracy:", test_accuracy)


# 4. Generate a confusion matrix and classification report (accuracy, precision, recall, F1-score).
y_probability = model.predict(x_test) # Predicted probabilities

# Convert probabilities into class labels
y_pred = (y_probability > 0.5).astype("int32").flatten()

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred
)
print("\nConfusion Matrix:")
print(cm)


# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)
display.plot()
plt.title("IMDB Sentiment Classification Confusion Matrix")
plt.show()