# 1: Implementing an RNN for Text Generation

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. Load the Shakespeare text dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
path = tf.keras.utils.get_file("shakespeare.txt", url)

with open(path, "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))
print("\nSample text:")
print(text[:500])


# 2. Convert text into character sequences
# Finding unique characters
vocab = sorted(set(text))
print("\nNumber of unique characters:", len(vocab))

# Mappings between Character-to-integer and integer-to-character 
char_to_idx = {char: index for index, char in enumerate(vocab)}
idx_to_char = np.array(vocab)

# Converting the entire text into integers
text_as_int = np.array(
    [char_to_idx[char] for char in text],
    dtype=np.int32
)
print("\nFirst 20 encoded characters:")
print(text_as_int[:20])

sequence_length = 100
# Create TensorFlow dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(
    sequence_length + 1,
    drop_remainder=True
)


# Split sequence into input and target
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]

    return input_text, target_text


dataset = sequences.map(split_input_target)

# Batching and shuffling the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# 3. Define the LSTM model
vocab_size = len(vocab)
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=256
    ),

    LSTM(
        512,
        return_sequences=True
    ),

    Dense(vocab_size)
])

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)

model.compile(
    optimizer="adam",
    loss=loss_function,
    metrics=["accuracy"]
)

model.build((None, sequence_length))

print("\nModel Summary:")
model.summary()

# 4. Train the model
EPOCHS = 10

history = model.fit(
    dataset,
    epochs=EPOCHS
)

# Generating new text
def generate_text(
        model,
        start_string,
        num_generate=500,
        temperature=0.8):

    # Converting start string to integer representation
    input_ids = [
        char_to_idx[char]
        for char in start_string
        if char in char_to_idx
    ]

    generated_text = start_string

    # Generating characters 
    for _ in range(num_generate):
        current_input = input_ids[-sequence_length:]
        current_input = tf.expand_dims(  # Adding batch dimension
            current_input,
            0
        )
        predictions = model( # Predicting the next character
            current_input,
            training=False
        )
        predictions = predictions[:, -1, :]
        predictions = predictions / temperature

        # Sampling from the predicted distribution to get the next character ID
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[0, 0].numpy()

        # Converted predicted ID back into character
        predicted_character = idx_to_char[predicted_id]
        generated_text += predicted_character
        input_ids.append(predicted_id)

    return generated_text


# Generated 500 characters
print("\nGenerated Text:\n")
generated_text = generate_text(
    model,
    start_string="ROMEO: ",
    num_generate=500,
    temperature=0.8
)

print(generated_text)