# Homework 2 – Neural Networks

## Overview

This repository contains the Python implementations for Homework 2 on Neural Networks. The assignment covers recurrent neural networks (RNNs), LSTM models, convolution operations, CNN feature extraction, pooling, AlexNet, and residual networks.

## Files

* `Q1.py` – Implements an LSTM-based RNN for character-level text generation using the Shakespeare dataset.
* `Q2.py` – Implements sentiment classification on the IMDB movie review dataset using an LSTM network and evaluates the model using a confusion matrix and classification report.
* `Q3.py` – Performs 2D convolution on a 5×5 input matrix using different stride and padding settings.
* `Q4_Task1.py` – Performs edge detection on an image using Sobel-X and Sobel-Y convolution filters.
* `Q4_Task2.py` – Demonstrates max pooling and average pooling on a randomly generated 4×4 matrix.
* `Q5_Task1.py` – Implements the AlexNet CNN architecture.
* `Q5_Task2.py` – Implements residual blocks and a simple ResNet-like architecture.
* `sample.jpg` – Sample image used for the Sobel edge detection task.

## Requirements

The programs use Python 3 and the following libraries:

* TensorFlow
* NumPy
* Matplotlib
* OpenCV
* scikit-learn

The required packages can be installed using:

```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

## Running the Programs

Open a terminal in the project directory and run the required Python file. For example:

```bash
python Q1.py
```

Similarly, other questions can be executed using:

```bash
python Q2.py
python Q3.py
python Q4_Task1.py
python Q4_Task2.py
python Q5_Task1.py
python Q5_Task2.py
```

For `Q4_Task1.py`, make sure that `sample.jpg` is located in the same directory as the Python file.

## Topics Covered

This homework demonstrates several important neural network concepts, including:

* Recurrent Neural Networks and LSTM
* Text generation
* Sentiment classification
* Convolution operations
* Padding and stride
* Sobel edge detection
* Max pooling and average pooling
* Convolutional Neural Networks
* AlexNet architecture
* Residual connections
* ResNet architecture
