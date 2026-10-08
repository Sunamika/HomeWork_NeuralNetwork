# CS5720 — Neural Network and Deep Learning
## Home Assignment 3 — Programming Questions

**Student Name:** Sunamika Karki  
**Student ID:** 700788005  
**University:** University of Central Missouri  
**Semester:** Fall 2026

## Overview

This assignment includes two programming questions focusing on convolutional neural networks (CNNs) and transfer learning.

In Question 1, I implemented a 2D convolution operation from scratch using Python and NumPy. In Question 2, I used a pretrained ResNet18 model to compare frozen feature extraction and fine-tuning using the CIFAR-10 dataset.

## Question 1 — Implement Convolution from Scratch

### Description

For this question, I wrote a Python program to perform 2D convolution without using any built-in convolution function.

I used a 5 × 5 input matrix and a 3 × 3 filter with stride 1 and padding 0.

The program uses nested loops to move the filter across the input matrix. At each position, it multiplies the corresponding values and adds them together to calculate the output feature map.

### Output

```text
Output Feature Map:
[[4 3 4]
 [2 4 3]
 [2 3 4]]

Output Shape:
(3, 3)
```

### Effect of Changing Stride

With stride 1, the filter moves one position at a time and produces a 3 × 3 output. When the stride is increased to 2, the filter moves two positions at a time, reducing the output size to 2 × 2.

## Question 2 — Transfer Learning: Freeze vs. Fine-Tune

### Description

For this question, I used a pretrained ResNet18 model and the CIFAR-10 dataset to compare two transfer learning approaches.

The dataset contains 50,000 training images and 10,000 testing images from 10 different classes.

Before training, I resized the images to 224 × 224 pixels and normalized them using ImageNet normalization values.

I performed two experiments, training each model for five epochs.

### Experiment A — Frozen Feature Extraction

In this experiment, I froze all the pretrained ResNet18 layers and replaced the final classification layer with a new layer for the 10 CIFAR-10 classes.

Only the new classification layer was trained.

**Results:**
- Trainable parameters: 5,130
- Training time: 679.71 seconds
- Test accuracy: 80.67%

### Experiment B — Fine-Tuning

In this experiment, I froze most of the pretrained ResNet18 layers but unfroze the final convolutional block (`layer4`).

I trained this block along with the new classification layer.

**Results:**
- Trainable parameters: 8,398,858
- Training time: 2,747.54 seconds
- Test accuracy: 90.67%

### Comparison of Results

| Method | Trainable Parameters | Training Time (seconds) | Test Accuracy |
|---|---:|---:|---:|
| Frozen Feature Extraction | 5,130 | 679.71 | 80.67% |
| Fine-Tuning | 8,398,858 | 2,747.54 | 90.67% |

### Training Loss

I recorded the training loss for both experiments and plotted the results in a graph.

The frozen feature extractor's training loss decreased from 0.8260 to 0.5673, while the fine-tuned model's training loss decreased from 0.4240 to 0.0179.

The fine-tuned model achieved higher test accuracy but required more training time.

## Tools and Libraries

- Python
- NumPy
- PyTorch
- Torchvision
- Matplotlib
- Pandas
- Jupyter Notebook
- VS Code

## How to Run

1. Clone or download the GitHub repository.
2. Open the project folder in VS Code or Jupyter Notebook.
3. Install the required libraries:

   ```bash
   pip install numpy torch torchvision matplotlib pandas notebook
   ```

4. Open the Question 1 notebook and run the cells to view the convolution results.
5. Open the Question 2 notebook and run the cells to train both models.
6. After training finishes, the notebook displays the training-loss graph, accuracy, training time, and comparison table.

**Note:** The CIFAR-10 dataset and pretrained ResNet18 weights will download automatically if needed. Training time may vary depending on the computer.

## Conclusion

This assignment helped me understand how convolution and transfer learning work in practice.

In Question 1, I learned how a filter moves across an input matrix and calculates the output feature map without using a built-in convolution function.

In Question 2, I learned how freezing and fine-tuning pretrained layers affect training time and accuracy.

Based on my results, frozen feature extraction was faster, while fine-tuning achieved better accuracy. Fine-tuning improved test accuracy by 10 percentage points but took considerably longer to train.
