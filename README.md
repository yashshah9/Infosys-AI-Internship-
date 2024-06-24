# Infosys Internship (Group V)
## Project - Handwritten Digit Recognition with Multiple Models in PyTorch

## Project Overview
The MNIST dataset is a benchmark in the field of computer vision and machine learning. It contains 60,000 training images and 10,000 testing images of handwritten digits. Our goal is to develop models that accurately classify these digits. We explore Logistic Regression, a Multilayer Perceptron (MLP), and the LeNet-5 Convolutional Neural Network, all implemented using PyTorch.

## Table Of Contents
- [Introduction](#introduction)
- [Flow Of Project](#flow-of-project)
- [Dataset](#dataset)
- [Models](#models)
  - [Logistic Regression](#logistic-regression)
  - [Multilayer Perceptron (MLP)](#multilayer-perceptron-mlp)
  - [LeNet-5](#lenet-5)
- [Contributing](#contributing)

## Introduction
Handwritten digit recognition is a fundamental problem in computer vision and machine learning. In this project, we aim to develop and compare different models using PyTorch to identify handwritten digits from the MNIST dataset. Our objective is to create highly accurate digit recognition systems capable of classifying digits with high precision.

## Flow Of Project
### The MNIST Handwritten Digit Recognition Problem
- Introduce the MNIST dataset and its significance in the field of machine learning.
- Explain the problem of handwritten digit recognition and its applications.

### Loading the MNIST Dataset in PyTorch
- Provide instructions for downloading and loading the MNIST dataset using PyTorch’s DataLoader.
- Preprocess the dataset, including normalization and data augmentation techniques.

## Dataset
The MNIST dataset consists of grayscale images of size 28x28 pixels, each representing a digit from 0 to 9. The dataset is split into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

![MNIST Sample](https://github.com/yashshah9/Infosys-AI-Internship-/assets/160280438/a06ada1d-b668-448f-8b7d-5ad09b570258)

## Models
### Logistic Regression
A simple model for binary and multi-class classification problems. For MNIST, it treats each pixel as a feature.

![fig-2](https://github.com/yashshah9/Infosys-AI-Internship-/assets/160280438/435aee7c-772f-4a82-99db-5a90e6df546e)


### Multilayer Perceptron (MLP)
A type of neural network with one or more hidden layers. For MNIST, our MLP model consists of:
- Input layer with 784 units (one for each pixel)
- One or more hidden layers with ReLU activation
- Output layer with 10 units and softmax activation
- 
![fig-3](https://github.com/yashshah9/Infosys-AI-Internship-/assets/160280438/f148023d-df94-46ff-9161-770cb7a81e5e)


### LeNet-5
A classic CNN architecture proposed by Yann LeCun. It consists of:
- Two convolutional layers
- Two subsampling (pooling) layers
- Two fully connected layers
- Output layer with softmax activation

![LeNet-5 Architecture](https://github.com/yashshah9/Infosys-AI-Internship-/assets/160280438/9bc8308a-c159-4b3f-aa10-24c97581d719)

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request with your improvements.

---

This README provides a comprehensive overview of the project, including the problem statement, dataset details, model architectures, and instructions for contributing. Adjust URLs and paths for images as necessary to match your repository structure.
