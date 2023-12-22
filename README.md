# CNN-Cifar 10 Project

Welcome to the CNN-Cifar 10 project! This project showcases various versions of Convolutional Neural Networks (CNNs) trained on the Cifar 10 dataset. These CNN models are designed for image classification tasks. Below, you'll find essential information to understand, set up, and use this project. This project was mainly intended for learning purposes, as I wanted to deepen my understanding of CNNs as well as some other DL-concepts.

## Overview

This project demonstrates the development of CNN models for image classification using the Cifar 10 dataset. The models are organized into versions (V0 to V7), each with its unique characteristics and improvements. You can explore different model architectures and training strategies by analyzing the code and running experiments.

## Features
- V0-V7 use keras.datasets
- V0: Base MLP model
- V1: First CNN model
- V2: Bigger model architecture than V1
- V3: use pre-training data augmentation
- V4: use complexer pre-training data augmentation than V3
- V5: use online data augmentation
- V6: further increase model architecture complexity
- V7: introduce a lr decayer

- V8-V13 use tfds 
- V8: base model for tfds (V2 architecture)
- V9: more callbacks (plateau, early-stopping, checkpoint)
- V10: Tensorboard 
- V11: variety of augmentation techniques (tf.image, custom layer, mixup, cutmix, Alubmentation)
- V12: hp search (grid, random, keras tuner)
- V13: 

## Technologies Used

This project leverages the following technologies and libraries:

- TensorFlow and Keras for deep learning model development.
- Matplotlib for data visualization.
- NumPy for numerical operations.
- Scikit-learn for generating confusion matrices.
- TensorFlow's ImageDataGenerator for data augmentation.

## Installation

To set up and use this project, follow these steps:

1. Clone this repository to your local machine.

2. Ensure you have the necessary dependencies installed, including TensorFlow, Matplotlib, NumPy, and Scikit-learn. You can install these using pip: pip install tensorflow matplotlib numpy scikit-learn


## Usage

To use this project effectively, follow these guidelines:

- Experiment with different model versions to understand their strengths and weaknesses.
- Modify data augmentation techniques to see how they impact model performance.
- Analyze training history and confusion matrices to evaluate model performance.

## License
This project is licensed under the [MIT License](LICENSE).

