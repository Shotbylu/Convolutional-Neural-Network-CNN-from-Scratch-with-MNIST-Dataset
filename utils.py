# utils.py

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils


def load_data():
    """
    Load and preprocess the MNIST dataset.
    Returns normalized data and one-hot encoded labels.
    """
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Normalize pixel values to the range [0, 1]
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    # Reshape images to (batch_size, height, width, channels)
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)

    # One-hot encode the labels
    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)

    return (train_X, train_y), (test_X, test_y)


def plot_loss(loss_history):
    """
    Plot the training loss trend over epochs.
    """
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Trend')
    plt.show()
