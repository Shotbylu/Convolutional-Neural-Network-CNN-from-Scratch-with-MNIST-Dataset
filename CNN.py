import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

# Load and preprocess the MNIST dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Normalize pixel values to range [0, 1]
train_X = train_X / 255.0
test_X = test_X / 255.0

# Reshape images to (batch_size, height, width, channels)
train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)

# One-hot encode the labels
train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y, 10)

# Initialize weights and biases for layers
def initialize_weights(shape):
    return np.random.randn(*shape) * 0.01

class CNNFromScratch:
    def __init__(self, input_shape, num_classes, epochs=10, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.loss_history = []
        self.weights = {
            "conv1": initialize_weights((3, 3, 1, 8)),  # 3x3 filters, 1 input channel, 8 output channels
            "fc1": initialize_weights((8 * 13 * 13, 128)),  # Flattened input to fully connected layer
            "fc2": initialize_weights((128, num_classes))  # Fully connected output layer
        }
        self.biases = {
            "conv1": np.zeros((8,)),
            "fc1": np.zeros((128,)),
            "fc2": np.zeros((num_classes,))
        }

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, predictions, labels):
        n_samples = labels.shape[0]
        return -np.sum(labels * np.log(predictions + 1e-9)) / n_samples

    def convolve(self, X, W, b):
        (n_filters, filter_height, filter_width, _) = W.shape
        n_samples, height, width, channels = X.shape
        output_height = height - filter_height + 1
        output_width = width - filter_width + 1
        output = np.zeros((n_samples, output_height, output_width, n_filters))

        for sample in range(n_samples):
            for filter_idx in range(n_filters):
                for i in range(output_height):
                    for j in range(output_width):
                        region = X[sample, i:i+filter_height, j:j+filter_width, :]
                        output[sample, i, j, filter_idx] = np.sum(region * W[filter_idx]) + b[filter_idx]

        return self.relu(output)

    def max_pooling(self, X, size=2, stride=2):
        n_samples, height, width, channels = X.shape
        output_height = (height - size) // stride + 1
        output_width = (width - size) // stride + 1
        output = np.zeros((n_samples, output_height, output_width, channels))

        for sample in range(n_samples):
            for c in range(channels):
                for i in range(0, height - size + 1, stride):
                    for j in range(0, width - size + 1, stride):
                        region = X[sample, i:i+size, j:j+size, c]
                        output[sample, i//stride, j//stride, c] = np.max(region)

        return output

    def flatten(self, X):
        return X.reshape(X.shape[0], -1)

    def forward(self, X):
        self.conv1_output = self.convolve(X, self.weights["conv1"], self.biases["conv1"])
        self.pool1_output = self.max_pooling(self.conv1_output)
        self.flatten_output = self.flatten(self.pool1_output)
        self.fc1_output = self.relu(np.dot(self.flatten_output, self.weights["fc1"]) + self.biases["fc1"])
        self.fc2_output = np.dot(self.fc1_output, self.weights["fc2"]) + self.biases["fc2"]
        return self.softmax(self.fc2_output)

    def backward(self, X, y, output):
        n_samples = X.shape[0]

        # Compute gradient for fully connected layer 2
        error = output - y
        dW_fc2 = np.dot(self.fc1_output.T, error) / n_samples
        db_fc2 = np.sum(error, axis=0) / n_samples

        # Gradient for fully connected layer 1
        error_fc1 = np.dot(error, self.weights["fc2"].T) * self.relu_derivative(self.fc1_output)
        dW_fc1 = np.dot(self.flatten_output.T, error_fc1) / n_samples
        db_fc1 = np.sum(error_fc1, axis=0) / n_samples

        # Gradient descent parameter update
        self.weights["fc2"] -= self.learning_rate * dW_fc2
        self.biases["fc2"] -= self.learning_rate * db_fc2
        self.weights["fc1"] -= self.learning_rate * dW_fc1
        self.biases["fc1"] -= self.learning_rate * db_fc1

    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            loss = self.cross_entropy_loss(output, y)
            self.loss_history.append(loss)
            self.backward(X, y, output)

            if epoch % 1 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

        # Plot loss history
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Trend')
        plt.show()

# Example usage (simplified for clarity):
# cnn = CNNFromScratch(input_shape=(28, 28, 1), num_classes=10, epochs=5, learning_rate=0.01)
# cnn.train(train_X[:1000], train_y[:1000])
