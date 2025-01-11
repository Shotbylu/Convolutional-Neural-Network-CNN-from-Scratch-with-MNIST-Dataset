# CNN from Scratch with MNIST Dataset

This repository implements a Convolutional Neural Network (CNN) from scratch without using frameworks like TensorFlow or PyTorch. The project demonstrates the foundational principles of CNN architecture, forward and backward propagation, and training using the MNIST dataset.

## **Project Structure**

- **`cnn.py`**: Core implementation of the CNN architecture, including convolution, pooling, activation functions, and fully connected layers.
- **`train.py`**: Script for training the CNN on the MNIST dataset.
- **`utils.py`**: Contains helper functions for data preprocessing and visualization.
- **`README.md`**: Documentation for understanding and reproducing the project.

## **How It Works**

### **1. Convolutional Neural Network (CNN) Architecture**

The CNN implemented consists of:

- **Convolution Layer**: Extracts feature maps using learnable filters.
- **ReLU Activation**: Introduces non-linearity.
- **Max Pooling Layer**: Reduces the spatial dimensions of the feature maps.
- **Flatten Layer**: Converts the feature maps into a 1D array.
- **Fully Connected Layer**: Performs classification using learned features.
- **Softmax with Cross-Entropy Loss**: Computes probabilities for classification.

### **2. Training the Model**

- **Dataset**: The MNIST dataset (28x28 grayscale images of digits 0-9) is used for training.
- **Normalization**: The pixel values are normalized to the range [0, 1].
- **One-Hot Encoding**: Labels are converted to a format suitable for multi-class classification.
- **Backpropagation**: Gradient descent is used to update weights and biases.

### **3. Visualization**

Training loss trends are plotted over epochs to visualize model performance.

## **Installation Instructions**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/cnn-from-scratch.git
   cd cnn-from-scratch
   ```

2. **Install Dependencies:**
   Ensure you have `numpy`, `matplotlib`, and `keras` installed.
   ```bash
   pip install numpy matplotlib keras
   ```

3. **Run the Training Script:**
   ```bash
   python train.py
   ```

## **Key Functions Explained**

### **ConvLayer**
- **Purpose**: Extracts features from input images.
- **Forward Pass**: Applies filters over input data to compute feature maps.
- **Backward Pass**: Computes gradients to update filters and biases.

### **MaxPoolLayer**
- **Purpose**: Reduces the spatial dimensions of feature maps.
- **Forward Pass**: Selects the maximum value within regions of the feature map.

### **FullyConnectedLayer**
- **Purpose**: Connects flattened feature maps to output predictions.
- **Forward Pass**: Computes weighted sums and adds biases.

### **SoftmaxCrossEntropyLoss**
- **Purpose**: Computes the loss and gradients for classification tasks.

## **Results**

- **Training Loss Plot:**
  Visualizes loss trends over training epochs.

  ![Training Loss](images/loss_plot.png)

## **Future Enhancements**

1. **Support for Batch Processing**
2. **Addition of Advanced CNN Features** (e.g., Dropout, Batch Normalization)
3. **Performance Optimization**
4. **Integration with Other Datasets**

## **License**

This project is licensed under the MIT License. See `LICENSE` for details.

---

Feel free to contribute and explore enhancements!

