import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist  # Import MNIST dataset from Keras
from tensorflow.keras.models import Sequential  # Import Sequential model type from Keras
from tensorflow.keras.layers import Dense, Flatten  # Import Dense and Flatten layer types from Keras
from tensorflow.keras.utils import to_categorical  # Import utility for one-hot encoding

# Load MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1 for better training performance
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the input images from 28x28 pixels to a 1D vector of 784 pixels
x_train_flat = x_train.reshape((-1, 28*28))
x_test_flat = x_test.reshape((-1, 28*28))

# Convert class vectors to binary class matrices (one-hot encoding)
# Example: class '3' becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model definition (Multilayer Perceptron)
model = Sequential()  # Initialize the model as a Sequential model

# Add input layer (Flatten layer to reshape the input)
model.add(Flatten(input_shape=(28*28,)))  # Flatten layer to convert 2D images to 1D vectors

# Add first hidden layer
# Dense layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu'))

# Add second hidden layer
# Dense layer with 64 units and ReLU activation
model.add(Dense(64, activation='relu'))

# Add output layer
# Dense layer with 10 units (one for each digit class 0-9)
# Activation function 'softmax' to output probability distribution over 10 classes
model.add(Dense(10, activation='softmax'))

# Compile model
# Loss function 'categorical_crossentropy' for multi-class classification
# Optimizer 'adam' for efficient stochastic gradient descent
# Metrics to monitor 'accuracy' during training and testing
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary to understand its architecture
model.summary()

# Train model
# Use training data (x_train_flat, y_train)
# Batch size of 64 for stochastic gradient descent
# Train for 30 epochs (iterations over the entire dataset)
# Verbose level 1 for progress bar display
# Validation data (x_test_flat, y_test) to evaluate the model on test data after each epoch
history = model.fit(x_train_flat, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(x_test_flat, y_test))

# Evaluate model
# Evaluate the trained model on the test dataset
# The evaluation returns loss and accuracy metrics
score = model.evaluate(x_test_flat, y_test, verbose=0)
print("Test loss:", score[0])  # Print the test loss
print("Test accuracy:", score[1])  # Print the test accuracy

# Save model
# Save the trained model to a file for later use
model.save("Handwritten-digit-recognition-1\model\model2.h5")
print("Model saved as 'model/model2.h5'.")
