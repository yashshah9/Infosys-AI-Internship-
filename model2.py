import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the input images
x_train_flat = x_train.reshape((-1, 28*28))
x_test_flat = x_test.reshape((-1, 28*28))

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model definition (Multilayer Perceptron)
model = Sequential()
model.add(Flatten(input_shape=(28*28,)))  # Adjust the input_shape to match the flattened input
model.add(Dense(128, activation='relu'))  # First hidden layer with 128 units and ReLU activation
model.add(Dense(64, activation='relu'))   # Second hidden layer with 64 units and ReLU activation
model.add(Dense(10, activation='softmax'))  # Output layer with softmax activation for multi-class classification

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Train model
history = model.fit(x_train_flat, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(x_test_flat, y_test))

# Evaluate model
score = model.evaluate(x_test_flat, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save model
model.save("Handwritten-digit-recognition-1\model\model2.h5")
print("Model saved as 'model/model2.h5'.")
