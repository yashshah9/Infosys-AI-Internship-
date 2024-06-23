import numpy as np  # Import numpy library for numerical operations with arrays
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import tensorflow as tf  # Import TensorFlow for deep learning operations
import shutil  # Import shutil for high-level file operations

from tensorflow.keras.utils import to_categorical  # Import to_categorical for one-hot encoding
from tensorflow.keras.datasets import mnist  # Import mnist dataset from Keras
from tensorflow.keras.models import Sequential  # Import Sequential model from Keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, Flatten  # Import layers needed for the model architecture

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to fit LeNet-5 architecture (input shape: 28x28)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model definition (LeNet-5 architecture)
model = Sequential()

# Layer 1: Convolutional Layer
# Input: 28x28x1 image
# Filters: 6 filters of size 5x5
# Strides: 1x1
# Activation: ReLU
# Output: 24x24x6 (valid padding, so the size is reduced)
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))

# Layer 2: Average Pooling Layer
# Pooling size: 2x2
# Strides: 2x2
# Output: 12x12x6 (downsampled by a factor of 2)
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 3: Convolutional Layer
# Filters: 16 filters of size 5x5
# Strides: 1x1
# Activation: ReLU
# Output: 8x8x16 (valid padding)
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

# Layer 4: Average Pooling Layer
# Pooling size: 2x2
# Strides: 2x2
# Output: 4x4x16 (downsampled by a factor of 2)
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer 5: Flatten Layer
# Converts the 3D output (4x4x16) to 1D
# Output: 256
model.add(Flatten())

# Layer 6: Fully Connected (Dense) Layer
# Units: 120
# Activation: ReLU
# Output: 120
model.add(Dense(120, activation='relu'))

# Layer 7: Fully Connected (Dense) Layer
# Units: 84
# Activation: ReLU
# Output: 84
model.add(Dense(84, activation='relu'))

# Layer 8: Fully Connected (Dense) Layer
# Units: 10 (number of classes for classification)
# Activation: Softmax (for classification)
# Output: 10 (class probabilities)
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Train model
history = model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(x_test, y_test))

# Evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save model
model.save("Handwritten-digit-recognition-1\\model\\lenet_model.h5")
print("Model saved as 'lenet5_mnist.h5'.")
