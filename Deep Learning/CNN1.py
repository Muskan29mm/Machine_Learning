import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load Example Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize Pixel Values between 0 and 1
x_train, x_test = x_train/255.0, x_test/255.0

# Reshape Data to Include Channel Dimension 

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

model = models.Sequential([
    # Input Layer
    layers.Input(shape=(28, 28, 1)),
    # First Convolutional Layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Second Convolutional Layer: Extract Deeper Features
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # Third layer: Extract Even deeper features
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten Layer: Convert 2D feature maps to 1D feature vectors
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Output layer 
    layers.Dense(19, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

model.summary()


