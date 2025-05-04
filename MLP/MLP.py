## Importing necessary modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize image pixel values by dividing by 255 (grayscale)
gray_scale = 255


x_train = x_train.astype('float32') / gray_scale
x_test = x_test.astype('float32') / gray_scale


# Checking the shape of feature and target matrices
print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)


# Visualizing 100 images from the training data
fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
   for j in range(10):
       ax[i][j].imshow(x_train[k].reshape(28, 28), aspect='auto', cmap='gray')
       ax[i][j].axis('off')  # Hide axes for better visualization
       k += 1
plt.suptitle("Sample Images from MNIST Dataset", fontsize=16)
plt.show()


# Building the Sequential neural network model
model = Sequential([
   # Flatten input from 28x28 images to 784 (28*28) vector
   Flatten(input_shape=(28, 28)),
    # Dense layer 1 (256 neurons)
   Dense(256, activation='sigmoid'), 
    # Dense layer 2 (128 neurons)
   Dense(128, activation='sigmoid'),
    # Output layer (10 classes)
   Dense(10, activation='softmax'), 
])


# Compiling the model
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# Training the model with training data
history = model.fit(x_train, y_train, epochs=10,
                   batch_size=2000,
                   validation_split=0.2)


# Evaluating the model on test data
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)


# Visualization of Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 5))


# Plotting Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)


# Plotting Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)


plt.suptitle("Model Training Performance", fontsize=16)
plt.tight_layout()
plt.show()
