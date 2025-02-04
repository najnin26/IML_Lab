import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' will suppress WARNING and INFO level messages
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # You can specify your GPU if needed (optional)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf

# Now, your imports and model code follow
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('/home/najnin/Desktop/IML Lab/CNN/mitbih_train.csv', header=None)
test = pd.read_csv('/home/najnin/Desktop/IML Lab/CNN/mitbih_test.csv', header=None)

X_train, y_train = train.iloc[:, :187].values, train.iloc[:, 187].values
X_test, y_test = test.iloc[:, :187].values, test.iloc[:, 187].values

# Reshape the input data to (samples, 187, 1)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)
print(X_test.shape)


CNN_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(187, 1)),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Flatten(),
    
    
    Dense(128, activation='relu'),  # Added L2 regularization
    Dense(5, activation='softmax')
])

print(CNN_model.summary())

from sklearn.utils import class_weight

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Compile the model with class weights
CNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model with class weights
history = CNN_model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=32,
    validation_data=(X_test, y_test),
   
    class_weight=class_weights  # Add class weights here
)

test_loss, test_accuracy = CNN_model.evaluate(X_test, y_test)

print(f"Test accuracy: {test_accuracy:.4f}")

from sklearn.metrics import classification_report
import numpy as np

y_pred_probs = CNN_model.predict(X_test)  # Get probabilities

# Get the predicted class labels
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class predictions

# Generate the classification report
print(classification_report(y_test, y_pred))

# Optional: Plot training history to visualize improvement
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

