import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch


# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'


# Load dataset
train = pd.read_csv('/home/najnin/Downloads/IML Lab/CNN/mitbih_train.csv', header=None)
test = pd.read_csv('/home/najnin/Downloads/IML Lab/CNN/mitbih_test.csv', header=None)


# Split into features and labels
X_train, y_train = train.iloc[:, :187].values, train.iloc[:, 187].values
X_test, y_test = test.iloc[:, :187].values, test.iloc[:, 187].values


# Reshape for Conv1D input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))


# Define the model builder function
def build_model(hp):
   model = Sequential()
   model.add(Conv1D(
       filters=hp.Choice('filters_1', [32, 64, 128]),
       kernel_size=hp.Choice('kernel_size_1', [3, 5, 7]),
       activation='relu',
       input_shape=(187, 1)
   ))
   model.add(MaxPooling1D(pool_size=2))


   model.add(Conv1D(
       filters=hp.Choice('filters_2', [64, 128]),
       kernel_size=hp.Choice('kernel_size_2', [3, 5]),
       activation='relu'
   ))
   model.add(MaxPooling1D(pool_size=2))
   model.add(Flatten())


   model.add(Dense(
       units=hp.Choice('dense_units', [64, 128, 256]),
       activation='relu'
   ))
   model.add(Dropout(hp.Choice('dropout_rate', [0.1, 0.25, 0.5])))
   model.add(Dense(5, activation='softmax'))


   model.compile(
       optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )


   return model


# Initialize the tuner
tuner = RandomSearch(
   build_model,
   objective='val_accuracy',
   max_trials=15,
   executions_per_trial=1,
   directory='cnn_tuning',
   project_name='ecg'
)


# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), class_weight=class_weights)


# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]


# Evaluate the best model
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# Fit the best model again to plot history
history = best_model.fit(
   X_train, y_train,
   validation_split=0.2,
   epochs=10,
   batch_size=128,
   verbose=0
)


# Plot training accuracy and loss
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.suptitle("Tuned Model Training Performance", fontsize=16)
plt.tight_layout()
plt.show()


# Print classification report
from sklearn.metrics import classification_report


y_pred_probs = best_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
