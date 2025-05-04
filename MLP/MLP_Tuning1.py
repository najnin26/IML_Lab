import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import keras_tuner as kt


# Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# Model building function for Keras Tuner
def build_model(hp):
   model = Sequential()
   model.add(Flatten(input_shape=(28, 28)))
  
   # Tune number of layers
   for i in range(hp.Int('num_layers', 1, 3)):  # 1 to 3 hidden layers
       model.add(Dense(
           units=hp.Int(f'units_{i}', min_value=64, max_value=512, step=64),
           activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])
       ))
  
   model.add(Dense(10, activation='softmax'))


   # Tune optimizer
   optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
   learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
   if optimizer_name == 'adam':
       optimizer = Adam(learning_rate=learning_rate)
   elif optimizer_name == 'rmsprop':
       optimizer = RMSprop(learning_rate=learning_rate)
   else:
       optimizer = SGD(learning_rate=learning_rate)


   model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   return model


# Hyperparameter tuner setup
tuner = kt.RandomSearch(
   build_model,
   objective='val_accuracy',
   max_trials=20,  # increase for more thorough tuning
   executions_per_trial=1,
   directory='mnist_tuning',
   project_name='mnist_hyperparam_search'
)


# Search for best hyperparameters
tuner.search(x_train, y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=kt.HyperParameters().Choice('batch_size', [64, 128, 256, 512]),
            verbose=1)


# Get best model
best_model = tuner.get_best_models(num_models=1)[0]


# Evaluate best model on test set
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)


# Visualize best model training
history = best_model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=10,
                        batch_size=128,  # or tuner.get_best_hyperparameters()[0].get('batch_size')
                        verbose=0)


# Plot accuracy and loss
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
