import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


# Load dataset
dataset = pd.read_csv('/home/najnin/Downloads/IML Lab/MLP/AirPassengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset.columns = ['Passengers']


# Normalize dataset
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)


# Create dataset with variable look_back
def create_dataset(data, look_back=3):
   X, y = [], []
   for i in range(look_back, len(data)):
       X.append(data[i - look_back:i, 0])
       y.append(data[i, 0])
   return np.array(X), np.array(y)


# Function to build LSTM model
def build_model(hp):
   look_back = hp.Int('look_back', min_value=1, max_value=10, step=1)
   X_train, y_train = create_dataset(train, look_back)
   X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))


   model = Sequential()
   model.add(LSTM(
       units=hp.Choice('lstm_units', [32, 50, 64, 100]),
       input_shape=(look_back, 1),
       return_sequences=False
   ))
   model.add(Dropout(hp.Choice('dropout_rate', [0.0, 0.2, 0.4, 0.5])))
   model.add(Dense(1))


   model.compile(
       loss='mean_squared_error',
       optimizer=hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd']),
   )


   return model


# Prepare data
train_size = int(len(dataset_scaled) * 0.67)
train, test = dataset_scaled[:train_size], dataset_scaled[train_size:]


# Tuner
tuner = RandomSearch(
   build_model,
   objective='val_loss',
   max_trials=20,
   executions_per_trial=1,
   directory='lstm_tuning',
   project_name='air_passengers'
)


# Dummy call to get look_back first
X_tmp, y_tmp = create_dataset(train, 3)
X_tmp = X_tmp.reshape((X_tmp.shape[0], 3, 1))


tuner.search(
   X_tmp, y_tmp,
   epochs=10,
   batch_size=32,
   validation_split=0.2,
   verbose=1
)


# Get best hyperparameters
best_hp = tuner.get_best_hyperparameters(1)[0]
look_back = best_hp.get('look_back')


# Prepare final datasets
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)
X_train = X_train.reshape((X_train.shape[0], look_back, 1))
X_test = X_test.reshape((X_test.shape[0], look_back, 1))


# Build and train best model
model = tuner.hypermodel.build(best_hp)
model.summary()
history = model.fit(X_train, y_train, epochs=100, batch_size=best_hp.get('batch_size', 32), verbose=1)


# Evaluate
train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
print('Train score: {:.4f} MSE'.format(train_score))
print('Test score: {:.4f} MSE'.format(test_score))


# Predictions
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)
train_prediction = scaler.inverse_transform(train_prediction)
test_prediction = scaler.inverse_transform(test_prediction)
dataset_unscaled = scaler.inverse_transform(dataset_scaled)


# Plot results
plt.figure(figsize=(12, 8))
train_stamp = np.arange(look_back, look_back + len(train_prediction))
test_stamp = np.arange(len(train) + look_back*2, len(dataset_unscaled))


plt.plot(dataset_unscaled, label='True values')
plt.plot(train_stamp, train_prediction, label='Train prediction')
plt.plot(test_stamp, test_prediction, label='Test prediction')
plt.title('Tuned LSTM Time Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.tight_layout()
plt.show()
