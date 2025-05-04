import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Load dataset
dataset = pd.read_csv('/home/najnin/Downloads/IML Lab/MLP/AirPassengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset.columns = ['Passengers']


# Normalize the dataset
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)


# Split the dataset
train_size = int(len(dataset_scaled) * 0.67)
train, test = dataset_scaled[:train_size], dataset_scaled[train_size:]


# Create dataset with look_back
def create_dataset(data, look_back=3):
   X, y = [], []
   for i in range(look_back, len(data)):
       X.append(data[i - look_back:i, 0])
       y.append(data[i, 0])
   return np.array(X), np.array(y)


look_back = 3
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)


# Reshape to [samples, time steps, features] for LSTM
X_train = X_train.reshape((X_train.shape[0], look_back, 1))
X_test = X_test.reshape((X_test.shape[0], look_back, 1))


# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2)


# Evaluate the model
train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
print('Train score: {:.4f} MSE'.format(train_score))
print('Test score: {:.4f} MSE'.format(test_score))


# Make predictions
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)


# Inverse transform predictions and true values
train_prediction = scaler.inverse_transform(train_prediction)
test_prediction = scaler.inverse_transform(test_prediction)
dataset_unscaled = scaler.inverse_transform(dataset_scaled)


# Plot predictions
plt.figure(figsize=(12, 8))
train_stamp = np.arange(look_back, look_back + len(train_prediction))
test_stamp = np.arange(look_back*2 + len(train_prediction), len(dataset_unscaled))
plt.plot(dataset_unscaled, label='True values')
plt.plot(train_stamp, train_prediction, label='Train prediction')
plt.plot(test_stamp, test_prediction, label='Test prediction')
plt.title('LSTM-based Time Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.show()
