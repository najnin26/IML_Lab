import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input


# Load dataset
dataset = pd.read_csv('/home/najnin/Downloads/IML Lab/MLP/AirPassengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset.columns = ['Passengers']


# Normalize the dataset
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)


# Split the dataset
train_size = int(dataset_scaled.shape[0] * 0.67)
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


# Reshape input to be [samples, time steps, features] for Conv1D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# Build CNN model
model = Sequential([
   Input(shape=(look_back, 1)),
   Conv1D(filters=64, kernel_size=2, activation='relu'),
   Flatten(),
   Dense(50, activation='relu'),
   Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2)


# Evaluate
train_score = model.evaluate(X_train, y_train)
test_score = model.evaluate(X_test, y_test)
print('Train score: {:.4f} MSE'.format(train_score))
print('Test score: {:.4f} MSE'.format(test_score))


# Predict
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)


# Inverse transform predictions to original scale
train_prediction = scaler.inverse_transform(train_prediction)
test_prediction = scaler.inverse_transform(test_prediction)
dataset_unscaled = scaler.inverse_transform(dataset_scaled)


# Plot
plt.figure(figsize=(12, 8))
train_stamp = np.arange(look_back, look_back + len(train_prediction))
test_stamp = np.arange(look_back*2 + len(train_prediction), len(dataset_unscaled))
plt.plot(dataset_unscaled, label='True values')
plt.plot(train_stamp, train_prediction, label='Train prediction')
plt.plot(test_stamp, test_prediction, label='Test prediction')
plt.title('CNN-based Time Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.show()
