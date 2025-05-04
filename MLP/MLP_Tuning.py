import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import keras_tuner as kt


# Load dataset
dataset = pd.read_csv('AirPassengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset.columns = ['Passengers']


# Create train/test split
train_size = int(dataset.shape[0] * 0.67)
train_df, test_df = dataset.iloc[:train_size, :], dataset.iloc[train_size:, :]


# Convert dataset to supervised format
def create_dataset(dataset, look_back=1):
   X, y = [], []
   for i in range(look_back, len(dataset)):
       X.append(dataset[i-look_back:i, 0])
       y.append(dataset[i, 0])
   return np.array(X), np.array(y)


look_back = 3
X_train, y_train = create_dataset(train_df.values, look_back=look_back)
X_test, y_test = create_dataset(test_df.values, look_back=look_back)


# Define model building function
def build_model(hp):
   model = Sequential()


   # Tune number of hidden layers
   for i in range(hp.Int('num_layers', 1, 3)):
       model.add(Dense(
           units=hp.Int(f'units_{i}', min_value=4, max_value=64, step=4),
           activation=hp.Choice('activation', ['relu', 'tanh', 'sigmoid']),
           input_dim=look_back if i == 0 else None
       ))


   model.add(Dense(1))  # output layer


   # Tune optimizer and learning rate
   optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
   lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')


   if optimizer_choice == 'adam':
       optimizer = Adam(learning_rate=lr)
   elif optimizer_choice == 'sgd':
       optimizer = SGD(learning_rate=lr)
   else:
       optimizer = RMSprop(learning_rate=lr)


   model.compile(optimizer=optimizer, loss='mean_squared_error')
   return model


# Configure tuner
tuner = kt.RandomSearch(
   build_model,
   objective='val_loss',
   max_trials=20,  # You can increase this for more thorough search
   executions_per_trial=1,
   directory='tuner_dir',
   project_name='full_hyperparam_tune'
)


# Search
tuner.search(
   X_train, y_train,
   validation_split=0.2,
   epochs=50,
   batch_size=kt.HyperParameters().Choice('batch_size', [2, 4, 8, 16]),
   verbose=1
)


# Save results
results = []
for trial in tuner.oracle.get_best_trials(num_trials=20):
   trial_result = {'trial_id': trial.trial_id, 'score': trial.score}
   trial_result.update(trial.hyperparameters.values)
   results.append(trial_result)


results_df = pd.DataFrame(results)
results_df.to_csv('all_hyperparam_results.csv', index=False)
print("All tuning results saved to 'all_hyperparam_results.csv'")
