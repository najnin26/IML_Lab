import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# Load and preprocess dataset
df = pd.read_csv('/home/najnin/Downloads/IML Lab/LSTM/spam.csv', delimiter=',', encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)


X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
max_words = 1000
max_len = 150


tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)


# Build the model (renamed from RNN to model)
def build_model():
   inputs = Input(name='inputs', shape=[max_len])
   layer = Embedding(max_words, 50, input_length=max_len)(inputs)
   layer = LSTM(64)(layer)
   layer = Dense(256, name='FC1')(layer)
   layer = Activation('relu')(layer)
   layer = Dropout(0.5)(layer)
   layer = Dense(1, name='out_layer')(layer)
   layer = Activation('sigmoid')(layer)
   model = Model(inputs=inputs, outputs=layer)
   return model


model = build_model()
model.summary()


model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])


# Train model and save training history
history = model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,
                   validation_split=0.2,
                   callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])


# Evaluate model
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
accr = model.evaluate(test_sequences_matrix, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))


# Plot accuracy and loss
plt.figure(figsize=(12, 5))


# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()
