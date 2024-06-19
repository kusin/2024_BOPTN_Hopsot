# libs manipulations array
import numpy as np

# lib neural network algorithms
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Bidirectional
# ----------------------------------------------------------------------------------------

# func model predictions
def get_lstm(timestep, activation, dropout, optimizer):

  # reset of session model
  tf.keras.backend.clear_session()

  # 1. The Neural Network Architecture
  model = Sequential()
  model.add(Bidirectional(LSTM(units=50, activation=activation, return_sequences=True, input_shape=(timestep, 1))))
  model.add(Bidirectional(LSTM(units=50, activation=activation, return_sequences=False)))
  model.add(Dropout(dropout))
  model.add(Dense(1))

  # 2. Compile models
  # model.compile(optimizer='adamax', loss='mean_squared_error')
  model.compile(
    optimizer=optimizer,
    loss="mae",
    metrics=[
      tf.keras.metrics.MeanAbsoluteError(),
      tf.keras.metrics.MeanSquaredError(),
      tf.keras.metrics.MeanAbsolutePercentageError(),
    ]
  )

  # return values
  return model
