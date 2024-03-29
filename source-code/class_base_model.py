# library algorithm lstm-rnn with keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import RNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.optimizers import Adam, Adamax, RMSprop, SGD
from keras.layers import LeakyReLU

# ----------------------------------------------------------------------------------------------------------
def lstm_algorithm(x_train, activation, dropout_rate, optimizer):

    # The Neural Network Architecture
    model = tf.keras.Sequential([
        
        # First layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            LSTM(units=50, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),
        
        # Secound layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            LSTM(units=50, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),

        # Third layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            LSTM(units=50, activation=activation, return_sequences=False)
        ),
        
        # Dropout layer
        tf.keras.layers.Dropout(dropout_rate),

        # The output layer
        tf.keras.layers.Dense(1)
    ])

    # Compile the model predictions
    model.compile(
        optimizer=optimizer,
        loss='mae',
        # metrics=[
        #     tf.keras.metrics.MeanAbsoluteError(),
        #     tf.keras.metrics.MeanSquaredError(),
        #     tf.keras.metrics.MeanAbsolutePercentageError(),
        # ]
    )

    # return values
    return model
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------
def gru_algorithm(x_train, activation, dropout_rate, optimizer):

    # The Neural Network Architecture
    model = tf.keras.Sequential([
        
        # First layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            GRU(units=50, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),
        
        # Secound layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            GRU(units=50, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),

        # Third layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            GRU(units=50, activation=activation, return_sequences=False)
        ),
        
        # Dropout layer
        tf.keras.layers.Dropout(dropout_rate),

        # The output layer
        tf.keras.layers.Dense(1)
    ])

    # Compile the model predictions
    model.compile(
        optimizer=optimizer,
        loss='mae',
        # metrics=[
        #     tf.keras.metrics.MeanAbsoluteError(),
        #     tf.keras.metrics.MeanSquaredError(),
        #     tf.keras.metrics.MeanAbsolutePercentageError(),
        # ]
    )

    # return values
    return model
# ----------------------------------------------------------------------------------------------------------
