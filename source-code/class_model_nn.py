# library algorithm lstm-rnn with keras
import tensorflow as tf
from keras.layers import LSTM
from keras.layers import GRU

# ----------------------------------------------------------------------------------------------------------
def lstm_algorithm(x_train, activation, Dropout):

    # The Neural Network Architecture
    model = tf.keras.Sequential([
        
        # First layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            LSTM(units=10, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),
        
        # Secound layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            LSTM(units=10, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),

        # Third layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            LSTM(units=10, activation=activation, return_sequences=False)
        ),
        
        # Dropout layer
        tf.keras.layers.Dropout(Dropout),

        # The output layer
        tf.keras.layers.Dense(1)
    ])

    # return values
    return model
# ----------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------
def gru_algorithm(x_train, activation, Dropout):

    # The Neural Network Architecture
    model = tf.keras.Sequential([
        
        # First layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            GRU(units=10, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),
        
        # Secound layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            GRU(units=10, activation=activation, return_sequences=True, input_shape=(x_train.shape[1], 1))
        ),

        # Third layer with Dropout regularisation
        tf.keras.layers.Bidirectional(
            GRU(units=10, activation=activation, return_sequences=False)
        ),
        
        # Dropout layer
        tf.keras.layers.Dropout(Dropout),

        # The output layer
        tf.keras.layers.Dense(1)
    ])

    # return values
    return model
# ----------------------------------------------------------------------------------------------------------