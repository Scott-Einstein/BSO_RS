import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, BatchNormalization, Dropout

def gru_model(input_shape, output_size):
    model = Sequential([
        GRU(3, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.1),
        TimeDistributed(Dense(10, activation='relu')),
        Dense(output_size, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model