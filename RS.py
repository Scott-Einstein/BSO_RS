from GRU import gru_model
import tensorflow as tf

class RS:
    def __init__(self, X_train=None, y_train=None, path=None):
        if path is None:
            self.X_train = X_train
            self.y_train = y_train
            self.input_shape = X_train.shape[1:]
            self.output_size = y_train.shape[-1]
            self.model = gru_model(self.input_shape, self.output_size)
        else:
            self.model = tf.keras.models.load_model(path)
        self.model.summary()

    def fit(self, epochs=200, verbose=1, batch_size=20):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def save(self, path):
        self.model.save(path)