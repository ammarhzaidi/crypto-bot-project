import numpy as np
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X, y, epochs=50, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test)
        )
        return history
