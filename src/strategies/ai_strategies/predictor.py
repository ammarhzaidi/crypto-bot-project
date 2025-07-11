import numpy as np


class Predictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, market_data):
        processed_data = self.preprocessor.prepare_features(market_data)
        predictions = self.model.predict(processed_data)
        return self._generate_signals(predictions)
