import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from .data_preprocessor import DataPreprocessor
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .model_evaluator import ModelEvaluator

# GPU setup and monitoring
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
    )

class GPUMonitor:
    @staticmethod
    def get_gpu_usage():
        return tf.config.experimental.get_memory_info('GPU:0')


class CryptoAITrader:
    def __init__(self):
        self.model = self.build_lstm_model()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer(self.model)
        self.predictor = Predictor(self.model, self.preprocessor)
        self.evaluator = ModelEvaluator()

    def build_lstm_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(60, 10)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate(self, market_data):
        X, y = self.preprocessor.prepare_features(market_data)
        history = self.trainer.train(X, y)
        metrics = self.evaluator.evaluate_model(self.model, X, y)
        return history, metrics

    def get_trading_signals(self, market_data):
        return self.predictor.predict(market_data)
