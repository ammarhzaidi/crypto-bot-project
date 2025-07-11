import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def prepare_features(self, df):
        features = self._create_technical_features(df)
        scaled_data = self.scaler.fit_transform(features)
        return self._create_sequences(scaled_data)

    def _create_technical_features(self, df):
        df['RSI'] = self._calculate_rsi(df['close'])
        df['MACD'] = self._calculate_macd(df['close'])
        return df
