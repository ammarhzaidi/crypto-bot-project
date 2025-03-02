from strategies.candlestick_patterns.base_pattern import CandlestickPattern
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timezone
import pytz


class HammerPattern(CandlestickPattern):
    """Hammer candlestick pattern implementation."""

    def __init__(self, body_percent_max: float = 30.0,
                 lower_shadow_body_ratio_min: float = 2.0,
                 upper_shadow_body_ratio_max: float = 0.5):
        """
        Initialize hammer pattern detector with configurable parameters.

        Args:
            body_percent_max: Maximum body size as percentage of total range
            lower_shadow_body_ratio_min: Minimum ratio of lower shadow to body size
            upper_shadow_body_ratio_max: Maximum ratio of upper shadow to body size
        """
        self.body_percent_max = body_percent_max
        self.lower_shadow_body_ratio_min = lower_shadow_body_ratio_min
        self.upper_shadow_body_ratio_max = upper_shadow_body_ratio_max
        self.pakistan_tz = pytz.timezone('Asia/Karachi')

    @property
    def name(self) -> str:
        return "Hammer"

    def _calculate_candle_features(self, candle: pd.Series) -> Dict[str, float]:
        """
        Calculate features for a single candlestick.

        Args:
            candle: Pandas series with OHLCV data

        Returns:
            Dictionary with calculated candlestick features
        """
        open_price = float(candle['open'])
        high_price = float(candle['high'])
        low_price = float(candle['low'])
        close_price = float(candle['close'])

        # Calculate price ranges
        total_range = high_price - low_price
        if total_range == 0:  # Avoid division by zero
            total_range = 0.0001  # Small non-zero value

        body_size = abs(close_price - open_price)

        # Calculate body and shadow percentages of total range
        body_percent = (body_size / total_range) * 100

        # Upper and lower shadows
        if close_price >= open_price:  # Bullish candle
            upper_shadow = high_price - close_price
            lower_shadow = open_price - low_price
        else:  # Bearish candle
            upper_shadow = high_price - open_price
            lower_shadow = close_price - low_price

        upper_shadow_percent = (upper_shadow / total_range) * 100
        lower_shadow_percent = (lower_shadow / total_range) * 100

        # Determine if bullish or bearish
        is_bullish = close_price > open_price

        return {
            'body_size': body_size,
            'body_percent': body_percent,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'upper_shadow_percent': upper_shadow_percent,
            'lower_shadow_percent': lower_shadow_percent,
            'total_range': total_range,
            'is_bullish': is_bullish
        }

    def find_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find hammer candlestick patterns in the provided dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of dictionaries with hammer pattern details
        """
        hammers = []

        # Make sure we have the required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return []

        # Process each candle
        for i in range(len(df)):
            try:
                candle = df.iloc[i]

                # Skip candles with zero range (shouldn't happen in real data)
                if float(candle['high']) == float(candle['low']):
                    continue

                # Calculate candle features
                features = self._calculate_candle_features(candle)

                # Skip candles with very small body (avoid division by zero issues)
                if features['body_size'] < 0.0001:
                    continue

                # Check hammer criteria
                if (
                        features['body_percent'] <= self.body_percent_max and
                        features['lower_shadow'] >= features['body_size'] * self.lower_shadow_body_ratio_min and
                        features['upper_shadow'] <= features['body_size'] * self.upper_shadow_body_ratio_max
                ):
                    # Calculate pattern strength (0.0-1.0)
                    # Higher strength for smaller body percent and longer lower shadow
                    body_score = 1.0 - (features['body_percent'] / self.body_percent_max)
                    shadow_ratio = features['lower_shadow'] / features['body_size']
                    shadow_score = min(1.0, (shadow_ratio / (self.lower_shadow_body_ratio_min * 3)))

                    strength = (body_score + shadow_score) / 2

                    # If we have a timestamp, convert it to Pakistan time
                    timestamp_pst = None
                    if 'timestamp' in candle:
                        # Handle different timestamp formats
                        try:
                            if isinstance(candle['timestamp'], (int, float)):
                                # Unix timestamp in milliseconds
                                if candle['timestamp'] > 1e10:  # Likely milliseconds
                                    dt = datetime.fromtimestamp(candle['timestamp'] / 1000, tz=timezone.utc)
                                else:  # Likely seconds
                                    dt = datetime.fromtimestamp(candle['timestamp'], tz=timezone.utc)
                            else:
                                # Try parsing as ISO format
                                dt = pd.to_datetime(candle['timestamp'])
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)

                            # Convert to Pakistan time
                            timestamp_pst = dt.astimezone(self.pakistan_tz)

                        except (ValueError, TypeError):
                            pass

                    # Add candle to results
                    hammers.append({
                        'index': i,
                        'timestamp': timestamp_pst,
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle['volume']) if 'volume' in candle else None,
                        'is_bullish': features['is_bullish'],
                        'body_percent': features['body_percent'],
                        'lower_shadow_ratio': features['lower_shadow'] / features['body_size'],
                        'strength': strength,
                        'pattern_type': self.name
                    })
            except Exception:
                continue

        return hammers