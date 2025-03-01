"""
Candlestick pattern detection module for cryptocurrency analysis.
This module provides functions to detect various candlestick patterns
in OHLCV (Open, High, Low, Close, Volume) data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timezone
import pytz


class CandlestickPatternFinder:
    """
    Detects various candlestick patterns in price data.
    Currently supported patterns:
    - Hammer
    """

    def __init__(self, logger=None):
        """
        Initialize the candlestick pattern finder.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger('candlestick_finder')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

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

    def find_hammers(self, df: pd.DataFrame, body_percent_max: float = 30.0,
                     lower_shadow_body_ratio_min: float = 2.0,
                     upper_shadow_body_ratio_max: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find hammer candlestick patterns in the provided dataframe.

        A hammer has:
        - Small body (typically less than 30% of total range)
        - Long lower shadow (at least 2x the body size)
        - Little to no upper shadow (less than 0.5x the body size)

        Args:
            df: DataFrame with OHLCV data
            body_percent_max: Maximum body size as percentage of total range
            lower_shadow_body_ratio_min: Minimum ratio of lower shadow to body size
            upper_shadow_body_ratio_max: Maximum ratio of upper shadow to body size

        Returns:
            List of dictionaries with hammer pattern details
        """
        self.logger.info(f"Searching for hammer patterns in {len(df)} candles")

        # Make sure we have the required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            self.logger.error(f"Available columns: {df.columns.tolist()}")
            return []

        # List to store results
        hammers = []

        # Convert UTC timestamps to Pakistan Standard Time
        pakistan_tz = pytz.timezone('Asia/Karachi')

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
                        features['body_percent'] <= body_percent_max and
                        features['lower_shadow'] >= features['body_size'] * lower_shadow_body_ratio_min and
                        features['upper_shadow'] <= features['body_size'] * upper_shadow_body_ratio_max
                ):
                    # Calculate pattern strength (0.0-1.0)
                    # Higher strength for smaller body percent and longer lower shadow
                    body_score = 1.0 - (features['body_percent'] / body_percent_max)
                    shadow_ratio = features['lower_shadow'] / features['body_size']
                    shadow_score = min(1.0, (shadow_ratio / (lower_shadow_body_ratio_min * 3)))

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
                            timestamp_pst = dt.astimezone(pakistan_tz)

                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Could not parse timestamp: {candle['timestamp']}, error: {str(e)}")

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
                        'strength': strength
                    })

                    self.logger.info(f"Found hammer pattern at index {i}, timestamp: {timestamp_pst}")
            except Exception as e:
                self.logger.error(f"Error processing candle at index {i}: {str(e)}")
                continue

        self.logger.info(f"Found {len(hammers)} hammer patterns")
        return hammers


def format_hammer_results(hammers: List[Dict[str, Any]]) -> str:
    """
    Format hammer results for terminal display.

    Args:
        hammers: List of hammer pattern dictionaries

    Returns:
        Formatted string for terminal display
    """
    if not hammers:
        return "No hammer patterns found."

    # Sort hammers by timestamp (newest first)
    sorted_hammers = sorted(hammers, key=lambda x: x['timestamp'] if x['timestamp'] else 0, reverse=True)

    # Create the header
    header = f"{'Index':<8} {'Timestamp (PST)':<20} {'Price':<10} {'Body %':<8} {'LS Ratio':<8} {'Bullish':<8} {'Strength':<8}"
    divider = "-" * len(header)

    # Create the result string
    result = [header, divider]

    for hammer in sorted_hammers:
        timestamp_str = hammer['timestamp'].strftime('%Y-%m-%d %H:%M') if hammer['timestamp'] else 'N/A'
        price_str = f"{hammer['close']:.4f}"
        body_percent_str = f"{hammer['body_percent']:.1f}%"
        lower_shadow_ratio_str = f"{hammer['lower_shadow_ratio']:.1f}x"
        is_bullish_str = "Yes" if hammer['is_bullish'] else "No"
        strength_str = f"{hammer['strength']:.2f}"

        line = f"{hammer['index']:<8} {timestamp_str:<20} {price_str:<10} {body_percent_str:<8} " \
               f"{lower_shadow_ratio_str:<8} {is_bullish_str:<8} {strength_str:<8}"
        result.append(line)

    return "\n".join(result)




