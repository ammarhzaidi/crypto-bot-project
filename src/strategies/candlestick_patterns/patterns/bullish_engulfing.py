from ..base_pattern import CandlestickPattern
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timezone
import pytz


class BullishEngulfingPattern(CandlestickPattern):
    """Bullish Engulfing candlestick pattern implementation."""

    def __init__(self, use_volume_confirmation: bool = False,
                 use_prior_trend: bool = False,
                 use_size_significance: bool = False,
                 min_size_ratio: float = 1.5,
                 trend_periods: int = 5):
        """
        Initialize bullish engulfing pattern detector with configurable confirmations.

        Args:
            use_volume_confirmation: Whether to require higher volume on engulfing candle
            use_prior_trend: Whether to require a prior downtrend
            use_size_significance: Whether to require the engulfing candle to be significantly larger
            min_size_ratio: Minimum size ratio of engulfing candle to previous candle
            trend_periods: Number of periods to check for prior trend
        """
        self.use_volume_confirmation = use_volume_confirmation
        self.use_prior_trend = use_prior_trend
        self.use_size_significance = use_size_significance
        self.min_size_ratio = min_size_ratio
        self.trend_periods = trend_periods
        self.pakistan_tz = pytz.timezone('Asia/Karachi')

    @property
    def name(self) -> str:
        return "Bullish Engulfing"

    def find_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find bullish engulfing patterns in the provided dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of dictionaries with pattern details
        """
        patterns = []

        # Make sure we have the required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return []

        # Need at least 2 candles for engulfing pattern
        if len(df) < 2:
            return []

        # Process each potential pattern (starting from second candle)
        for i in range(1, len(df)):
            try:
                # Get the current and previous candles
                current_candle = df.iloc[i]
                prev_candle = df.iloc[i - 1]

                # Check the core engulfing pattern criteria
                # 1. Previous candle is bearish (close < open)
                # 2. Current candle is bullish (close > open)
                # 3. Current candle's open is below previous candle's close
                # 4. Current candle's close is above previous candle's open

                if (float(prev_candle['close']) < float(prev_candle['open']) and
                        float(current_candle['close']) > float(current_candle['open']) and
                        float(current_candle['open']) <= float(prev_candle['close']) and
                        float(current_candle['close']) >= float(prev_candle['open'])):

                    # Calculate base pattern strength (0-1)
                    # How much the current candle engulfs the previous one
                    prev_body_size = abs(float(prev_candle['close']) - float(prev_candle['open']))
                    curr_body_size = abs(float(current_candle['close']) - float(current_candle['open']))

                    # Avoid division by zero
                    if prev_body_size == 0:
                        prev_body_size = 0.0001

                    size_ratio = curr_body_size / prev_body_size
                    engulfing_score = min(1.0, size_ratio / 3.0)  # Cap at 1.0

                    # Initialize strength with base pattern strength
                    strength = engulfing_score
                    confirmations_passed = []
                    confirmations_failed = []

                    # Apply additional confirmations if requested
                    if self.use_volume_confirmation:
                        # Check if the current candle has higher volume
                        has_volume_confirmation = (
                                float(current_candle['volume']) > float(prev_candle['volume'])
                        )

                        if has_volume_confirmation:
                            strength += 0.1  # Boost strength for volume confirmation
                            confirmations_passed.append("volume")
                        else:
                            confirmations_failed.append("volume")

                    if self.use_prior_trend:
                        # Check for prior downtrend (at least trend_periods candles)
                        # Use a simple moving average comparison or consecutive lower closes
                        has_prior_trend = False

                        if i >= self.trend_periods + 1:
                            # Calculate average price change over the period
                            price_changes = []
                            for j in range(i - self.trend_periods, i):
                                change = float(df.iloc[j]['close']) - float(df.iloc[j - 1]['close'])
                                price_changes.append(change)

                            # If most changes are negative, consider it a downtrend
                            negative_changes = sum(1 for change in price_changes if change < 0)
                            has_prior_trend = negative_changes > (self.trend_periods // 2)

                        if has_prior_trend:
                            strength += 0.1  # Boost strength for prior trend
                            confirmations_passed.append("trend")
                        else:
                            confirmations_failed.append("trend")

                    if self.use_size_significance:
                        # Check if the engulfing candle is significantly larger
                        has_size_significance = size_ratio >= self.min_size_ratio

                        if has_size_significance:
                            strength += 0.1  # Boost strength for size significance
                            confirmations_passed.append("size")
                        else:
                            confirmations_failed.append("size")

                    # If any required confirmation failed, skip this pattern
                    required_confirmations = []
                    if self.use_volume_confirmation:
                        required_confirmations.append("volume")
                    if self.use_prior_trend:
                        required_confirmations.append("trend")
                    if self.use_size_significance:
                        required_confirmations.append("size")

                    if any(conf in confirmations_failed for conf in required_confirmations):
                        continue

                    # Normalize strength score to 0.0-1.0 range
                    strength = min(1.0, strength)

                    # Convert timestamp to Pakistan time
                    timestamp_pst = None
                    if 'timestamp' in current_candle:
                        # Handle different timestamp formats
                        try:
                            if isinstance(current_candle['timestamp'], (int, float)):
                                # Unix timestamp in milliseconds
                                if current_candle['timestamp'] > 1e10:  # Likely milliseconds
                                    dt = datetime.fromtimestamp(current_candle['timestamp'] / 1000, tz=timezone.utc)
                                else:  # Likely seconds
                                    dt = datetime.fromtimestamp(current_candle['timestamp'], tz=timezone.utc)
                            else:
                                # Try parsing as ISO format
                                dt = pd.to_datetime(current_candle['timestamp'])
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)

                            # Convert to Pakistan time
                            timestamp_pst = dt.astimezone(self.pakistan_tz)

                        except (ValueError, TypeError):
                            pass

                    # Add pattern to results
                    patterns.append({
                        'index': i,
                        'timestamp': timestamp_pst,
                        'open': float(current_candle['open']),
                        'high': float(current_candle['high']),
                        'low': float(current_candle['low']),
                        'close': float(current_candle['close']),
                        'volume': float(current_candle['volume']) if 'volume' in current_candle else None,
                        'is_bullish': True,  # Bullish Engulfing is always bullish
                        'strength': strength,
                        'pattern_type': self.name,
                        'confirmations_passed': confirmations_passed,
                        'confirmations_failed': confirmations_failed,
                        'size_ratio': size_ratio
                    })
            except Exception as e:
                # Skip any errors in individual candle processing
                continue

        return patterns