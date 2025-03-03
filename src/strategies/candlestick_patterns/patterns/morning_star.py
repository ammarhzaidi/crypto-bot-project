from src.strategies.candlestick_patterns.base_pattern import CandlestickPattern
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timezone
import pytz


class MorningStarPattern(CandlestickPattern):
    """Morning Star candlestick pattern implementation."""

    def __init__(self,
                 gap_percent_min: float = 0.1,
                 third_candle_penetration_min: float = 0.1,
                 middle_body_max_percent: float = 50.0,
                 use_volume_confirmation: bool = False,
                 use_prior_trend: bool = False,
                 trend_periods: int = 5):
        """
        Initialize Morning Star pattern detector with configurable parameters.

        Args:
            gap_percent_min: Minimum gap between first and second candle as percentage
            third_candle_penetration_min: Minimum penetration of third candle into first candle's body (0.3 = 30%)
            middle_body_max_percent: Maximum body size of the middle candle as percentage of its total range
            use_volume_confirmation: Whether to require higher volume on third candle
            use_prior_trend: Whether to require a prior downtrend
            trend_periods: Number of periods to check for prior trend
        """
        self.gap_percent_min = gap_percent_min
        self.third_candle_penetration_min = third_candle_penetration_min
        self.middle_body_max_percent = middle_body_max_percent
        self.use_volume_confirmation = use_volume_confirmation
        self.use_prior_trend = use_prior_trend
        self.trend_periods = trend_periods
        self.pakistan_tz = pytz.timezone('Asia/Karachi')

    @property
    def name(self) -> str:
        return "Morning Star"

    def find_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:

        if len(df) > 0:
            self.logger(
                f"First candlestick data: Open={df.iloc[0]['open']}, High={df.iloc[0]['high']}, Low={df.iloc[0]['low']}, Close={df.iloc[0]['close']}")
        """
        Find Morning Star patterns in the provided dataframe.

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

        # Need at least 3 candles for the Morning Star pattern
        if len(df) < 3:
            return []

        # Process each potential pattern (starting from the third candle)
        for i in range(2, len(df)):
            try:
                # Get the three candles that make up the potential pattern
                first_candle = df.iloc[i - 2]  # First candle (bearish)
                middle_candle = df.iloc[i - 1]  # Middle candle (small body)
                third_candle = df.iloc[i]  # Third candle (bullish)

                # Extract price data
                first_open = float(first_candle['open'])
                first_close = float(first_candle['close'])
                first_high = float(first_candle['high'])
                first_low = float(first_candle['low'])

                middle_open = float(middle_candle['open'])
                middle_close = float(middle_candle['close'])
                middle_high = float(middle_candle['high'])
                middle_low = float(middle_candle['low'])

                third_open = float(third_candle['open'])
                third_close = float(third_candle['close'])
                third_high = float(third_candle['high'])
                third_low = float(third_candle['low'])

                # Calculate body sizes
                first_body_size = abs(first_open - first_close)
                middle_body_size = abs(middle_open - middle_close)
                third_body_size = abs(third_open - third_close)

                # Calculate total ranges (high to low)
                first_range = first_high - first_low
                middle_range = middle_high - middle_low
                third_range = third_high - third_low

                # 1. Check if first candle is bearish (close < open)
                is_first_bearish = first_close < first_open

                # 2. Check if middle candle has small body
                # Small body is defined as a percentage of its total range
                if middle_range == 0:  # Avoid division by zero
                    middle_body_percent = 0
                else:
                    middle_body_percent = (middle_body_size / middle_range) * 100

                is_middle_small_body = middle_body_percent <= self.middle_body_max_percent

                # 3. Check if third candle is bullish (close > open)
                is_third_bullish = third_close > third_open

                # 4. Check for gap down between first and middle candle
                # Gap is defined as the difference between first candle's close and middle candle's high
                gap_down = first_close > middle_high  # True gap down

                # Alternative: Check if there's a "body gap" (more commonly used)
                body_gap_down = False
                if is_first_bearish:  # First candle is bearish
                    body_gap_down = first_close > middle_open and first_close > middle_close
                else:  # First candle is bullish (not ideal for Morning Star)
                    body_gap_down = first_open > middle_open and first_open > middle_close

                # Use either strict gap down or body gap down
                has_gap_down = gap_down or body_gap_down

                # 5. Check if third candle closes well into first candle's body
                # Calculate penetration into first candle's body
                if is_first_bearish:
                    # For bearish first candle
                    first_body_top = first_open
                    first_body_bottom = first_close
                else:
                    # For bullish first candle (not ideal for Morning Star)
                    first_body_top = first_close
                    first_body_bottom = first_open

                first_body_range = first_body_top - first_body_bottom

                if first_body_range > 0:  # Avoid division by zero
                    penetration = (third_close - first_body_bottom) / first_body_range
                else:
                    penetration = 0

                has_sufficient_penetration = penetration >= self.third_candle_penetration_min

                # Combine all conditions for a Morning Star pattern
                is_morning_star = (
                        is_first_bearish and
                        is_middle_small_body and
                        is_third_bullish and
                        has_gap_down and
                        has_sufficient_penetration
                )

                if is_morning_star:
                    # Calculate pattern strength (0-1)
                    # Based on various factors:
                    # - Penetration of third candle into first candle's body
                    # - Size of the middle candle (smaller is better)
                    # - Gap size

                    # Penetration score (0.3-1.0)
                    penetration_score = min(1.0, max(0.3, penetration))

                    # Middle candle size score (smaller is better)
                    middle_size_score = 1.0 - (middle_body_percent / self.middle_body_max_percent)

                    # Gap score
                    if first_body_range > 0:
                        gap_percent = (first_close - max(middle_high, middle_close, middle_open)) / first_body_range
                        gap_score = min(1.0, gap_percent / self.gap_percent_min)
                    else:
                        gap_score = 0.5  # Default if there's no proper first body range

                    # Average the scores
                    strength = (penetration_score * 0.5) + (middle_size_score * 0.3) + (gap_score * 0.2)
                    strength = min(1.0, max(0.1, strength))  # Ensure between 0.1 and 1.0

                    confirmations_passed = []
                    confirmations_failed = []

                    # Apply additional confirmations if requested
                    if self.use_volume_confirmation:
                        # Check if the third candle has higher volume than first candle
                        has_volume_confirmation = (
                                float(third_candle['volume']) > float(first_candle['volume'])
                        )

                        if has_volume_confirmation:
                            strength += 0.1  # Boost strength for volume confirmation
                            confirmations_passed.append("volume")
                        else:
                            confirmations_failed.append("volume")

                    if self.use_prior_trend:
                        # Check for prior downtrend before the pattern
                        has_prior_trend = False

                        if i >= self.trend_periods + 2:
                            # Calculate average price change over the period before the pattern
                            price_changes = []
                            for j in range(i - 2 - self.trend_periods, i - 2):
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

                    # If any required confirmation failed, skip this pattern
                    required_confirmations = []
                    if self.use_volume_confirmation:
                        required_confirmations.append("volume")
                    if self.use_prior_trend:
                        required_confirmations.append("trend")

                    if any(conf in confirmations_failed for conf in required_confirmations):
                        continue

                    # Normalize strength score to 0.0-1.0 range
                    strength = min(1.0, strength)

                    # Convert timestamp to Pakistan time
                    timestamp_pst = None
                    if 'timestamp' in third_candle:
                        # Handle different timestamp formats
                        try:
                            if isinstance(third_candle['timestamp'], (int, float)):
                                # Unix timestamp in milliseconds
                                if third_candle['timestamp'] > 1e10:  # Likely milliseconds
                                    dt = datetime.fromtimestamp(third_candle['timestamp'] / 1000, tz=timezone.utc)
                                else:  # Likely seconds
                                    dt = datetime.fromtimestamp(third_candle['timestamp'], tz=timezone.utc)
                            else:
                                # Try parsing as ISO format
                                dt = pd.to_datetime(third_candle['timestamp'])
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
                        'open': float(third_candle['open']),
                        'high': float(third_candle['high']),
                        'low': float(third_candle['low']),
                        'close': float(third_candle['close']),
                        'volume': float(third_candle['volume']) if 'volume' in third_candle else None,
                        'is_bullish': True,  # Morning Star is a bullish pattern
                        'strength': strength,
                        'pattern_type': self.name,
                        'confirmations_passed': confirmations_passed,
                        'confirmations_failed': confirmations_failed,
                        'penetration': penetration,
                        'middle_body_percent': middle_body_percent
                    })
            except Exception as e:
                # Skip any errors in individual candle processing
                continue

        if is_first_bearish:
            self.logger(f"First candle is bearish at index {i - 2}")
        else:
            self.logger(f"Failed: First candle not bearish at index {i - 2}")

        if is_middle_small_body:
            self.logger(f"Middle candle has small body ({middle_body_percent:.2f}%) at index {i - 1}")
        else:
            self.logger(f"Failed: Middle body too large ({middle_body_percent:.2f}%) at index {i - 1}")

        return patterns