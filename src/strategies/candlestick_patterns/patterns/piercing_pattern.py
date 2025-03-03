from src.strategies.candlestick_patterns.base_pattern import CandlestickPattern
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timezone
import pytz


class PiercingPattern(CandlestickPattern):
    """Piercing candlestick pattern implementation."""

    def __init__(self, penetration_min: float = 0.5,
                 penetration_max: float = 1.0,
                 use_volume_confirmation: bool = False,
                 use_prior_trend: bool = False,
                 trend_periods: int = 5):
        """
        Initialize piercing pattern detector with configurable parameters.

        Args:
            penetration_min: Minimum penetration into previous candle's body (0.5 = 50%)
            penetration_max: Maximum penetration (1.0 = 100%, full engulfing not allowed)
            use_volume_confirmation: Whether to require higher volume on second candle
            use_prior_trend: Whether to require a prior downtrend
            trend_periods: Number of periods to check for prior trend
        """
        self.penetration_min = penetration_min
        self.penetration_max = penetration_max
        self.use_volume_confirmation = use_volume_confirmation
        self.use_prior_trend = use_prior_trend
        self.trend_periods = trend_periods
        self.pakistan_tz = pytz.timezone('Asia/Karachi')

    @property
    def name(self) -> str:
        return "Piercing"

    def find_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find piercing patterns in the provided dataframe.

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

        # Need at least 2 candles for piercing pattern
        if len(df) < 2:
            return []

        # Process each potential pattern (starting from second candle)
        for i in range(1, len(df)):
            try:
                # Get the current and previous candles
                current_candle = df.iloc[i]
                prev_candle = df.iloc[i - 1]

                # Check the core piercing pattern criteria
                # 1. Previous candle is bearish (close < open)
                # 2. Current candle is bullish (close > open)
                # 3. Current candle opens below previous candle's low
                # 4. Current candle closes above midpoint of previous candle's body
                # 5. Current candle doesn't close above previous candle's open (not engulfing)

                prev_open = float(prev_candle['open'])
                prev_close = float(prev_candle['close'])
                prev_low = float(prev_candle['low'])
                current_open = float(current_candle['open'])
                current_close = float(current_candle['close'])

                # Calculate previous candle's body size and midpoint
                prev_body_size = prev_open - prev_close  # For bearish candle, open > close
                prev_body_midpoint = prev_close + (prev_body_size / 2)

                # Calculate penetration percentage (how much into previous body)
                if prev_body_size > 0:  # Avoid division by zero
                    penetration = (current_close - prev_close) / prev_body_size
                else:
                    penetration = 0

                is_piercing = (
                        prev_close < prev_open and  # Previous candle is bearish
                        current_close > current_open and  # Current candle is bullish
                        current_open <= prev_low and  # Opens below previous low (or equal)
                        current_close > prev_body_midpoint and  # Closes above midpoint
                        current_close < prev_open and  # But doesn't close above previous open
                        penetration >= self.penetration_min and  # Minimum penetration
                        penetration <= self.penetration_max  # Maximum penetration
                )

                if prev_close < prev_open and current_close > current_open:
                    if hasattr(self, 'logger') and callable(self.logger):
                        self.logger(f"Potential piercing at index {i}: open gap={current_open <= prev_low}, " +
                                    f"midpoint penetration={current_close > prev_body_midpoint}, " +
                                    f"not full engulfing={current_close < prev_open}, " +
                                    f"penetration ratio={penetration:.2f}")

                if is_piercing:
                    # Calculate pattern strength (0-1)
                    # Based on penetration percentage and other factors
                    strength = (penetration - self.penetration_min) / (self.penetration_max - self.penetration_min)
                    strength = min(1.0, max(0.1, strength))  # Ensure between 0.1 and 1.0

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
                        # Check for prior downtrend
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
                        'is_bullish': True,  # Piercing is always bullish
                        'strength': strength,
                        'pattern_type': self.name,
                        'confirmations_passed': confirmations_passed,
                        'confirmations_failed': confirmations_failed,
                        'penetration': penetration
                    })
            except Exception as e:
                # Skip any errors in individual candle processing
                continue



        return patterns