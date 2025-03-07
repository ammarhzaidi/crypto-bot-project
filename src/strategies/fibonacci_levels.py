"""
Fibonacci level analysis for crypto trading.
Implements retracement and extension calculations for price targets and support/resistance identification.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd


class FibonacciAnalyzer:
    """
    Analyzes price data to identify Fibonacci retracement and extension levels.
    """

    # Standard Fibonacci ratios
    RETRACEMENT_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENSION_LEVELS = [0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618, 3.618]

    def __init__(self, use_custom_levels: bool = False, custom_levels: Optional[List[float]] = None):
        """
        Initialize the Fibonacci analyzer.

        Args:
            use_custom_levels: Whether to use custom Fibonacci levels
            custom_levels: List of custom Fibonacci levels (if use_custom_levels is True)
        """
        self.use_custom_levels = use_custom_levels
        self.custom_levels = custom_levels if custom_levels else self.RETRACEMENT_LEVELS

    def calculate_retracement_levels(self, high_price: float, low_price: float) -> Dict[float, float]:
        """
        Calculate Fibonacci retracement levels for a given price range.

        Args:
            high_price: The highest price in the range
            low_price: The lowest price in the range

        Returns:
            Dictionary mapping Fibonacci ratios to price levels
        """
        price_range = high_price - low_price
        levels = {}

        for ratio in self.RETRACEMENT_LEVELS:
            if ratio != 0:  # Skip 0 level since it's same as the high
                retracement = high_price - (price_range * ratio)
                levels[ratio] = retracement
            else:
                levels[ratio] = high_price

        return levels

    def calculate_extension_levels(self, high_price: float, low_price: float,
                                   is_uptrend: bool = True) -> Dict[float, float]:
        """
        Calculate Fibonacci extension levels for a given price range.

        Args:
            high_price: The highest price in the range
            low_price: The lowest price in the range
            is_uptrend: Whether the market is in an uptrend (calculates extensions up)

        Returns:
            Dictionary mapping Fibonacci ratios to price levels
        """
        price_range = high_price - low_price
        levels = {}

        for ratio in self.EXTENSION_LEVELS:
            if is_uptrend:
                # Extensions above the high
                extension = high_price + (price_range * ratio)
            else:
                # Extensions below the low
                extension = low_price - (price_range * ratio)

            levels[ratio] = extension

        return levels

    def find_swing_points(self, prices: List[float], window_size: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find swing high and swing low points in a price series.

        Args:
            prices: List of price values
            window_size: Size of the window to look for local maxima/minima

        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        if len(prices) < window_size * 2:
            return [], []

        # Convert to numpy array for easier manipulation
        prices_array = np.array(prices)

        swing_highs = []
        swing_lows = []

        # Find swing highs and lows
        for i in range(window_size, len(prices_array) - window_size):
            # Window around the current point
            window = prices_array[i - window_size:i + window_size + 1]

            # Check if current point is a local maximum (swing high)
            if prices_array[i] == max(window):
                swing_highs.append(i)

            # Check if current point is a local minimum (swing low)
            if prices_array[i] == min(window):
                swing_lows.append(i)

        return swing_highs, swing_lows

    def identify_key_swings(self, prices: List[float], window_size: int = 5,
                            lookback: int = 20) -> Optional[Dict[str, Any]]:
        """
        Identify the most recent significant swing high and swing low.

        Args:
            prices: List of price values
            window_size: Size of the window to look for local maxima/minima
            lookback: Number of candles to look back for significant swings

        Returns:
            Dictionary with key swing information or None if not found
        """
        if len(prices) < lookback:
            return None

        # Get the recent price data
        recent_prices = prices[-lookback:]

        # Find all swing points in the recent data
        swing_high_indices, swing_low_indices = self.find_swing_points(recent_prices, window_size)

        if not swing_high_indices or not swing_low_indices:
            return None

        # Get the most recent swing high and low
        recent_high_idx = max(swing_high_indices)
        recent_low_idx = max(swing_low_indices)

        # Get the actual prices
        recent_high = recent_prices[recent_high_idx]
        recent_low = recent_prices[recent_low_idx]

        # Determine trend based on which came first
        if recent_high_idx > recent_low_idx:
            trend = "uptrend"
        else:
            trend = "downtrend"

        # Get the absolute indices (in the full price array)
        abs_high_idx = len(prices) - lookback + recent_high_idx
        abs_low_idx = len(prices) - lookback + recent_low_idx

        return {
            "trend": trend,
            "high_price": recent_high,
            "low_price": recent_low,
            "high_idx": abs_high_idx,
            "low_idx": abs_low_idx,
            "current_price": prices[-1]
        }

    def analyze_retracements(self, prices: List[float], window_size: int = 5,
                             lookback: int = 20) -> Optional[Dict[str, Any]]:
        """
        Analyze price data for Fibonacci retracement levels and price interactions.

        Args:
            prices: List of price values
            window_size: Size of the window to look for local maxima/minima
            lookback: Number of candles to look back for significant swings

        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        # Identify key swing points
        swing_data = self.identify_key_swings(prices, window_size, lookback)

        if not swing_data:
            return None

        # Calculate retracement levels
        if swing_data["trend"] == "uptrend":
            # In an uptrend, retracement is from low to high
            retracement_levels = self.calculate_retracement_levels(
                swing_data["high_price"], swing_data["low_price"]
            )
            # And extensions continue upward
            extension_levels = self.calculate_extension_levels(
                swing_data["high_price"], swing_data["low_price"], True
            )
        else:
            # In a downtrend, retracement is from high to low
            retracement_levels = self.calculate_retracement_levels(
                swing_data["low_price"], swing_data["high_price"]
            )
            # And extensions continue downward
            extension_levels = self.calculate_extension_levels(
                swing_data["high_price"], swing_data["low_price"], False
            )

        # Check where current price is relative to retracement levels
        current_price = swing_data["current_price"]

        # Find the closest retracement level
        closest_level = None
        closest_distance = float('inf')

        for ratio, price in retracement_levels.items():
            distance = abs(current_price - price)
            if distance < closest_distance:
                closest_distance = distance
                closest_level = ratio

        # Calculate percentage distance to closest level
        level_price = retracement_levels[closest_level]
        price_range = abs(swing_data["high_price"] - swing_data["low_price"])
        distance_percent = (closest_distance / price_range) * 100

        # Determine if price is at a key level (within 1% of price range)
        is_at_key_level = distance_percent < 1.0

        # Compile analysis results
        result = {
            "trend": swing_data["trend"],
            "high_price": swing_data["high_price"],
            "low_price": swing_data["low_price"],
            "current_price": current_price,
            "retracement_levels": retracement_levels,
            "extension_levels": extension_levels,
            "closest_level": closest_level,
            "closest_level_price": level_price,
            "distance_percent": distance_percent,
            "is_at_key_level": is_at_key_level
        }

        return result

    def get_trade_signals(self, prices: List[float], window_size: int = 5,
                          lookback: int = 20) -> Optional[Dict[str, Any]]:
        """
        Generate potential trade signals based on Fibonacci analysis.

        Args:
            prices: List of price values
            window_size: Size of the window to look for local maxima/minima
            lookback: Number of candles to look back for significant swings

        Returns:
            Dictionary with trade signal information or None if no signals
        """
        # Get Fibonacci analysis
        analysis = self.analyze_retracements(prices, window_size, lookback)

        if not analysis:
            return None

        # Initialize result
        signal = {
            "signal": None,
            "side": None,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": None,
            "strength": 0.0,
            "notes": []
        }

        # Current price
        current_price = analysis["current_price"]

        # Check if price is at a key Fibonacci level
        if analysis["is_at_key_level"]:
            level = analysis["closest_level"]

            # In an uptrend, potential buy signals occur at retracement levels
            if analysis["trend"] == "uptrend":
                # Strong retracement levels are 0.382, 0.5, 0.618
                if level in [0.382, 0.5, 0.618]:
                    signal["signal"] = "potential_buy"
                    signal["side"] = "BUY"
                    signal["entry"] = current_price

                    # Stop loss just below the swing low
                    signal["stop_loss"] = analysis["low_price"] * 0.99

                    # Take profit at extension levels
                    if level == 0.382:
                        # Moderate retracement, aim for 1.0 extension
                        tp_ratio = 1.0
                    elif level == 0.5:
                        # Deeper retracement, aim for 0.618 extension
                        tp_ratio = 0.618
                    else:  # 0.618
                        # Deep retracement, aim for 0.382 extension
                        tp_ratio = 0.382

                    signal["take_profit"] = analysis["extension_levels"][tp_ratio]

                    # Calculate risk-reward ratio
                    risk = current_price - signal["stop_loss"]
                    reward = signal["take_profit"] - current_price
                    signal["risk_reward_ratio"] = reward / risk if risk > 0 else 0

                    # Signal strength
                    if level == 0.618:
                        signal["strength"] = 0.8  # Strongest level
                        signal["notes"].append("Strong support at 0.618 Fibonacci level")
                    elif level == 0.5:
                        signal["strength"] = 0.7
                        signal["notes"].append("Good support at 0.5 Fibonacci level")
                    elif level == 0.382:
                        signal["strength"] = 0.6
                        signal["notes"].append("Moderate support at 0.382 Fibonacci level")

            # In a downtrend, potential sell signals occur at retracement levels
            else:
                # Strong retracement levels are 0.382, 0.5, 0.618
                if level in [0.382, 0.5, 0.618]:
                    signal["signal"] = "potential_sell"
                    signal["side"] = "SELL"
                    signal["entry"] = current_price

                    # Stop loss just above the swing high
                    signal["stop_loss"] = analysis["high_price"] * 1.01

                    # Take profit at extension levels
                    if level == 0.382:
                        # Moderate retracement, aim for 1.0 extension
                        tp_ratio = 1.0
                    elif level == 0.5:
                        # Deeper retracement, aim for 0.618 extension
                        tp_ratio = 0.618
                    else:  # 0.618
                        # Deep retracement, aim for 0.382 extension
                        tp_ratio = 0.382

                    signal["take_profit"] = analysis["extension_levels"][tp_ratio]

                    # Calculate risk-reward ratio
                    risk = signal["stop_loss"] - current_price
                    reward = current_price - signal["take_profit"]
                    signal["risk_reward_ratio"] = reward / risk if risk > 0 else 0

                    # Signal strength
                    if level == 0.618:
                        signal["strength"] = 0.8  # Strongest level
                        signal["notes"].append("Strong resistance at 0.618 Fibonacci level")
                    elif level == 0.5:
                        signal["strength"] = 0.7
                        signal["notes"].append("Good resistance at 0.5 Fibonacci level")
                    elif level == 0.382:
                        signal["strength"] = 0.6
                        signal["notes"].append("Moderate resistance at 0.382 Fibonacci level")

        # Check risk-reward ratio
        if signal["risk_reward_ratio"] is not None:
            if signal["risk_reward_ratio"] >= 3.0:
                signal["notes"].append(f"Excellent risk-reward ratio: {signal['risk_reward_ratio']:.2f}")
                signal["strength"] += 0.2
            elif signal["risk_reward_ratio"] >= 2.0:
                signal["notes"].append(f"Good risk-reward ratio: {signal['risk_reward_ratio']:.2f}")
                signal["strength"] += 0.1
            elif signal["risk_reward_ratio"] < 1.0:
                signal["notes"].append(f"Poor risk-reward ratio: {signal['risk_reward_ratio']:.2f}")
                signal["strength"] -= 0.3

        # If no signal was generated or risk-reward is too poor, return None
        if signal["signal"] is None or (signal["risk_reward_ratio"] is not None and signal["risk_reward_ratio"] < 1.0):
            return None

        return signal

    def get_price_targets(self, current_price: float, swing_high: float, swing_low: float,
                          is_long: bool = True) -> Dict[str, float]:
        """
        Calculate price targets based on Fibonacci levels.

        Args:
            current_price: Current price
            swing_high: Recent swing high price
            swing_low: Recent swing low price
            is_long: Whether this is a long (buy) trade

        Returns:
            Dictionary with price targets
        """
        price_range = swing_high - swing_low

        if is_long:
            # For long positions, calculate targets above the swing high
            targets = {
                "target_1": swing_high + (price_range * 0.382),  # Conservative target
                "target_2": swing_high + (price_range * 0.618),  # Moderate target
                "target_3": swing_high + (price_range * 1.0),  # Aggressive target
                "target_4": swing_high + (price_range * 1.618)  # Very aggressive target
            }

            # Calculate stop loss based on Fibonacci retracement
            targets["stop_loss"] = current_price - (current_price - swing_low) * 0.382
        else:
            # For short positions, calculate targets below the swing low
            targets = {
                "target_1": swing_low - (price_range * 0.382),  # Conservative target
                "target_2": swing_low - (price_range * 0.618),  # Moderate target
                "target_3": swing_low - (price_range * 1.0),  # Aggressive target
                "target_4": swing_low - (price_range * 1.618)  # Very aggressive target
            }

            # Calculate stop loss based on Fibonacci retracement
            targets["stop_loss"] = current_price + (swing_high - current_price) * 0.382

        return targets

    def combine_with_hh_hl_strategy(self, prices: List[float], hh_hl_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine Fibonacci analysis with HH/HL strategy for enhanced signals.

        Args:
            prices: List of price values
            hh_hl_signal: Signal from the HH/HL strategy

        Returns:
            Enhanced trade signal
        """
        # Get Fibonacci analysis
        fib_analysis = self.analyze_retracements(prices)

        if not fib_analysis or not hh_hl_signal:
            return hh_hl_signal

        # Enhanced signal starts with the HH/HL signal
        enhanced_signal = hh_hl_signal.copy()

        # Check for confluence with Fibonacci levels
        if fib_analysis["is_at_key_level"]:
            level = fib_analysis["closest_level"]

            # Check if HH/HL and Fibonacci signals align
            if (hh_hl_signal["side"] == "BUY" and fib_analysis["trend"] == "uptrend") or \
                    (hh_hl_signal["side"] == "SELL" and fib_analysis["trend"] == "downtrend"):

                # Strong confirmation from Fibonacci
                enhanced_signal["strength"] += 0.2
                enhanced_signal["notes"] = enhanced_signal.get("notes", [])
                enhanced_signal["notes"].append(
                    f"Confirmed by Fibonacci {level} level. Price at {fib_analysis['current_price']:.4f}"
                )

                # Add Fibonacci-based targets
                if hh_hl_signal["side"] == "BUY":
                    targets = self.get_price_targets(
                        fib_analysis["current_price"],
                        fib_analysis["high_price"],
                        fib_analysis["low_price"],
                        True
                    )
                else:
                    targets = self.get_price_targets(
                        fib_analysis["current_price"],
                        fib_analysis["high_price"],
                        fib_analysis["low_price"],
                        False
                    )

                enhanced_signal["fib_targets"] = targets

                # Consider adjusting stop loss based on Fibonacci
                if "stop_loss" in targets:
                    # Use the more conservative stop loss between HH/HL and Fibonacci
                    if hh_hl_signal["side"] == "BUY":
                        if targets["stop_loss"] > hh_hl_signal.get("sl", 0):
                            enhanced_signal["sl"] = targets["stop_loss"]
                            enhanced_signal["notes"].append("Using Fibonacci-based stop loss")
                    else:
                        if targets["stop_loss"] < hh_hl_signal.get("sl", float('inf')):
                            enhanced_signal["sl"] = targets["stop_loss"]
                            enhanced_signal["notes"].append("Using Fibonacci-based stop loss")

        return enhanced_signal


def find_fibonacci_zones(prices: List[float], window_size: int = 5, lookback: int = 50) -> Dict[str, Any]:
    """
    Identify Fibonacci retracement and extension zones for a price series.
    Standalone function for easy integration with existing code.

    Args:
        prices: List of price values
        window_size: Size of the window to look for local maxima/minima
        lookback: Number of candles to look back for significant swings

    Returns:
        Dictionary with Fibonacci zones information
    """
    analyzer = FibonacciAnalyzer()
    analysis = analyzer.analyze_retracements(prices, window_size, lookback)

    if not analysis:
        return {
            "has_zones": False,
            "zones": {},
            "current_price": prices[-1] if prices else None
        }

    # Extract retracement and extension levels
    zones = {
        "retracement": analysis["retracement_levels"],
        "extension": analysis["extension_levels"],
        "trend": analysis["trend"],
        "swing_high": analysis["high_price"],
        "swing_low": analysis["low_price"],
        "closest_level": analysis["closest_level"],
        "is_at_key_level": analysis["is_at_key_level"]
    }

    return {
        "has_zones": True,
        "zones": zones,
        "current_price": analysis["current_price"]
    }


def get_fibonacci_signal(prices: List[float]) -> Optional[Dict[str, Any]]:
    """
    Get trading signal based on Fibonacci analysis.
    Standalone function for easy integration with existing code.

    Args:
        prices: List of price values

    Returns:
        Trade signal dictionary or None if no signal
    """
    analyzer = FibonacciAnalyzer()
    return analyzer.get_trade_signals(prices)


# Example usage
if __name__ == "__main__":
    # Example price data (use with numpy and matplotlib for visualization)
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate sample price data (simulated uptrend)
    np.random.seed(42)
    x = np.arange(100)
    prices = 100 + 0.5 * x + 10 * np.sin(x / 10) + np.random.normal(0, 3, 100)

    # Create analyzer
    analyzer = FibonacciAnalyzer()

    # Get analysis
    analysis = analyzer.analyze_retracements(prices.tolist())

    if analysis:
        print(f"Trend: {analysis['trend']}")
        print(f"Current price: {analysis['current_price']:.2f}")
        print(f"Closest Fibonacci level: {analysis['closest_level']}")
        print(f"Is at key level: {analysis['is_at_key_level']}")

        print("\nRetracement levels:")
        for level, price in analysis['retracement_levels'].items():
            print(f"  {level}: {price:.2f}")

        print("\nExtension levels:")
        for level, price in analysis['extension_levels'].items():
            print(f"  {level}: {price:.2f}")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(prices)

        # Plot retracement levels
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        i = 0
        for level, price in analysis['retracement_levels'].items():
            plt.axhline(price, color=colors[i % len(colors)], linestyle='--',
                        label=f"Retracement {level}")
            i += 1

        plt.legend()
        plt.title("Price with Fibonacci Retracement Levels")
        plt.show()

        # Get trading signals
        signal = analyzer.get_trade_signals(prices.tolist())
        if signal:
            print("\nTrading Signal:")
            for key, value in signal.items():
                print(f"  {key}: {value}")