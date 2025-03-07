"""
Fibonacci analysis integration with existing trading strategies.
Demonstrates how to combine Fibonacci levels with HH/HL and candlestick pattern strategies.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.strategies.hh_hl_strategy import analyze_price_action
from src.strategies.fibonacci_levels import FibonacciAnalyzer, find_fibonacci_zones
from src.risk_management.position_sizer import calculate_take_profit, calculate_stop_loss


def enhanced_hh_hl_with_fibonacci(prices: List[float],
                                  smoothing: int = 1,
                                  consecutive_count: int = 2) -> Dict[str, Any]:
    """
    Enhanced HH/HL strategy that integrates Fibonacci levels for better entries and targets.

    Args:
        prices: List of price values
        smoothing: Smoothing parameter for HH/HL detection
        consecutive_count: Number of consecutive patterns required for HH/HL

    Returns:
        Dictionary with analysis results and signals
    """
    # Step 1: Run the standard HH/HL analysis
    hh_hl_result = analyze_price_action(prices, smoothing, consecutive_count)

    # Step 2: Get Fibonacci zones and analysis
    fib_zones = find_fibonacci_zones(prices)

    # Step 3: Create the enhanced result structure
    enhanced_result = {
        "trend": hh_hl_result["trend"],
        "uptrend_analysis": hh_hl_result["uptrend_analysis"],
        "downtrend_analysis": hh_hl_result["downtrend_analysis"],
        "fibonacci_zones": fib_zones["zones"] if fib_zones["has_zones"] else {},
        "has_fib_confluence": False,
        "signals": []
    }

    # Step 4: Check for confluence between HH/HL and Fibonacci
    if fib_zones["has_zones"]:
        # Check if both analyses agree on the trend
        hh_hl_trend = hh_hl_result["trend"]
        fib_trend = fib_zones["zones"]["trend"]

        trend_match = (
                (hh_hl_trend == "uptrend" and fib_trend == "uptrend") or
                (hh_hl_trend == "downtrend" and fib_trend == "downtrend")
        )

        # Check if price is at a key Fibonacci level
        at_key_level = fib_zones["zones"]["is_at_key_level"]

        # Determine confluence
        enhanced_result["has_fib_confluence"] = trend_match and at_key_level

        # Generate signals based on confluence
        if enhanced_result["has_fib_confluence"]:
            current_price = fib_zones["current_price"]

            # Create signal details
            if hh_hl_trend == "uptrend":
                # In uptrend, we're looking for buy signals
                signal = {
                    "side": "BUY",
                    "entry_price": current_price,
                    "pattern": f"HH/HL with Fibonacci {fib_zones['zones']['closest_level']} retracement",
                    "strength": min(1.0, 0.5 + 0.3 * hh_hl_result["uptrend_analysis"]["consecutive_hh"]),
                    "fibonacci_level": fib_zones["zones"]["closest_level"]
                }

                # Calculate standard TP/SL
                tp = calculate_take_profit(current_price, 1.0)  # 1% default
                sl = calculate_stop_loss(current_price, 1.0)  # 1% default

                # Enhance with Fibonacci-based targets if possible
                fib_retracement = fib_zones["zones"]["retracement"]
                fib_extension = fib_zones["zones"]["extension"]

                # Use 0.618 retracement as stop loss if available
                if 0.618 in fib_retracement and fib_retracement[0.618] < current_price:
                    sl = fib_retracement[0.618]

                # Use 1.618 extension as take profit if available
                if 1.618 in fib_extension:
                    tp = fib_extension[1.618]

                signal["tp"] = tp
                signal["sl"] = sl

                # Calculate risk/reward ratio
                risk = current_price - sl
                reward = tp - current_price
                signal["risk_reward"] = reward / risk if risk > 0 else 0

                enhanced_result["signals"].append(signal)

            elif hh_hl_trend == "downtrend":
                # In downtrend, we're looking for sell signals
                signal = {
                    "side": "SELL",
                    "entry_price": current_price,
                    "pattern": f"LH/LL with Fibonacci {fib_zones['zones']['closest_level']} retracement",
                    "strength": min(1.0, 0.5 + 0.3 * hh_hl_result["downtrend_analysis"]["consecutive_ll"]),
                    "fibonacci_level": fib_zones["zones"]["closest_level"]
                }

                # For shorts, stop loss is above, take profit is below
                # Calculate standard TP/SL for short positions
                tp = current_price * (1 - 1.0 / 100)  # 1% default
                sl = current_price * (1 + 1.0 / 100)  # 1% default

                # Enhance with Fibonacci-based targets if possible
                fib_retracement = fib_zones["zones"]["retracement"]
                fib_extension = fib_zones["zones"]["extension"]

                # Use 0.618 retracement as stop loss if available
                if 0.618 in fib_retracement and fib_retracement[0.618] > current_price:
                    sl = fib_retracement[0.618]

                # Use 1.618 extension as take profit if available
                if 1.618 in fib_extension:
                    tp = fib_extension[1.618]

                signal["tp"] = tp
                signal["sl"] = sl

                # Calculate risk/reward ratio
                risk = sl - current_price
                reward = current_price - tp
                signal["risk_reward"] = reward / risk if risk > 0 else 0

                enhanced_result["signals"].append(signal)

    return enhanced_result


def integrate_fibonacci_with_candlestick(candlestick_pattern: Dict[str, Any],
                                         prices: List[float]) -> Dict[str, Any]:
    """
    Enhance candlestick pattern signals with Fibonacci analysis.

    Args:
        candlestick_pattern: Dictionary containing candlestick pattern information
        prices: List of price values

    Returns:
        Enhanced pattern with Fibonacci targets
    """
    # Create a copy of the pattern to modify
    enhanced_pattern = candlestick_pattern.copy()

    # Initialize Fibonacci analyzer
    fib_analyzer = FibonacciAnalyzer()

    # Get Fibonacci zones
    fib_zones = find_fibonacci_zones(prices)

    if not fib_zones["has_zones"]:
        # No Fibonacci confluence
        return enhanced_pattern

    # Add Fibonacci analysis to the pattern
    enhanced_pattern["fibonacci_zones"] = fib_zones["zones"]

    # Set trend alignment
    candlestick_bullish = enhanced_pattern.get("is_bullish", True)
    fib_trend = fib_zones["zones"]["trend"]

    trend_aligned = (candlestick_bullish and fib_trend == "uptrend") or \
                    (not candlestick_bullish and fib_trend == "downtrend")

    enhanced_pattern["trend_aligned"] = trend_aligned

    # Check if price is at key Fibonacci level
    at_key_level = fib_zones["zones"]["is_at_key_level"]
    closest_level = fib_zones["zones"]["closest_level"]

    enhanced_pattern["at_fib_level"] = at_key_level
    enhanced_pattern["closest_fib_level"] = closest_level

    # Determine confluence strength
    if trend_aligned and at_key_level:
        # Strong confluence - both trend and level align
        confluence_strength = 0.3

        # Higher strength for key levels
        if closest_level in [0.5, 0.618]:
            confluence_strength += 0.1
    elif at_key_level:
        # Moderate confluence - only at key level but trend doesn't align
        confluence_strength = 0.1
    else:
        # No confluence
        confluence_strength = 0

    # Add confluence strength to original pattern strength
    original_strength = enhanced_pattern.get("strength", 0.5)
    enhanced_pattern["strength"] = min(1.0, original_strength + confluence_strength)

    # Add Fibonacci-based price targets
    current_price = fib_zones["current_price"]
    swing_high = fib_zones["zones"]["swing_high"]
    swing_low = fib_zones["zones"]["swing_low"]

    # Calculate price targets based on Fibonacci
    if candlestick_bullish:
        # For bullish patterns
        targets = {
            "entry": current_price,
            "stop_loss": swing_low,  # Use swing low as stop loss
            "targets": {
                "target_1": swing_high,  # First target at previous swing high
                "target_2": swing_high + (swing_high - swing_low) * 0.618,  # 61.8% extension
                "target_3": swing_high + (swing_high - swing_low) * 1.0,  # 100% extension
                "target_4": swing_high + (swing_high - swing_low) * 1.618  # 161.8% extension
            }
        }
    else:
        # For bearish patterns
        targets = {
            "entry": current_price,
            "stop_loss": swing_high,  # Use swing high as stop loss
            "targets": {
                "target_1": swing_low,  # First target at previous swing low
                "target_2": swing_low - (swing_high - swing_low) * 0.618,  # 61.8% extension
                "target_3": swing_low - (swing_high - swing_low) * 1.0,  # 100% extension
                "target_4": swing_low - (swing_high - swing_low) * 1.618  # 161.8% extension
            }
        }

    # Add targets to enhanced pattern
    enhanced_pattern["fibonacci_targets"] = targets

    # Add risk-reward ratio
    if candlestick_bullish:
        risk = current_price - targets["stop_loss"]
        reward = targets["targets"]["target_2"] - current_price
    else:
        risk = targets["stop_loss"] - current_price
        reward = current_price - targets["targets"]["target_2"]

    if risk > 0:
        enhanced_pattern["risk_reward_ratio"] = reward / risk
    else:
        enhanced_pattern["risk_reward_ratio"] = 0

    return enhanced_pattern


def fibonacci_based_trade_setup(prices: List[float], trend: str = "auto") -> Dict[str, Any]:
    """
    Create a complete trade setup based on Fibonacci analysis.

    Args:
        prices: List of price values
        trend: Specify "uptrend", "downtrend", or "auto" to detect automatically

    Returns:
        Dictionary with complete trade setup
    """
    # Initialize Fibonacci analyzer
    fib_analyzer = FibonacciAnalyzer()

    # Automatically detect trend if set to auto
    if trend == "auto":
        # Use simple moving average crossover to determine trend
        if len(prices) >= 50:
            sma20 = np.mean(prices[-20:])
            sma50 = np.mean(prices[-50:])

            if sma20 > sma50:
                detected_trend = "uptrend"
            else:
                detected_trend = "downtrend"
        else:
            # Not enough data for reliable trend detection
            detected_trend = "unclear"
    else:
        detected_trend = trend

    # Get Fibonacci analysis
    fib_analysis = fib_analyzer.analyze_retracements(prices)

    if not fib_analysis:
        return {
            "status": "error",
            "message": "Could not perform Fibonacci analysis - insufficient price data or no clear swing points"
        }

    # Check if the trends align
    trends_aligned = detected_trend == fib_analysis["trend"]

    # Get trade signal
    trade_signal = fib_analyzer.get_trade_signals(prices)

    # Prepare response
    result = {
        "status": "success",
        "detected_trend": detected_trend,
        "fibonacci_trend": fib_analysis["trend"],
        "trends_aligned": trends_aligned,
        "current_price": fib_analysis["current_price"],
        "fibonacci_levels": {
            "retracement": fib_analysis["retracement_levels"],
            "extension": fib_analysis["extension_levels"]
        },
        "at_key_level": fib_analysis["is_at_key_level"],
        "closest_level": fib_analysis["closest_level"]
    }

    # Add trade signal if available
    if trade_signal:
        result["trade_signal"] = trade_signal

        # Calculate risk percentage
        entry = trade_signal["entry"]
        stop_loss = trade_signal["stop_loss"]

        if entry and stop_loss:
            if trade_signal["side"] == "BUY":
                risk_pct = ((entry - stop_loss) / entry) * 100
            else:
                risk_pct = ((stop_loss - entry) / entry) * 100

            result["risk_percentage"] = risk_pct
    else:
        result["trade_signal"] = None

    return result


def find_fibonacci_trade_opportunities(candles: List[Dict], min_risk_reward: float = 2.0) -> List[Dict[str, Any]]:
    """
    Scan price data to find Fibonacci-based trade opportunities.

    Args:
        candles: List of candlestick data dictionaries
        min_risk_reward: Minimum risk-reward ratio to consider a valid opportunity

    Returns:
        List of trade opportunities
    """
    # Extract close prices
    if not candles or len(candles) < 50:
        return []

    close_prices = [candle["close"] for candle in candles]

    # Initialize analyzer
    fib_analyzer = FibonacciAnalyzer()

    # Get Fibonacci analysis
    fib_analysis = fib_analyzer.analyze_retracements(close_prices)

    if not fib_analysis:
        return []

    # Get potential trade signal
    trade_signal = fib_analyzer.get_trade_signals(close_prices)

    # Check if we have a valid opportunity
    if not trade_signal or trade_signal["risk_reward_ratio"] < min_risk_reward:
        return []

    # Format the opportunity
    opportunity = {
        "signal": trade_signal["signal"],
        "side": trade_signal["side"],
        "pattern": f"Fibonacci {fib_analysis['closest_level']} retracement",
        "entry": trade_signal["entry"],
        "stop_loss": trade_signal["stop_loss"],
        "take_profit": trade_signal["take_profit"],
        "risk_reward_ratio": trade_signal["risk_reward_ratio"],
        "strength": trade_signal["strength"],
        "notes": trade_signal["notes"]
    }

    # Add the opportunity to the list
    return [opportunity]


# Example usage
if __name__ == "__main__":
    # Example price data
    import numpy as np

    # Generate sample price data (simulated uptrend)
    np.random.seed(42)
    x = np.arange(100)
    prices = 100 + 0.5 * x + 10 * np.sin(x / 10) + np.random.normal(0, 3, 100)

    # Test enhanced HH/HL with Fibonacci
    result = enhanced_hh_hl_with_fibonacci(prices.tolist())
    print("Enhanced HH/HL with Fibonacci:")
    print(f"Trend: {result['trend']}")
    print(f"Fibonacci confluence: {result['has_fib_confluence']}")

    if result["signals"]:
        print("\nTrade Signals:")
        for signal in result["signals"]:
            print(f"Side: {signal['side']}")
            print(f"Entry: {signal['entry_price']:.2f}")
            print(f"Stop Loss: {signal['sl']:.2f}")
            print(f"Take Profit: {signal['tp']:.2f}")
            print(f"Risk/Reward: {signal['risk_reward']:.2f}")
            print(f"Pattern: {signal['pattern']}")

    # Test Fibonacci-based trade setup
    setup = fibonacci_based_trade_setup(prices.tolist())
    print("\nFibonacci Trade Setup:")
    print(f"Status: {setup['status']}")
    print(f"Detected trend: {setup['detected_trend']}")
    print(f"Fibonacci trend: {setup['fibonacci_trend']}")
    print(f"At key level: {setup['at_key_level']} ({setup['closest_level']})")

    if setup["trade_signal"]:
        print("\nTrade Signal:")
        print(f"Type: {setup['trade_signal']['signal']}")
        print(f"Side: {setup['trade_signal']['side']}")
        print(f"Entry: {setup['trade_signal']['entry']:.2f}")
        print(f"Stop Loss: {setup['trade_signal']['stop_loss']:.2f}")
        print(f"Take Profit: {setup['trade_signal']['take_profit']:.2f}")
        print(f"Risk/Reward: {setup['trade_signal']['risk_reward_ratio']:.2f}")
        print(f"Risk %: {setup.get('risk_percentage', 0):.2f}%")