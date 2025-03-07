"""
Adapters for using Fibonacci strategy with the backtesting engine.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.strategies.fibonacci_levels import FibonacciAnalyzer, get_fibonacci_signal
from src.strategies.fibonacci_integration import (
    enhanced_hh_hl_with_fibonacci,
    fibonacci_based_trade_setup,
    find_fibonacci_trade_opportunities,
)


def adapt_pure_fibonacci_strategy(candles: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Adapter for pure Fibonacci strategy for backtesting.

    Args:
        candles: List of historical candle data

    Returns:
        Trade signal dictionary or None
    """
    if len(candles) < 50:  # Need enough data for reliable Fibonacci analysis
        return None

    # Extract close prices
    close_prices = [candle['close'] for candle in candles]

    # Get Fibonacci-based trade signal
    signal = get_fibonacci_signal(close_prices)

    if not signal or signal.get('signal') is None:
        return None

    # Convert to standard signal format
    return {
        'side': signal['side'],
        'pattern': f"Fibonacci {signal['closest_level'] if 'closest_level' in signal else ''} Retracement",
        'strength': signal.get('strength', 0.5)
    }


def adapt_fibonacci_hh_hl_strategy(candles: List[Dict], consecutive_count: int = 2) -> Optional[Dict[str, Any]]:
    """
    Adapter for enhanced HH/HL with Fibonacci strategy for backtesting.

    Args:
        candles: List of historical candle data
        consecutive_count: Number of consecutive HH/HL required

    Returns:
        Trade signal dictionary or None
    """
    if len(candles) < 50:  # Need enough data
        return None

    # Extract close prices
    close_prices = [candle['close'] for candle in candles]

    # Get enhanced analysis
    result = enhanced_hh_hl_with_fibonacci(close_prices, consecutive_count=consecutive_count)

    # Check if we have signals with Fibonacci confluence
    if not result["has_fib_confluence"] or not result["signals"]:
        return None

    # Use the first signal
    signal = result["signals"][0]

    # Convert to standard signal format
    return {
        'side': signal['side'],
        'pattern': signal['pattern'],
        'strength': signal['strength']
    }


def adapt_fibonacci_setup_strategy(candles: List[Dict], min_risk_reward: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    Adapter for complete Fibonacci trade setup strategy for backtesting.

    Args:
        candles: List of historical candle data
        min_risk_reward: Minimum risk-reward ratio required

    Returns:
        Trade signal dictionary or None
    """
    if len(candles) < 50:  # Need enough data
        return None

    # Extract close prices
    close_prices = [candle['close'] for candle in candles]

    # Get trade setup
    setup = fibonacci_based_trade_setup(close_prices)

    # Check if setup was successful and has a trade signal
    if setup["status"] != "success" or not setup["trade_signal"]:
        return None

    # Get trade signal
    trade_signal = setup["trade_signal"]

    # Check risk-reward ratio
    if trade_signal.get("risk_reward_ratio", 0) < min_risk_reward:
        return None

    # Convert to standard signal format
    return {
        'side': trade_signal['side'],
        'pattern': f"Fibonacci {setup['closest_level']} Setup",
        'strength': trade_signal.get('strength', 0.5)
    }


def adapt_multi_timeframe_fibonacci(candles_1h: List[Dict], candles_4h: List[Dict],
                                    candles_1d: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Adapter for multi-timeframe Fibonacci strategy for backtesting.
    Looks for confluence across multiple timeframes.

    Args:
        candles_1h: List of 1-hour candle data
        candles_4h: List of 4-hour candle data
        candles_1d: List of daily candle data

    Returns:
        Trade signal dictionary or None
    """
    # Need enough data for all timeframes
    if len(candles_1h) < 100 or len(candles_4h) < 50 or len(candles_1d) < 30:
        return None

    # Extract close prices for each timeframe
    close_1h = [candle['close'] for candle in candles_1h]
    close_4h = [candle['close'] for candle in candles_4h]
    close_1d = [candle['close'] for candle in candles_1d]

    # Get signals for each timeframe
    signal_1h = get_fibonacci_signal(close_1h)
    signal_4h = get_fibonacci_signal(close_4h)
    signal_1d = get_fibonacci_signal(close_1d)

    # Count valid signals
    valid_signals = [s for s in [signal_1h, signal_4h, signal_1d] if s and s.get('signal') is not None]

    if not valid_signals:
        return None

    # Check for signal agreement
    sides = [s['side'] for s in valid_signals]

    # If we have disagreement, no signal
    if 'BUY' in sides and 'SELL' in sides:
        return None

    # Need at least 2 timeframes to agree
    if len(valid_signals) < 2:
        return None

    # Use the side from the first signal
    side = valid_signals[0]['side']

    # Calculate combined strength
    strength = min(1.0, sum(s.get('strength', 0.5) for s in valid_signals) / len(valid_signals))

    # Enhance strength based on number of aligned timeframes
    if len(valid_signals) == 3:
        strength += 0.2  # Bonus for all timeframes aligned

    return {
        'side': side,
        'pattern': f"Multi-timeframe Fibonacci ({len(valid_signals)}/3)",
        'strength': min(1.0, strength)
    }


def fibonacci_opportunity_scanner(candles: List[Dict], min_risk_reward: float = 2.0) -> List[Dict[str, Any]]:
    """
    Scan for Fibonacci trade opportunities for backtesting.

    Args:
        candles: List of historical candle data
        min_risk_reward: Minimum risk-reward ratio required

    Returns:
        List of trade opportunities
    """
    return find_fibonacci_trade_opportunities(candles, min_risk_reward)


# Example usage
if __name__ == "__main__":
    # Example data (this would normally come from historical data)
    import numpy as np

    # Generate sample candle data
    np.random.seed(42)
    sample_candles = []
    close = 100.0

    for i in range(100):
        # Simple random walk
        close = close + np.random.normal(0.1, 1.0)
        candle = {
            'timestamp': i,
            'open': close - np.random.uniform(0, 1),
            'high': close + np.random.uniform(0, 2),
            'low': close - np.random.uniform(0, 2),
            'close': close,
            'volume': np.random.uniform(1000, 5000)
        }
        sample_candles.append(candle)

    # Test the Fibonacci strategy adapter
    signal = adapt_pure_fibonacci_strategy(sample_candles)
    if signal:
        print(f"Pure Fibonacci Signal: {signal['side']} ({signal['pattern']})")
    else:
        print("No pure Fibonacci signal")

    # Test the enhanced HH/HL with Fibonacci adapter
    signal = adapt_fibonacci_hh_hl_strategy(sample_candles)
    if signal:
        print(f"Enhanced HH/HL Signal: {signal['side']} ({signal['pattern']})")
    else:
        print("No enhanced HH/HL signal")

    # Test the Fibonacci trade setup adapter
    signal = adapt_fibonacci_setup_strategy(sample_candles)
    if signal:
        print(f"Fibonacci Setup Signal: {signal['side']} ({signal['pattern']})")
    else:
        print("No Fibonacci setup signal")