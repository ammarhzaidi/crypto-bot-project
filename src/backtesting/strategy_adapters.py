"""
Strategy adapters for backtesting existing strategies.
"""

import pandas as pd
from typing import List, Dict, Any, Callable, Optional
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategies.hh_hl_strategy import analyze_price_action
from src.strategies.candlestick_patterns.finder import CandlestickPatternFinder


def adapt_hhhl_strategy(candles: List[Dict], consecutive_count: int = 2) -> Dict:
    """
    Adapter for the HH/HL strategy for backtesting.

    Args:
        candles: List of historical candle data
        consecutive_count: Number of consecutive patterns required

    Returns:
        Trade signal dictionary or None
    """
    if len(candles) < 10:  # Need enough data for pattern detection
        return None

    # Extract close prices
    close_prices = [candle['close'] for candle in candles]

    # Analyze for HH/HL patterns
    result = analyze_price_action(close_prices, smoothing=1, consecutive_count=consecutive_count)

    # Get the detected trend
    trend = result.get('trend', 'no_trend')

    if trend == 'uptrend':
        hh_count = result['uptrend_analysis']['consecutive_hh']
        hl_count = result['uptrend_analysis']['consecutive_hl']
        pattern = f"{hh_count} HH, {hl_count} HL"

        # Calculate pattern strength
        strength = min(hh_count, hl_count) / 5  # Normalize to 0-1 range

        return {
            'side': 'BUY',
            'pattern': pattern,
            'strength': strength
        }

    elif trend == 'downtrend':
        lh_count = result['downtrend_analysis']['consecutive_lh']
        ll_count = result['downtrend_analysis']['consecutive_ll']
        pattern = f"{lh_count} LH, {ll_count} LL"

        # Calculate pattern strength
        strength = min(lh_count, ll_count) / 5  # Normalize to 0-1 range

        return {
            'side': 'SELL',
            'pattern': pattern,
            'strength': strength
        }

    return None


def adapt_candlestick_strategy(
        candles: List[Dict],
        pattern_types: List[str] = None,
        min_strength: float = 0.3,
        volume_confirmation: bool = False,
        prior_trend: bool = False
) -> Dict:
    """
    Adapter for candlestick pattern strategy for backtesting.

    Args:
        candles: List of historical candle data
        pattern_types: List of pattern types to look for
        min_strength: Minimum pattern strength to generate a signal
        volume_confirmation: Whether to require volume confirmation
        prior_trend: Whether to require prior trend

    Returns:
        Trade signal dictionary or None
    """
    if len(candles) < 5:  # Need enough data for pattern detection
        return None

    # Default to common bullish patterns if none specified
    if pattern_types is None:
        pattern_types = ['hammer', 'bullish_engulfing', 'piercing', 'morning_star']

    # Convert candles to DataFrame
    df = pd.DataFrame(candles)

    # Initialize pattern finder with silent logger
    finder = CandlestickPatternFinder(logger=lambda x: None)

    # Configure pattern confirmations if needed
    if volume_confirmation or prior_trend:
        for pattern in pattern_types:
            if pattern in ['bullish_engulfing', 'piercing', 'morning_star']:
                finder.set_pattern_confirmation(pattern, 'use_volume_confirmation', volume_confirmation)
                finder.set_pattern_confirmation(pattern, 'use_prior_trend', prior_trend)

    # Find patterns in the most recent candles
    patterns = finder.find_patterns(df, pattern_types)

    # Filter for patterns in the most recent candle
    last_idx = len(candles) - 1
    recent_patterns = [p for p in patterns if p.get('index', 0) == last_idx]

    # Find the strongest pattern above minimum strength
    if recent_patterns:
        # Sort by strength (descending)
        sorted_patterns = sorted(recent_patterns, key=lambda p: p.get('strength', 0), reverse=True)

        # Get the strongest pattern
        strongest = sorted_patterns[0]

        # Check if it meets the minimum strength requirement
        if strongest.get('strength', 0) >= min_strength:
            is_bullish = strongest.get('is_bullish', True)

            return {
                'side': 'BUY' if is_bullish else 'SELL',
                'pattern': strongest.get('pattern_type', 'Unknown'),
                'strength': strongest.get('strength', 0)
            }

    return None


# Create a combined strategy adapter
def combined_strategy(
        candles: List[Dict],
        strategies: List[Callable] = None,
        weights: List[float] = None
) -> Dict:
    """
    Combine multiple strategies with optional weighting.

    Args:
        candles: List of historical candle data
        strategies: List of strategy functions to combine
        weights: Optional weights for each strategy

    Returns:
        Trade signal dictionary or None
    """
    if strategies is None:
        # Default to HHHL and candlestick strategies
        strategies = [
            adapt_hhhl_strategy,
            lambda c: adapt_candlestick_strategy(c, ['hammer', 'bullish_engulfing'])
        ]

    if weights is None:
        # Equal weights by default
        weights = [1.0 / len(strategies)] * len(strategies)

    # Collect signals from all strategies
    signals = []
    for strategy, weight in zip(strategies, weights):
        signal = strategy(candles)
        if signal:
            signal['weight'] = weight
            signals.append(signal)

    if not signals:
        return None

    # Combine signals - simple implementation just returns the highest weighted signal
    return max(signals, key=lambda s: s.get('weight', 0) * s.get('strength', 0))