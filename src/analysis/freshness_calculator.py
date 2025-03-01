"""
Utilities for calculating and sorting by pattern freshness.
"""

from typing import List, Dict, Any, Tuple, Union

def calculate_freshness(close_prices: List[float],
                       peaks: List[Tuple[int, float]],
                       troughs: List[Tuple[int, float]]) -> int:
    """
    Calculate how fresh a pattern is based on peak/trough indices.

    Args:
        close_prices: List of price values
        peaks: List of (index, price) tuples for peaks
        troughs: List of (index, price) tuples for troughs

    Returns:
        Freshness score (lower is fresher)
    """
    if not peaks and not troughs:
        return 0

    last_candle_index = len(close_prices) - 1

    # Find latest peak and trough indices
    latest_peak_index = peaks[-1][0] if peaks else 0
    latest_trough_index = troughs[-1][0] if troughs else 0

    # Calculate freshness (how many candles since pattern formed)
    freshness = last_candle_index - max(latest_peak_index, latest_trough_index)

    return freshness

def sort_by_freshness_and_strength(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort patterns by freshness (primary) and strength (secondary).

    Args:
        patterns: List of pattern dictionaries with freshness and strength values

    Returns:
        Sorted list of patterns
    """
    # Check if patterns have freshness field
    if patterns and 'freshness' in patterns[0]:
        # Sort by freshness (ascending) then by strength (descending)
        return sorted(patterns, key=lambda x: (x['freshness'], -x['strength']))
    else:
        # Sort by strength only (descending)
        return sorted(patterns, key=lambda x: x['strength'], reverse=True)

def format_volume(volume: float) -> str:
    """
    Format volume value for display.

    Args:
        volume: Volume in USD

    Returns:
        Formatted string (e.g., "$5.2M" or "$423.5K")
    """
    if volume >= 1000000:
        return f"${volume/1000000:.2f}M"
    else:
        return f"${volume/1000:.2f}K"