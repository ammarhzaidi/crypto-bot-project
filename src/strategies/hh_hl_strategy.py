import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def find_peaks_and_troughs(prices: List[float], smoothing: int = 2) -> Tuple[List[int], List[int]]:
    """
    Find peaks (highs) and troughs (lows) in a price series.

    Args:
        prices: List of price values
        smoothing: Number of points on each side to use for comparison

    Returns:
        Tuple of (peak_indices, trough_indices)
    """
    if len(prices) < (2 * smoothing + 1):
        return [], []

    # Convert to numpy array for easier manipulation
    price_array = np.array(prices)

    # Find peaks (local maxima)
    peak_indices = []
    # Find troughs (local minima)
    trough_indices = []

    # Skip the first and last few points based on smoothing factor
    for i in range(smoothing, len(price_array) - smoothing):
        # Define window for comparison
        window = price_array[i - smoothing:i + smoothing + 1]
        current_price = price_array[i]

        # Check if current point is a peak
        if current_price == max(window):
            peak_indices.append(i)

        # Check if current point is a trough
        if current_price == min(window):
            trough_indices.append(i)

    return peak_indices, trough_indices


def detect_hh_hl(prices: List[float], smoothing: int = 2,
                 consecutive_count: int = 2, timestamps: Optional[List] = None) -> Dict[str, Any]:
    """
    Detect if there are Higher Highs and Higher Lows in the price series.

    Args:
        prices: List of price values
        smoothing: Number of points to use for finding peaks and troughs
        consecutive_count: Number of consecutive HH/HL required for confirmation

    Returns:
        Dictionary with detection results
    """


    peak_indices, trough_indices = find_peaks_and_troughs(prices, smoothing)

    if len(peak_indices) < consecutive_count or len(trough_indices) < consecutive_count:
        return {
            "higher_highs": False,
            "higher_lows": False,
            "uptrend": False,
            "consecutive_hh": 0,
            "consecutive_hl": 0,
            "peaks": [],
            "troughs": []
        }

    # Get the actual price values at peaks and troughs
    peaks = [prices[i] for i in peak_indices]
    troughs = [prices[i] for i in trough_indices]

    # Check for Higher Highs
    higher_highs = True
    consecutive_hh = 0

    # Start from the end and look back
    for i in range(len(peaks) - 1, 0, -1):
        if peaks[i] > peaks[i - 1]:
            consecutive_hh += 1
        else:
            higher_highs = False
            break

        # If we've found enough consecutive HH, we can stop
        if consecutive_hh >= consecutive_count - 1:
            higher_highs = True
            break

    # Check for Higher Lows
    higher_lows = True
    consecutive_hl = 0

    # Start from the end and look back
    for i in range(len(troughs) - 1, 0, -1):
        if troughs[i] > troughs[i - 1]:
            consecutive_hl += 1
        else:
            higher_lows = False
            break

        # If we've found enough consecutive HL, we can stop
        if consecutive_hl >= consecutive_count - 1:
            higher_lows = True
            break

    # An uptrend is confirmed if both HH and HL are true
    uptrend = higher_highs and higher_lows

    latest_pattern_timestamp = None
    if timestamps and (higher_highs or higher_lows):
        # Get the latest peak/trough index
        latest_indices = []
        if higher_highs and peak_indices:
            latest_indices.append(peak_indices[-1])
        if higher_lows and trough_indices:
            latest_indices.append(trough_indices[-1])

        if latest_indices:
            # Get the most recent peak/trough index
            latest_idx = max(latest_indices)
            if 0 <= latest_idx < len(timestamps):
                latest_pattern_timestamp = timestamps[latest_idx]



    return {
        "higher_highs": higher_highs,
        "higher_lows": higher_lows,
        "uptrend": uptrend,
        "consecutive_hh": consecutive_hh + 1 if higher_highs else 0,
        "consecutive_hl": consecutive_hl + 1 if higher_lows else 0,
        "peaks": list(zip(peak_indices, peaks)),
        "troughs": list(zip(trough_indices, troughs)),
        "latest_pattern_timestamp": latest_pattern_timestamp
    }


def detect_lh_ll(prices: List[float], smoothing: int = 2,
                 consecutive_count: int = 2, timestamps: Optional[List] = None) -> Dict[str, Any]:
    """
    Detect if there are Lower Highs and Lower Lows in the price series.

    Args:
        prices: List of price values
        smoothing: Number of points to use for finding peaks and troughs
        consecutive_count: Number of consecutive LH/LL required for confirmation
        timestamps: Optional list of timestamps corresponding to price data points

    Returns:
        Dictionary with detection results
    """
    peak_indices, trough_indices = find_peaks_and_troughs(prices, smoothing)

    if len(peak_indices) < consecutive_count or len(trough_indices) < consecutive_count:
        return {
            "lower_highs": False,
            "lower_lows": False,
            "downtrend": False,
            "consecutive_lh": 0,
            "consecutive_ll": 0,
            "peaks": [],
            "troughs": [],
            "latest_pattern_timestamp": None
        }

    # Get the actual price values at peaks and troughs
    peaks = [prices[i] for i in peak_indices]
    troughs = [prices[i] for i in trough_indices]

    # Check for Lower Highs
    lower_highs = True
    consecutive_lh = 0

    # Start from the end and look back
    for i in range(len(peaks) - 1, 0, -1):
        if peaks[i] < peaks[i - 1]:
            consecutive_lh += 1
        else:
            lower_highs = False
            break

        # If we've found enough consecutive LH, we can stop
        if consecutive_lh >= consecutive_count - 1:
            lower_highs = True
            break

    # Check for Lower Lows
    lower_lows = True
    consecutive_ll = 0

    # Start from the end and look back
    for i in range(len(troughs) - 1, 0, -1):
        if troughs[i] < troughs[i - 1]:
            consecutive_ll += 1
        else:
            lower_lows = False
            break

        # If we've found enough consecutive LL, we can stop
        if consecutive_ll >= consecutive_count - 1:
            lower_lows = True
            break

    # A downtrend is confirmed if both LH and LL are true
    downtrend = lower_highs and lower_lows

    # Calculate the timestamp of the latest LH/LL formation
    latest_pattern_timestamp = None
    if timestamps and (lower_highs or lower_lows):
        # Get the latest peak/trough index
        latest_indices = []
        if lower_highs and peak_indices:
            latest_indices.append(peak_indices[-1])
        if lower_lows and trough_indices:
            latest_indices.append(trough_indices[-1])

        if latest_indices:
            # Get the most recent peak/trough index
            latest_idx = max(latest_indices)
            if 0 <= latest_idx < len(timestamps):
                latest_pattern_timestamp = timestamps[latest_idx]

    return {
        "lower_highs": lower_highs,
        "lower_lows": lower_lows,
        "downtrend": downtrend,
        "consecutive_lh": consecutive_lh + 1 if lower_highs else 0,
        "consecutive_ll": consecutive_ll + 1 if lower_lows else 0,
        "peaks": list(zip(peak_indices, peaks)),
        "troughs": list(zip(trough_indices, troughs)),
        "latest_pattern_timestamp": latest_pattern_timestamp
    }


def analyze_price_action(prices: List[float], smoothing: int = 2,
                         consecutive_count: int = 2, timestamps: Optional[List] = None) -> Dict[str, Any]:
    """
    Analyze price action for both uptrend and downtrend patterns.
    """
    uptrend_analysis = detect_hh_hl(prices, smoothing, consecutive_count, timestamps)
    downtrend_analysis = detect_lh_ll(prices, smoothing, consecutive_count, timestamps)

    # Determine overall trend
    if uptrend_analysis["uptrend"] and not downtrend_analysis["downtrend"]:
        trend = "uptrend"
    elif downtrend_analysis["downtrend"] and not uptrend_analysis["uptrend"]:
        trend = "downtrend"
    elif not uptrend_analysis["uptrend"] and not downtrend_analysis["downtrend"]:
        trend = "no_trend"
    else:
        # This shouldn't happen in theory, but just in case
        trend = "conflicting"


    return {
        "trend": trend,
        "uptrend_analysis": uptrend_analysis,
        "downtrend_analysis": downtrend_analysis
    }


# Example usage
if __name__ == "__main__":
    # Example: Clear uptrend with HH and HL
    uptrend_prices = [10, 9, 12, 11, 15, 13, 17, 16, 20]

    # Example: Clear downtrend with LH and LL
    downtrend_prices = [20, 18, 19, 16, 17, 14, 15, 12, 10]

    # Example: Sideways/no clear trend
    sideways_prices = [10, 11, 9, 12, 10, 11, 9, 12, 10]

    print("Uptrend Analysis:")
    print(analyze_price_action(uptrend_prices, smoothing=1, consecutive_count=2))

    print("\nDowntrend Analysis:")
    print(analyze_price_action(downtrend_prices, smoothing=1, consecutive_count=2))

    print("\nSideways Analysis:")
    print(analyze_price_action(sideways_prices, smoothing=1, consecutive_count=2))