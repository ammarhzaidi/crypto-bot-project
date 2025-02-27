import sys
import os
import logging
import time

# Add the src directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_data.okx_client import OKXClient
from src.strategies.hh_hl_strategy import analyze_price_action, find_peaks_and_troughs


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def test_basic_patterns():
    """Test the HH/HL and LH/LL detection with predefined patterns."""
    logger = logging.getLogger('test_basic_patterns')

    # Test uptrend pattern
    uptrend = [10, 9, 12, 11, 15, 13, 17, 16, 20]
    uptrend_result = analyze_price_action(uptrend, smoothing=1, consecutive_count=2)
    logger.info(f"Uptrend pattern result: {uptrend_result['trend']}")

    # Test downtrend pattern
    downtrend = [20, 18, 19, 16, 17, 14, 15, 12, 10]
    downtrend_result = analyze_price_action(downtrend, smoothing=1, consecutive_count=2)
    logger.info(f"Downtrend pattern result: {downtrend_result['trend']}")

    # Test flat pattern
    sideways = [10, 11, 9, 12, 10, 11, 9, 12, 10]
    sideways_result = analyze_price_action(sideways, smoothing=1, consecutive_count=2)
    logger.info(f"Sideways pattern result: {sideways_result['trend']}")

    assert uptrend_result['trend'] == 'uptrend', "Failed to detect uptrend"
    assert downtrend_result['trend'] == 'downtrend', "Failed to detect downtrend"
    assert sideways_result['trend'] == 'no_trend', "Failed to detect sideways/no trend"

    return True


def test_with_real_data():
    """Test the HH/HL strategy with real market data from OKX."""
    logger = logging.getLogger('test_real_data')

    # Initialize OKX client
    logger.info("Initializing OKX client...")
    client = OKXClient()

    # Fetch available symbols
    logger.info("Fetching available symbols...")
    symbols = client.get_all_symbols()

    if not symbols:
        logger.error("No symbols found. Exiting...")
        return False

    # Take top 5 symbols for testing
    test_symbols = symbols[:5]
    logger.info(f"Testing strategy on symbols: {test_symbols}")

    # Dictionary to store results
    results = {}

    # Analyze each symbol
    for symbol in test_symbols:
        logger.info(f"\nAnalyzing {symbol}...")

        # Get historical data (1-hour candles)
        klines = client.get_klines(symbol, interval="1h", limit=48)  # Last 48 hours

        if not klines:
            logger.warning(f"No historical data available for {symbol}, skipping.")
            continue

        # Extract close prices
        close_prices = [candle["close"] for candle in klines]

        # Analyze price action
        logger.info(f"Analyzing price action for {symbol} with {len(close_prices)} data points...")
        analysis = analyze_price_action(close_prices, smoothing=1, consecutive_count=2)

        # Store results
        results[symbol] = analysis

        # Log results
        trend = analysis["trend"]
        if trend == "uptrend":
            logger.info(f"✅ {symbol} is in an UPTREND")
            logger.info(f"  Higher Highs: {analysis['uptrend_analysis']['consecutive_hh']} consecutive")
            logger.info(f"  Higher Lows: {analysis['uptrend_analysis']['consecutive_hl']} consecutive")
        elif trend == "downtrend":
            logger.info(f"🔻 {symbol} is in a DOWNTREND")
            logger.info(f"  Lower Highs: {analysis['downtrend_analysis']['consecutive_lh']} consecutive")
            logger.info(f"  Lower Lows: {analysis['downtrend_analysis']['consecutive_ll']} consecutive")
        else:
            logger.info(f"➖ {symbol} has NO CLEAR TREND")

        # Output peaks and troughs for debugging
        logger.info(f"  Peaks detected: {analysis['uptrend_analysis']['peaks']}")
        logger.info(f"  Troughs detected: {analysis['uptrend_analysis']['troughs']}")

    # Summary
    uptrends = [s for s, r in results.items() if r["trend"] == "uptrend"]
    downtrends = [s for s, r in results.items() if r["trend"] == "downtrend"]
    no_trends = [s for s, r in results.items() if r["trend"] == "no_trend"]

    logger.info("\n=== SUMMARY ===")
    logger.info(f"Total symbols analyzed: {len(results)}")
    logger.info(f"Uptrends: {len(uptrends)} - {uptrends}")
    logger.info(f"Downtrends: {len(downtrends)} - {downtrends}")
    logger.info(f"No clear trend: {len(no_trends)} - {no_trends}")

    return len(results) > 0


def test_with_simulated_data():
    """Test with simulated data when real API fails."""
    logger = logging.getLogger('test_simulated')

    import random

    # Create simulated price series
    def generate_simulated_prices(pattern, length=48):
        prices = []
        base = 1000.0

        if pattern == "uptrend":
            for i in range(length):
                if i % 3 == 0:
                    base += random.uniform(10, 20)  # Higher high
                elif i % 3 == 1:
                    base -= random.uniform(1, 8)  # Pull back but not too much
                else:
                    base += random.uniform(5, 15)  # Higher low
                prices.append(base)

        elif pattern == "downtrend":
            for i in range(length):
                if i % 3 == 0:
                    base -= random.uniform(10, 20)  # Lower low
                elif i % 3 == 1:
                    base += random.uniform(1, 8)  # Bounce but not too much
                else:
                    base -= random.uniform(5, 15)  # Lower high
                prices.append(base)