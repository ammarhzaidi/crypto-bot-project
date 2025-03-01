#!/usr/bin/env python
"""
Script to run candlestick pattern detection on real market data from OKX.
This integrates the pattern detection with the existing OKX client.
"""

import sys
import os
import logging
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from src directory
from src.market_data.okx_client import OKXClient
from src.strategies.candlestick_patterns import CandlestickPatternFinder, format_hammer_results

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('candlestick_finder.log')
        ]
    )
    return logging.getLogger('candlestick_finder_script')


def run_candlestick_detection(symbols_count=100, timeframe="1h", candles_count=100):
    """
    Run candlestick pattern detection on real market data.

    Args:
        symbols_count: Number of top volume symbols to analyze
        timeframe: Candlestick timeframe ("1h", "4h", "1d", etc.)
        candles_count: Number of historical candles to fetch
    """
    logger = setup_logging()
    logger.info(f"Starting candlestick pattern detection with top {symbols_count} symbols")

    # Initialize market data client
    client = OKXClient()

    # Get top volume symbols
    symbols = client.get_top_volume_symbols(limit=symbols_count)

    if not symbols:
        logger.error("Failed to fetch symbols. Exiting.")
        return

    logger.info(f"Analyzing {len(symbols)} symbols: {symbols}")

    # Initialize the candlestick pattern finder
    pattern_finder = CandlestickPatternFinder(logger=logger)

    # Store results for all symbols
    all_hammers = []

    # Analyze each symbol
    for symbol in symbols:
        logger.info(f"Fetching {timeframe} candlestick data for {symbol}")

        try:
            # Get historical price data
            klines = client.get_klines(symbol, interval=timeframe, limit=candles_count)

            if not klines:
                logger.warning(f"No data available for {symbol}, skipping")
                continue

            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(klines)

            # Rename columns to match the pattern finder's expected format
            df = df.rename(columns={
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })

            # Find hammer patterns
            hammers = pattern_finder.find_hammers(df)

            if hammers:
                logger.info(f"Found {len(hammers)} hammer patterns for {symbol}")

                # Add symbol to each result
                for hammer in hammers:
                    hammer['symbol'] = symbol

                all_hammers.extend(hammers)
            else:
                logger.info(f"No hammer patterns found for {symbol}")

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")

        # Small delay to avoid hammering the API
        time.sleep(0.5)

    # Sort hammers by timestamp (newest first)
    all_hammers.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)

    # Display results
    if all_hammers:
        logger.info(f"Found a total of {len(all_hammers)} hammer patterns across all symbols")

        # Create the header
        print("\n=== HAMMER PATTERN DETECTION RESULTS ===")
        header = f"{'Symbol':<10} {'Timestamp (PST)':<20} {'Price':<10} {'Body %':<8} {'LS Ratio':<8} {'Bullish':<8} {'Strength':<8}"
        divider = "-" * len(header)

        print(header)
        print(divider)

        for hammer in all_hammers:
            timestamp_str = hammer['timestamp'].strftime('%Y-%m-%d %H:%M') if hammer['timestamp'] else 'N/A'
            price_str = f"{hammer['close']:.4f}"
            body_percent_str = f"{hammer['body_percent']:.1f}%"
            lower_shadow_ratio_str = f"{hammer['lower_shadow_ratio']:.1f}x"
            is_bullish_str = "Yes" if hammer['is_bullish'] else "No"
            strength_str = f"{hammer['strength']:.2f}"

            line = f"{hammer['symbol']:<10} {timestamp_str:<20} {price_str:<10} {body_percent_str:<8} " \
                   f"{lower_shadow_ratio_str:<8} {is_bullish_str:<8} {strength_str:<8}"
            print(line)
    else:
        print("No hammer patterns found across all analyzed symbols.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run candlestick pattern detection on OKX market data')
    parser.add_argument('--symbols', type=int, default=20, help='Number of top volume symbols to analyze')
    parser.add_argument('--timeframe', type=str, default='1h', help='Candlestick timeframe (1h, 4h, 1d, etc.)')
    parser.add_argument('--candles', type=int, default=100, help='Number of historical candles to fetch')

    args = parser.parse_args()

    # Run the pattern detection
    run_candlestick_detection(
        symbols_count=args.symbols,
        timeframe=args.timeframe,
        candles_count=args.candles
    )