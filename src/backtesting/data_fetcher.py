"""
Fetches and manages historical data for backtesting.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.market_data.okx_client import OKXClient


class HistoricalDataFetcher:
    """
    Fetches and manages historical data for backtesting.
    """

    def __init__(self, client: Optional[OKXClient] = None, cache_dir: str = "cache/backtest"):
        """
        Initialize the data fetcher.

        Args:
            client: OKXClient instance for fetching data
            cache_dir: Directory to cache data
        """
        self.client = client or OKXClient()
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(
            self,
            symbol: str,
            timeframe: str = "1h",
            days: int = 30,
            use_cache: bool = True,
            logger: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical data for a symbol.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
            days: Number of days of data to fetch
            use_cache: Whether to use cached data if available
            logger: Optional logger function

        Returns:
            List of candle data dictionaries
        """
        # Check if we have cached data
        cache_file = self._get_cache_path(symbol, timeframe, days)

        if use_cache and os.path.exists(cache_file):
            # Check if cache is recent enough (less than 1 day old)
            cache_age = time.time() - os.path.getmtime(cache_file)

            if cache_age < 86400:  # 86400 seconds = 1 day
                if logger:
                    logger(f"Loading cached data for {symbol} ({timeframe}, {days} days)")

                # Load data from cache
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                return data

        # Calculate number of candles based on timeframe and days
        candles_count = self._calculate_candles_count(timeframe, days)

        if logger:
            logger(f"Fetching {candles_count} {timeframe} candles for {symbol}")

        # Fetch data from API
        candles = self.client.get_klines(symbol, interval=timeframe, limit=candles_count)

        if not candles:
            if logger:
                logger(f"No data available for {symbol}")
            return []

        if logger:
            logger(f"Fetched {len(candles)} candles for {symbol}")

        # Cache the data
        if len(candles) > 0:
            with open(cache_file, 'w') as f:
                json.dump(candles, f)

        return candles

    def fetch_multiple_symbols(
            self,
            symbols: List[str],
            timeframe: str = "1h",
            days: int = 30,
            use_cache: bool = True,
            logger: Optional[callable] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
            days: Number of days of data to fetch
            use_cache: Whether to use cached data if available
            logger: Optional logger function

        Returns:
            Dictionary mapping symbols to their historical data
        """
        result = {}

        for symbol in symbols:
            if logger:
                logger(f"Fetching data for {symbol}...")

            data = self.fetch_data(symbol, timeframe, days, use_cache, logger)

            if data:
                result[symbol] = data

            # Add a small delay to avoid rate limiting
            time.sleep(0.5)

        if logger:
            logger(f"Fetched data for {len(result)} symbols")

        return result

    def _get_cache_path(self, symbol: str, timeframe: str, days: int) -> str:
        """
        Get the file path for caching data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            days: Number of days

        Returns:
            Cache file path
        """
        filename = f"{symbol.replace('-', '_')}_{timeframe}_{days}d.json"
        return os.path.join(self.cache_dir, filename)

    def _calculate_candles_count(self, timeframe: str, days: int) -> int:
        """
        Calculate the number of candles needed for the given timeframe and days.

        Args:
            timeframe: Candle timeframe
            days: Number of days

        Returns:
            Number of candles
        """
        # Map timeframe to minutes
        timeframe_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
        }

        # Get minutes for the given timeframe
        if timeframe in timeframe_minutes:
            minutes_per_candle = timeframe_minutes[timeframe]
        else:
            # Default to 1h if timeframe not recognized
            minutes_per_candle = 60

        # Calculate candles per day
        candles_per_day = 24 * 60 // minutes_per_candle

        # Add 10% buffer for safety
        return int(days * candles_per_day * 1.1)


# Utility functions

def prepare_data_for_charts(backtest_results: Dict[str, Any]) -> Tuple[List, List]:
    """
    Prepare data for charting from backtest results.

    Args:
        backtest_results: Dictionary of backtest results

    Returns:
        Tuple of (equity_data, trade_markers)
    """
    # Prepare equity curve data
    equity_data = []
    for point in backtest_results.get('equity_curve', []):
        # Convert timestamp to datetime if needed
        if isinstance(point['timestamp'], (int, float)):
            if point['timestamp'] > 1e12:  # milliseconds
                dt = datetime.fromtimestamp(point['timestamp'] / 1000)
            else:  # seconds
                dt = datetime.fromtimestamp(point['timestamp'])
            time_str = dt.strftime('%Y-%m-%d %H:%M')
        else:
            time_str = point['timestamp']

        equity_data.append({
            'timestamp': time_str,
            'equity': point['equity']
        })

    # Prepare trade markers
    trade_markers = []
    for trade in backtest_results.get('trades', []):
        # Convert timestamps if needed
        if isinstance(trade['entry_time'], (int, float)):
            if trade['entry_time'] > 1e12:
                entry_dt = datetime.fromtimestamp(trade['entry_time'] / 1000)
            else:
                entry_dt = datetime.fromtimestamp(trade['entry_time'])
            entry_time = entry_dt.strftime('%Y-%m-%d %H:%M')
        else:
            entry_time = trade['entry_time']

        if isinstance(trade['exit_time'], (int, float)):
            if trade['exit_time'] > 1e12:
                exit_dt = datetime.fromtimestamp(trade['exit_time'] / 1000)
            else:
                exit_dt = datetime.fromtimestamp(trade['exit_time'])
            exit_time = exit_dt.strftime('%Y-%m-%d %H:%M')
        else:
            exit_time = trade['exit_time']

        # Create marker
        trade_markers.append({
            'entry_time': entry_time,
            'entry_price': trade['entry_price'],
            'exit_time': exit_time,
            'exit_price': trade['exit_price'],
            'side': trade['side'],
            'profit_pct': trade['profit_pct'],
            'profit_amount': trade['profit_amount'],
            'exit_reason': trade['exit_reason'],
            'pattern': trade.get('pattern', 'Unknown')
        })

    return equity_data, trade_markers


def format_backtest_summary(results: Dict[str, Any]) -> str:
    """
    Format backtest results into a readable summary.

    Args:
        results: Dictionary of backtest results

    Returns:
        Formatted summary string
    """
    summary = []

    # Basic performance metrics
    summary.append(f"Symbol: {results.get('symbol', 'Unknown')}")
    summary.append(f"Initial Capital: ${results.get('initial_capital', 0):.2f}")
    summary.append(f"Final Capital: ${results.get('final_capital', 0):.2f}")
    summary.append(f"Absolute Return: ${results.get('absolute_return', 0):.2f}")
    summary.append(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
    summary.append("")

    # Trade statistics
    summary.append(f"Total Trades: {results.get('total_trades', 0)}")
    summary.append(f"Winning Trades: {results.get('winning_trades', 0)} ({results.get('win_rate', 0):.2f}%)")
    summary.append(f"Losing Trades: {results.get('total_trades', 0) - results.get('winning_trades', 0)}")
    summary.append("")

    # Profit statistics
    summary.append(f"Average Win: {results.get('avg_win_pct', 0):.2f}% (${results.get('avg_win_amount', 0):.2f})")
    summary.append(f"Average Loss: {results.get('avg_loss_pct', 0):.2f}% (${results.get('avg_loss_amount', 0):.2f})")
    summary.append(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    summary.append("")

    # Risk statistics
    summary.append(
        f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}% (${results.get('max_drawdown_amount', 0):.2f})")
    summary.append(f"Max Consecutive Wins: {results.get('max_consecutive_wins', 0)}")
    summary.append(f"Max Consecutive Losses: {results.get('max_consecutive_losses', 0)}")

    return "\n".join(summary)