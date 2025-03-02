"""
Analyzer for candlestick patterns.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import pandas as pd
from datetime import datetime, timezone
import pytz
from src.market_data.okx_client import OKXClient
from src.strategies.candlestick_patterns.finder import CandlestickPatternFinder
from src.analysis.freshness_calculator import format_volume

class CandlestickAnalyzer:
    """
    Analyzer for detecting candlestick patterns.
    """

    def __init__(self, logger: Optional[Callable] = None):
        """
        Initialize the candlestick analyzer.

        Args:
            logger: Optional logger function for analysis progress
        """
        self.logger = logger
        self.pattern_finder = CandlestickPatternFinder(logger=logger)
        self.pakistan_tz = pytz.timezone('Asia/Karachi')

    def analyze_symbols(self,
                       symbols: List[str],
                       client: OKXClient,
                       timeframe: str = "1h",
                       candles_count: int = 100,
                       selected_patterns: List[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple symbols for candlestick patterns.

        Args:
            symbols: List of symbols to analyze
            client: OKXClient instance for data fetching
            timeframe: Candlestick timeframe
            candles_count: Number of candles to analyze
            selected_patterns: List of pattern names to look for

        Returns:
            List of detected patterns
        """
        if selected_patterns is None:
            selected_patterns = ["hammer"]  # Default to hammer pattern

        if self.logger:
            self.logger(f"Starting candlestick pattern analysis with top {len(symbols)} symbols")
            self.logger(f"Selected patterns: {', '.join(selected_patterns)}")

        all_patterns = []

        # Analyze each symbol
        for i, symbol in enumerate(symbols):
            if self.logger:
                self.logger(f"Analyzing {symbol} ({i+1}/{len(symbols)})...")

            try:
                # Get historical price data
                klines = client.get_klines(symbol, interval=timeframe, limit=candles_count)

                if not klines:
                    if self.logger:
                        self.logger(f"No data available for {symbol}, skipping")
                    continue

                # Get ticker data for volume
                ticker = client.get_ticker(symbol)
                usd_volume = 0
                if ticker and "volume_24h" in ticker and "last_price" in ticker:
                    usd_volume = ticker["volume_24h"] * ticker["last_price"]

                # Convert to DataFrame
                df = pd.DataFrame(klines)

                # Check if data has required columns
                if not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close']):
                    if self.logger:
                        self.logger(f"Data for {symbol} is missing required columns, skipping")
                        self.logger(f"Available columns: {df.columns.tolist()}")
                    continue

                # Find patterns based on selection
                patterns_found = False

                # Hammer pattern detection
                if 'hammer' in selected_patterns:
                    hammers = self.pattern_finder.find_hammers(df)
                    if hammers:
                        patterns_found = True
                        if self.logger:
                            self.logger(f"Found {len(hammers)} hammer patterns for {symbol}")

                        for hammer in hammers:
                            # Add symbol, pattern type and volume
                            hammer['symbol'] = symbol
                            hammer['pattern_type'] = 'Hammer'
                            hammer['usd_volume'] = usd_volume

                            # Convert timestamp to Pakistan time zone if needed
                            if 'timestamp' in hammer and hammer['timestamp']:
                                try:
                                    # Keep existing timezone conversion
                                    pass
                                except Exception as e:
                                    if self.logger:
                                        self.logger(f"Error converting timestamp: {e}")

                            # Calculate quality rating based on strength
                            hammer['quality'] = self._calculate_quality(hammer['strength'])

                            all_patterns.append(hammer)
                    elif self.logger:
                        self.logger(f"No hammer patterns found for {symbol}")

                # Bullish Engulfing pattern detection
                if 'bullish_engulfing' in selected_patterns:
                    # Placeholder for future implementation
                    if self.logger:
                        self.logger(f"Bullish Engulfing detection not yet implemented")

                # Piercing pattern detection
                if 'piercing' in selected_patterns:
                    # Placeholder for future implementation
                    if self.logger:
                        self.logger(f"Piercing pattern detection not yet implemented")

                # Morning Star pattern detection
                if 'morning_star' in selected_patterns:
                    # Placeholder for future implementation
                    if self.logger:
                        self.logger(f"Morning Star detection not yet implemented")

                # Doji pattern detection
                if 'doji' in selected_patterns:
                    # Placeholder for future implementation
                    if self.logger:
                        self.logger(f"Doji detection not yet implemented")

                if not patterns_found and self.logger:
                    self.logger(f"No selected patterns found for {symbol}")

            except Exception as e:
                if self.logger:
                    self.logger(f"Error analyzing {symbol}: {str(e)}")

            # Small delay to avoid hammering the API
            time.sleep(0.2)

        # Sort patterns by timestamp (newest first)
        all_patterns.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)

        if self.logger:
            self.logger(f"Found {len(all_patterns)} patterns across all symbols")

        return all_patterns

    def _calculate_quality(self, strength: float) -> str:
        """
        Calculate pattern quality rating based on strength.

        Args:
            strength: Pattern strength value (0.0-1.0)

        Returns:
            Quality rating string ('Good', 'Moderate', or 'Poor')
        """
        if strength >= 0.7:
            return "Good"
        elif strength >= 0.4:
            return "Moderate"
        else:
            return "Poor"

    def _format_pattern_for_display(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format pattern data for display in the UI.

        Args:
            pattern: Raw pattern data

        Returns:
            Formatted pattern data
        """
        display_data = pattern.copy()

        # Format price
        if 'close' in pattern:
            display_data['price_formatted'] = f"${pattern['close']:.4f}"

        # Format timestamp
        if 'timestamp' in pattern and pattern['timestamp']:
            display_data['timestamp_formatted'] = pattern['timestamp'].strftime("%Y-%m-%d %H:%M")
        else:
            display_data['timestamp_formatted'] = "N/A"

        # Format volume
        if 'usd_volume' in pattern:
            display_data['volume_formatted'] = format_volume(pattern['usd_volume'])

        # Format bullish indicator
        if 'is_bullish' in pattern:
            display_data['is_bullish_formatted'] = "Yes" if pattern['is_bullish'] else "No"

        return display_data