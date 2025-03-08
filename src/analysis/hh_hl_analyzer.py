"""
Analyzer for Higher Highs/Higher Lows (HH/HL) patterns.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import time
from datetime import datetime
from src.market_data.okx_client import OKXClient
from src.strategies.hh_hl_strategy import analyze_price_action
from src.risk_management.position_sizer import calculate_take_profit, calculate_stop_loss
from src.analysis.freshness_calculator import calculate_freshness, format_volume


class HHHLAnalyzer:
    """
    Analyzer for detecting Higher Highs/Higher Lows (HH/HL) patterns.
    """

    def __init__(self, check_freshness: bool = False, logger: Optional[Callable] = None):
        """
        Initialize the HH/HL analyzer.

        Args:
            check_freshness: Whether to calculate and sort by freshness
            logger: Optional logger function for analysis progress
        """
        self.check_freshness = check_freshness
        self.logger = logger

    def analyze_symbols(self,
                        symbols: List[str],
                        client: OKXClient,
                        tp_percent: float = 1.0,
                        sl_percent: float = 1.0,
                        timeframe: str = "1h",
                        candles_count: int = 48) -> Tuple[List[Dict], List[Dict], List[str]]:

        """
        Analyze multiple symbols for HH/HL patterns.

        Args:
            symbols: List of symbols to analyze
            client: OKXClient instance for data fetching
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
            timeframe: Candlestick timeframe
            candles_count: Number of candles to analyze

        Returns:
            Tuple of (uptrends, downtrends, no_trends)
        """
        uptrends = []
        downtrends = []
        no_trends = []

        if self.logger:
            self.logger(f"Analyzing {len(symbols)} symbols for HH/HL patterns...")

        for i, symbol in enumerate(symbols):
            if self.logger:
                self.logger(f"Analyzing {symbol} ({i + 1}/{len(symbols)})...")

            # Get historical price data
            klines = client.get_klines(symbol, interval=timeframe, limit=candles_count)

            if not klines:
                if self.logger:
                    self.logger(f"No data available for {symbol}, skipping")
                continue

            # Extract close prices
            close_prices = [candle["close"] for candle in klines]
            timestamps = [candle["timestamp"] for candle in klines]

            # Apply HH/HL strategy
            result = analyze_price_action(close_prices, smoothing=1, consecutive_count=2, timestamps=timestamps)
            trend = result["trend"]

            # Get current price
            ticker = client.get_ticker(symbol)
            if not ticker or "last_price" not in ticker:
                if self.logger:
                    self.logger(f"Could not get current price for {symbol}, skipping")
                continue

            current_price = ticker["last_price"]

            # Calculate USD volume
            volume_24h = ticker.get("volume_24h", 0) * current_price
            volume_formatted = format_volume(volume_24h)

            # Process results based on trend
            if trend == "uptrend":
                hh_count = result["uptrend_analysis"]["consecutive_hh"]
                hl_count = result["uptrend_analysis"]["consecutive_hl"]
                pattern = f"{hh_count} HH, {hl_count} HL"

                # Calculate pattern strength
                pattern_strength = min(hh_count, hl_count)

                # Calculate TP/SL
                tp = calculate_take_profit(current_price, tp_percent)
                sl = calculate_stop_loss(current_price, sl_percent)

                # Create uptrend data
                uptrend_data = {
                    'symbol': symbol,
                    'pattern': pattern,
                    'price': current_price,
                    'tp': tp,
                    'sl': sl,
                    'side': 'BUY',
                    'strength': pattern_strength,
                    'volume': volume_24h,
                    'volume_formatted': volume_formatted
                }

                # Add freshness if required
                if self.check_freshness:
                    freshness = calculate_freshness(
                        close_prices,
                        result["uptrend_analysis"]["peaks"],
                        result["uptrend_analysis"]["troughs"]
                    )
                    uptrend_data['freshness'] = freshness

                uptrends.append(uptrend_data)

                if self.logger:
                    prefix = "✅✅✅" if pattern_strength >= 3 else "✅"
                    self.logger(f"{prefix} UPTREND: {symbol} - {pattern}")

            elif trend == "downtrend":
                lh_count = result["downtrend_analysis"]["consecutive_lh"]
                ll_count = result["downtrend_analysis"]["consecutive_ll"]
                pattern = f"{lh_count} LH, {ll_count} LL"

                # Calculate pattern strength
                pattern_strength = min(lh_count, ll_count)

                # For shorts, TP is lower and SL is higher
                tp = current_price * (1 - tp_percent / 100)
                sl = current_price * (1 + sl_percent / 100)

                # Create downtrend data
                downtrend_data = {
                    'symbol': symbol,
                    'pattern': pattern,
                    'price': current_price,
                    'tp': tp,
                    'sl': sl,
                    'side': 'SELL',
                    'strength': pattern_strength,
                    'volume': volume_24h,
                    'volume_formatted': volume_formatted
                }

                # Add freshness if required
                if self.check_freshness:
                    freshness = calculate_freshness(
                        close_prices,
                        result["downtrend_analysis"]["peaks"],
                        result["downtrend_analysis"]["troughs"]
                    )
                    downtrend_data['freshness'] = freshness

                downtrends.append(downtrend_data)

                if self.logger:
                    prefix = "🔻🔻🔻" if pattern_strength >= 3 else "🔻"
                    self.logger(f"{prefix} DOWNTREND: {symbol} - {pattern}")

            else:
                # No clear trend
                no_trends.append(symbol)
                if self.logger:
                    self.logger(f"➖ NO TREND: {symbol}")

            # Small delay to avoid hammering the API
            time.sleep(0.1)

        # Sort results
        if self.check_freshness:
            # Sort by freshness first, then by strength
            uptrends = sorted(uptrends, key=lambda x: (x['freshness'], -x['strength']))
            downtrends = sorted(downtrends, key=lambda x: (x['freshness'], -x['strength']))
        else:
            # Sort by pattern strength only
            uptrends = sorted(uptrends, key=lambda x: x['strength'], reverse=True)
            downtrends = sorted(downtrends, key=lambda x: x['strength'], reverse=True)

        return uptrends, downtrends, no_trends