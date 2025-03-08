import logging
import time
import sys

# Import modules from src directory
from src.market_data.okx_client import OKXClient
from src.strategies.hh_hl_strategy import analyze_price_action
from src.risk_management.position_sizer import calculate_take_profit, calculate_stop_loss, format_trade_summary


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('crypto_bot.log')
        ]
    )
    return logging.getLogger('main')


def main():
    """Main entry point for the trading bot."""
    logger = setup_logging()
    logger.info("Starting crypto trading analysis...")

    # Initialize market data client
    client = OKXClient()

    # Get available symbols
    symbols = client.get_top_volume_symbols(limit=30)

    if not symbols:
        logger.error("Failed to fetch symbols. Exiting.")
        return

    # Use top volume symbols
    top_symbols = symbols
    logger.info(f"Analyzing top {len(top_symbols)} symbols by volume: {top_symbols}")

    # Results containers
    uptrends = []
    downtrends = []
    no_trends = []

    # Analyze each symbol
    for symbol in top_symbols:
        logger.info(f"Analyzing {symbol}...")

        # Get historical price data
        klines = client.get_klines(symbol, interval="1h", limit=48)

        if not klines:
            logger.warning(f"No data available for {symbol}, skipping")
            continue

        # Extract close prices
        close_prices = [candle["close"] for candle in klines]
        timestamps = [candle["timestamp"] for candle in klines]

        # Apply HH/HL strategy
        result = analyze_price_action(close_prices, smoothing=1, consecutive_count=2, timestamps=timestamps)
        trend = result["trend"]

        # Process results
        if trend == "uptrend":
            hh_count = result["uptrend_analysis"]["consecutive_hh"]
            hl_count = result["uptrend_analysis"]["consecutive_hl"]
            uptrends.append((symbol, hh_count, hl_count))
            logger.info(f"✅ UPTREND: {symbol} - HH: {hh_count}, HL: {hl_count}")

        elif trend == "downtrend":
            lh_count = result["downtrend_analysis"]["consecutive_lh"]
            ll_count = result["downtrend_analysis"]["consecutive_ll"]
            downtrends.append((symbol, lh_count, ll_count))
            logger.info(f"🔻 DOWNTREND: {symbol} - LH: {lh_count}, LL: {ll_count}")

        else:
            no_trends.append(symbol)
            logger.info(f"➖ NO TREND: {symbol}")

    # Display summary
    print("\n=== TREND ANALYSIS SUMMARY ===")
    print(f"Total symbols analyzed: {len(top_symbols)}")

    # Define TP/SL percentages
    tp_percent = 1.0
    sl_percent = 1.0

    if uptrends:
        print(f"\n✅ UPTRENDS ({len(uptrends)}):")
        for symbol, hh, hl in uptrends:
            print(f"  {symbol}: {hh} Higher Highs, {hl} Higher Lows")

            # Get current price and calculate TP/SL for long position
            ticker = client.get_ticker(symbol)
            if ticker and "last_price" in ticker:
                price = ticker["last_price"]
                tp = calculate_take_profit(price, tp_percent)
                sl = calculate_stop_loss(price, sl_percent)

                # Print trade summary
                summary = format_trade_summary(symbol, "BUY", price, tp, sl, tp_percent, sl_percent)
                print(f"  {summary.replace('Trade Summary for', 'BUY Signal for').replace(chr(10), chr(10) + '  ')}")
                print()

    if downtrends:
        print(f"\n🔻 DOWNTRENDS ({len(downtrends)}):")
        for symbol, lh, ll in downtrends:
            print(f"  {symbol}: {lh} Lower Highs, {ll} Lower Lows")

            # Get current price and calculate TP/SL for short position
            ticker = client.get_ticker(symbol)
            if ticker and "last_price" in ticker:
                price = ticker["last_price"]
                # For shorts, TP is lower and SL is higher
                tp = price * (1 - tp_percent / 100)
                sl = price * (1 + sl_percent / 100)

                # Print trade summary
                summary = format_trade_summary(symbol, "SELL", price, tp, sl, tp_percent, sl_percent)
                print(f"  {summary.replace('Trade Summary for', 'SELL Signal for').replace(chr(10), chr(10) + '  ')}")
                print()

    if no_trends:
        print(f"\n➖ NO CLEAR TREND ({len(no_trends)}):")
        for symbol in no_trends:
            print(f"  {symbol}")


if __name__ == "__main__":
    main()