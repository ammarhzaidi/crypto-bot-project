"""
Simple position sizing module with basic risk management functions.
"""


def calculate_take_profit(entry_price: float, tp_percent: float = 1.0) -> float:
    """
    Calculate take profit level based on entry price and percentage.

    Args:
        entry_price: The entry price
        tp_percent: Take profit percentage (default: 1.0%)

    Returns:
        Take profit price level
    """
    return entry_price * (1 + tp_percent / 100)


def calculate_stop_loss(entry_price: float, sl_percent: float = 1.0) -> float:
    """
    Calculate stop loss level based on entry price and percentage.

    Args:
        entry_price: The entry price
        sl_percent: Stop loss percentage (default: 1.0%)

    Returns:
        Stop loss price level
    """
    return entry_price * (1 - sl_percent / 100)


def format_trade_summary(symbol: str, side: str, entry_price: float,
                         tp_price: float, sl_price: float, tp_percent: float = 1.0,
                         sl_percent: float = 1.0) -> str:
    """
    Format trade details into a readable string.

    Args:
        symbol: Trading pair symbol
        side: Trade direction ('BUY' or 'SELL')
        entry_price: Entry price
        tp_price: Take profit price
        sl_price: Stop loss price
        tp_percent: Take profit percentage
        sl_percent: Stop loss percentage

    Returns:
        Formatted trade summary
    """
    return (f"Trade Summary for {symbol} ({side}):\n"
            f"Entry Price: ${entry_price:.4f}\n"
            f"Take Profit: ${tp_price:.4f} (+{tp_percent}%)\n"
            f"Stop Loss:   ${sl_price:.4f} (-{sl_percent}%)")


# Example usage
if __name__ == "__main__":
    # Example parameters
    symbol = "BTC-USDT"
    entry_price = 50000.0
    side = "BUY"
    tp_percent = 1.0
    sl_percent = 1.0

    # Calculate levels
    tp_price = calculate_take_profit(entry_price, tp_percent)
    sl_price = calculate_stop_loss(entry_price, sl_percent)

    # Format and print summary
    summary = format_trade_summary(symbol, side, entry_price, tp_price, sl_price, tp_percent, sl_percent)
    print(summary)