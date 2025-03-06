"""
Simple backtesting engine for cryptocurrency trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple, Callable, Optional


class Backtest:
    """
    A simple backtesting engine for cryptocurrency trading strategies.
    """

    def __init__(
            self,
            symbol: str,
            historical_data: List[Dict[str, Any]],
            initial_capital: float = 1000.0,
            position_size_pct: float = 10.0,  # Use 10% of capital per trade by default
            tp_pct: float = 3.0,
            sl_pct: float = 2.0,
            max_trades: int = 100,
            logger: Optional[Callable] = None
    ):
        """
        Initialize the backtesting engine.

        Args:
            symbol: Trading pair symbol
            historical_data: List of historical price data dictionaries
            initial_capital: Starting capital amount
            position_size_pct: Percentage of capital to use per trade
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            max_trades: Maximum number of trades to simulate
            logger: Optional logging function
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_trades = max_trades
        self.logger = logger

        # Convert historical data to DataFrame for easier processing
        self.df = pd.DataFrame(historical_data)

        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"Historical data must contain these columns: {required_cols}")

        # Ensure data is sorted chronologically
        self.df.sort_values('timestamp', inplace=True)

        # Initialize tracking variables
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.trades_count = 0

        if self.logger:
            self.logger(f"Backtest initialized for {symbol} with {len(historical_data)} candles")
            self.logger(f"Initial capital: ${initial_capital:.2f}, Position size: {position_size_pct:.1f}%")
            self.logger(f"Take profit: {tp_pct:.1f}%, Stop loss: {sl_pct:.1f}%")

    def run_strategy(self, strategy_func: Callable, lookback: int = 50) -> Dict[str, Any]:
        """
        Run a trading strategy on historical data.

        Args:
            strategy_func: Strategy function that returns signals
            lookback: Number of candles required for strategy calculations

        Returns:
            Dictionary with backtest results
        """
        start_time = time.time()

        if self.logger:
            self.logger(f"Starting backtest with lookback period of {lookback} candles")

        # Reset tracking variables
        self.trades = []
        self.equity_curve = [{'timestamp': self.df.iloc[0]['timestamp'], 'equity': self.initial_capital}]
        self.capital = self.initial_capital
        self.current_position = None
        self.trades_count = 0

        # For each candle in the historical data (except the lookback period)
        for i in range(lookback, len(self.df)):
            # Get current candle data
            current_candle = self.df.iloc[i].to_dict()
            current_idx = i
            current_time = current_candle['timestamp']
            current_price = current_candle['close']

            # Update equity curve
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': self.capital if self.current_position is None else
                self.capital + self._calculate_position_value(self.current_position, current_price)
            })

            # Check if we have an open position that needs to be closed
            if self.current_position is not None:
                # Calculate current profit/loss
                entry_price = self.current_position['entry_price']
                position_size = self.current_position['position_size']
                side = self.current_position['side']

                if side == 'BUY':  # Long position
                    profit_pct = ((current_price / entry_price) - 1) * 100
                    tp_price = entry_price * (1 + self.tp_pct / 100)
                    sl_price = entry_price * (1 - self.sl_pct / 100)

                    # Check if take profit hit
                    if current_candle['high'] >= tp_price:
                        self._close_position(current_time, tp_price, 'Take Profit')
                        if self.logger:
                            self.logger(f"Take profit hit at {tp_price:.4f} for {side} position from {entry_price:.4f}")

                    # Check if stop loss hit
                    elif current_candle['low'] <= sl_price:
                        self._close_position(current_time, sl_price, 'Stop Loss')
                        if self.logger:
                            self.logger(f"Stop loss hit at {sl_price:.4f} for {side} position from {entry_price:.4f}")

                else:  # Short position
                    profit_pct = ((entry_price / current_price) - 1) * 100
                    tp_price = entry_price * (1 - self.tp_pct / 100)
                    sl_price = entry_price * (1 + self.sl_pct / 100)

                    # Check if take profit hit
                    if current_candle['low'] <= tp_price:
                        self._close_position(current_time, tp_price, 'Take Profit')
                        if self.logger:
                            self.logger(f"Take profit hit at {tp_price:.4f} for {side} position from {entry_price:.4f}")

                    # Check if stop loss hit
                    elif current_candle['high'] >= sl_price:
                        self._close_position(current_time, sl_price, 'Stop Loss')
                        if self.logger:
                            self.logger(f"Stop loss hit at {sl_price:.4f} for {side} position from {entry_price:.4f}")

            # Check if we've reached the maximum number of trades
            if self.trades_count >= self.max_trades:
                if self.logger:
                    self.logger(f"Maximum trade count reached ({self.max_trades}). Stopping backtest.")
                break

            # If we don't have an open position, check for new signals
            if self.current_position is None:
                # Get data for the lookback period up to the current candle
                lookback_data = self.df.iloc[i - lookback:i + 1].to_dict('records')

                # Get signals from the strategy function
                signal = strategy_func(lookback_data)

                # Process signals
                if signal and 'side' in signal:
                    side = signal['side']
                    # Only process valid BUY or SELL signals
                    if side in ['BUY', 'SELL']:
                        # Calculate position size based on capital
                        position_capital = self.capital * (self.position_size_pct / 100)
                        position_size = position_capital / current_price

                        # Open new position
                        self.current_position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'side': side,
                            'position_size': position_size,
                            'position_capital': position_capital,
                            'pattern': signal.get('pattern', 'Unknown'),
                            'strength': signal.get('strength', 0)
                        }

                        if self.logger:
                            self.logger(f"Opening {side} position at {current_price:.4f} with ${position_capital:.2f}")

        # Close any remaining position at the end
        if self.current_position is not None:
            last_candle = self.df.iloc[-1]
            self._close_position(last_candle['timestamp'], last_candle['close'], 'End of Backtest')

        # Calculate results
        results = self._calculate_results()
        results['execution_time'] = time.time() - start_time

        if self.logger:
            self.logger(f"Backtest completed in {results['execution_time']:.2f} seconds")
            self.logger(f"Final capital: ${results['final_capital']:.2f} ({results['total_return_pct']:.2f}%)")
            self.logger(f"Total trades: {results['total_trades']}")
            self.logger(f"Win rate: {results['win_rate']:.2f}%")

        return results

    def _close_position(self, exit_time, exit_price, exit_reason):
        """
        Close the current position and record the trade.
        """
        if self.current_position is None:
            return

        entry_price = self.current_position['entry_price']
        position_size = self.current_position['position_size']
        position_capital = self.current_position['position_capital']
        side = self.current_position['side']

        # Calculate profit/loss
        if side == 'BUY':  # Long position
            profit_pct = ((exit_price / entry_price) - 1) * 100
            profit_amount = position_capital * (profit_pct / 100)
        else:  # Short position
            profit_pct = ((entry_price / exit_price) - 1) * 100
            profit_amount = position_capital * (profit_pct / 100)

        # Update capital
        self.capital += profit_amount

        # Record trade
        trade = {
            'entry_time': self.current_position['entry_time'],
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'side': side,
            'position_size': position_size,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'exit_reason': exit_reason,
            'pattern': self.current_position.get('pattern', 'Unknown'),
            'strength': self.current_position.get('strength', 0)
        }

        self.trades.append(trade)
        self.trades_count += 1

        # Clear current position
        self.current_position = None

    def _calculate_position_value(self, position, current_price):
        """
        Calculate the current value of an open position.
        """
        if position is None:
            return 0

        entry_price = position['entry_price']
        position_size = position['position_size']
        side = position['side']

        if side == 'BUY':  # Long position
            return position_size * (current_price - entry_price)
        else:  # Short position
            return position_size * (entry_price - current_price)

    def _calculate_results(self):
        """
        Calculate performance metrics from the backtest.
        """
        results = {}

        # Basic metrics
        results['symbol'] = self.symbol
        results['initial_capital'] = self.initial_capital
        results['final_capital'] = self.equity_curve[-1]['equity']
        results['absolute_return'] = results['final_capital'] - self.initial_capital
        results['total_return_pct'] = ((results['final_capital'] / self.initial_capital) - 1) * 100

        # Equity curve
        results['equity_curve'] = self.equity_curve

        # Trade statistics
        results['trades'] = self.trades
        results['total_trades'] = len(self.trades)

        if results['total_trades'] > 0:
            # Calculate win rate
            winning_trades = [t for t in self.trades if t['profit_amount'] > 0]
            results['winning_trades'] = len(winning_trades)
            results['win_rate'] = (results['winning_trades'] / results['total_trades']) * 100

            # Calculate average profit
            if results['winning_trades'] > 0:
                results['avg_win_pct'] = sum(t['profit_pct'] for t in winning_trades) / results['winning_trades']
                results['avg_win_amount'] = sum(t['profit_amount'] for t in winning_trades) / results['winning_trades']
            else:
                results['avg_win_pct'] = 0
                results['avg_win_amount'] = 0

            # Calculate average loss
            losing_trades = [t for t in self.trades if t['profit_amount'] <= 0]
            if len(losing_trades) > 0:
                results['avg_loss_pct'] = sum(t['profit_pct'] for t in losing_trades) / len(losing_trades)
                results['avg_loss_amount'] = sum(t['profit_amount'] for t in losing_trades) / len(losing_trades)
            else:
                results['avg_loss_pct'] = 0
                results['avg_loss_amount'] = 0

            # Calculate profit factor (if there are losses)
            total_wins = sum(t['profit_amount'] for t in winning_trades)
            total_losses = abs(sum(t['profit_amount'] for t in losing_trades))

            if total_losses > 0:
                results['profit_factor'] = total_wins / total_losses
            else:
                results['profit_factor'] = float('inf') if total_wins > 0 else 0

            # Calculate maximum drawdown
            cumulative_returns = [e['equity'] for e in self.equity_curve]

            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)

            # Calculate drawdown in percentage terms
            drawdown = (running_max - cumulative_returns) / running_max * 100

            results['max_drawdown_pct'] = drawdown.max()
            results['max_drawdown_amount'] = (running_max - cumulative_returns).max()

            # Count consecutive wins/losses
            if self.trades:
                consecutive_wins = 0
                consecutive_losses = 0
                current_streak = 0

                for i, trade in enumerate(self.trades):
                    if trade['profit_amount'] > 0:  # Winning trade
                        if i > 0 and self.trades[i - 1]['profit_amount'] > 0:  # Previous trade was also winning
                            current_streak += 1
                        else:
                            current_streak = 1

                        consecutive_wins = max(consecutive_wins, current_streak)
                    else:  # Losing trade
                        if i > 0 and self.trades[i - 1]['profit_amount'] <= 0:  # Previous trade was also losing
                            current_streak -= 1
                        else:
                            current_streak = -1

                        consecutive_losses = min(consecutive_losses, current_streak)

                results['max_consecutive_wins'] = consecutive_wins
                results['max_consecutive_losses'] = abs(consecutive_losses)
        else:
            # No trades executed
            results['winning_trades'] = 0
            results['win_rate'] = 0
            results['avg_win_pct'] = 0
            results['avg_win_amount'] = 0
            results['avg_loss_pct'] = 0
            results['avg_loss_amount'] = 0
            results['profit_factor'] = 0
            results['max_drawdown_pct'] = 0
            results['max_drawdown_amount'] = 0
            results['max_consecutive_wins'] = 0
            results['max_consecutive_losses'] = 0

        return results


# Example of a simple strategy function
def simple_strategy(candles: List[Dict]) -> Dict:
    """
    Example strategy function. Returns a signal if the last candle closed higher than the previous one.

    Args:
        candles: List of candle data dictionaries

    Returns:
        Signal dictionary or None if no signal
    """
    if len(candles) < 2:
        return None

    last_candle = candles[-1]
    prev_candle = candles[-2]

    if last_candle['close'] > prev_candle['close']:
        return {
            'side': 'BUY',
            'pattern': 'Price Up',
            'strength': 0.5
        }

    return None