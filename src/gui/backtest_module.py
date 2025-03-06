"""
Backtesting module with tab creator function.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from typing import Dict, List, Any, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import sys
import os
import traceback

# Add the parent directory to the path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtesting.backtest_engine import Backtest
from src.backtesting.data_fetcher import HistoricalDataFetcher, format_backtest_summary
from src.backtesting.strategy_adapters import (
    adapt_hhhl_strategy,
    adapt_candlestick_strategy,
    combined_strategy
)


class BacktestTab:
    """
    Tab for backtesting trading strategies.
    """

    def __init__(self, parent, client=None):
        """
        Initialize the backtesting tab.

        Args:
            parent: Parent frame
            client: OKXClient instance (optional)
        """
        self.parent = parent
        self.client = client
        self.data_fetcher = HistoricalDataFetcher(client=client)

        # Store last backtest results
        self.last_results = None

        # Track running state
        self.running = False
        self.backtest_thread = None

        # Create main frame
        self.frame = ttk.Frame(parent, padding="10")

        # Create widgets
        self.create_widgets()

        # Pack the main frame
        self.frame.pack(fill=tk.BOTH, expand=True)

    def create_widgets(self):
        """Create all widgets for the tab."""
        # Create control panel
        self.create_control_panel()

        # Create results panel
        self.create_results_panel()

    def create_control_panel(self):
        """Create the control panel with input options."""
        control_frame = ttk.LabelFrame(self.frame, text="Backtest Settings", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Create grid for controls
        control_frame.columnconfigure(0, weight=0)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=0)
        control_frame.columnconfigure(3, weight=1)

        # Symbol input
        ttk.Label(control_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.symbol_var = tk.StringVar(value="BTC-USDT")
        symbol_entry = ttk.Entry(control_frame, textvariable=self.symbol_var, width=15)
        symbol_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Timeframe selection
        ttk.Label(control_frame, text="Timeframe:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.timeframe_var = tk.StringVar(value="1h")
        timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var,
                                       values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], width=10)
        timeframe_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Days to backtest
        ttk.Label(control_frame, text="Days:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.days_var = tk.IntVar(value=30)
        days_spinbox = ttk.Spinbox(control_frame, from_=1, to=365, textvariable=self.days_var, width=10)
        days_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Initial capital
        ttk.Label(control_frame, text="Initial Capital ($):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.capital_var = tk.DoubleVar(value=1000.0)
        capital_spinbox = ttk.Spinbox(
            control_frame, from_=100, to=1000000, increment=100, textvariable=self.capital_var, width=10
        )
        capital_spinbox.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Position size percentage
        ttk.Label(control_frame, text="Position Size (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.pos_size_var = tk.DoubleVar(value=10.0)
        pos_size_spinbox = ttk.Spinbox(
            control_frame, from_=1, to=100, increment=1, textvariable=self.pos_size_var, width=10
        )
        pos_size_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Take profit percentage
        ttk.Label(control_frame, text="Take Profit (%):").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        self.tp_var = tk.DoubleVar(value=3.0)
        tp_spinbox = ttk.Spinbox(
            control_frame, from_=0.1, to=100, increment=0.1, textvariable=self.tp_var, width=10
        )
        tp_spinbox.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

        # Stop loss percentage
        ttk.Label(control_frame, text="Stop Loss (%):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.sl_var = tk.DoubleVar(value=2.0)
        sl_spinbox = ttk.Spinbox(
            control_frame, from_=0.1, to=100, increment=0.1, textvariable=self.sl_var, width=10
        )
        sl_spinbox.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        # Strategy selection
        ttk.Label(control_frame, text="Strategy:").grid(row=3, column=2, sticky=tk.W, padx=5, pady=5)
        self.strategy_var = tk.StringVar(value="HH/HL")
        strategy_combo = ttk.Combobox(
            control_frame, textvariable=self.strategy_var,
            values=["HH/HL", "Candlestick", "Combined"], width=10
        )
        strategy_combo.grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)

        # Strategy options frame
        strategy_options_frame = ttk.LabelFrame(control_frame, text="Strategy Options", padding="5")
        strategy_options_frame.grid(row=4, column=0, columnspan=4, sticky=tk.W + tk.E, padx=5, pady=5)

        # HH/HL options
        self.hhhl_consecutive_var = tk.IntVar(value=2)
        ttk.Label(strategy_options_frame, text="HH/HL Consecutive:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        hhhl_consecutive_spinbox = ttk.Spinbox(
            strategy_options_frame, from_=1, to=5, textvariable=self.hhhl_consecutive_var, width=5
        )
        hhhl_consecutive_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Candlestick options
        self.cs_patterns_frame = ttk.Frame(strategy_options_frame)
        self.cs_patterns_frame.grid(row=0, column=2, rowspan=2, sticky=tk.W, padx=20, pady=2)

        ttk.Label(self.cs_patterns_frame, text="Patterns:").grid(row=0, column=0, sticky=tk.W)

        # Pattern checkboxes
        self.pattern_vars = {
            'hammer': tk.BooleanVar(value=True),
            'bullish_engulfing': tk.BooleanVar(value=True),
            'piercing': tk.BooleanVar(value=False),
            'morning_star': tk.BooleanVar(value=False)
        }

        ttk.Checkbutton(
            self.cs_patterns_frame, text="Hammer", variable=self.pattern_vars['hammer']
        ).grid(row=1, column=0, sticky=tk.W)

        ttk.Checkbutton(
            self.cs_patterns_frame, text="Bullish Engulfing", variable=self.pattern_vars['bullish_engulfing']
        ).grid(row=2, column=0, sticky=tk.W)

        ttk.Checkbutton(
            self.cs_patterns_frame, text="Piercing", variable=self.pattern_vars['piercing']
        ).grid(row=3, column=0, sticky=tk.W)

        ttk.Checkbutton(
            self.cs_patterns_frame, text="Morning Star", variable=self.pattern_vars['morning_star']
        ).grid(row=4, column=0, sticky=tk.W)

        # Candlestick min strength
        self.cs_min_strength_var = tk.DoubleVar(value=0.3)
        ttk.Label(strategy_options_frame, text="Min Strength:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        cs_min_strength_spinbox = ttk.Spinbox(
            strategy_options_frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.cs_min_strength_var, width=5
        )
        cs_min_strength_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Confirmation options
        self.volume_confirmation_var = tk.BooleanVar(value=False)
        self.prior_trend_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            strategy_options_frame, text="Volume Confirmation", variable=self.volume_confirmation_var
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(
            strategy_options_frame, text="Prior Trend Required", variable=self.prior_trend_var
        ).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)

        # Add run button
        run_frame = ttk.Frame(control_frame)
        run_frame.grid(row=5, column=0, columnspan=4, sticky=tk.E, padx=5, pady=10)

        self.run_button = ttk.Button(run_frame, text="Run Backtest", command=self.run_backtest)
        self.run_button.pack(side=tk.RIGHT)

        # Progress indicator
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(run_frame, textvariable=self.progress_var)
        progress_label.pack(side=tk.RIGHT, padx=10)

    def create_results_panel(self):
        """Create the panel for displaying backtest results."""
        results_frame = ttk.LabelFrame(self.frame, text="Backtest Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create notebook for results tabs
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.summary_frame, text="Summary")

        # Create summary text widget
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Equity curve tab
        self.equity_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.equity_frame, text="Equity Curve")

        # Trade details tab
        self.trades_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.trades_frame, text="Trades")

        # Create trades treeview
        self.create_trades_treeview(self.trades_frame)

        # Log tab
        self.log_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.log_frame, text="Log")

        # Create log text widget
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def create_trades_treeview(self, parent):
        """Create a treeview to display trade details."""
        # Create frame for treeview and scrollbar
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create treeview
        columns = ("Entry Time", "Side", "Entry Price", "Exit Time", "Exit Price",
                   "Profit %", "Profit $", "Pattern", "Exit Reason")
        self.trades_treeview = ttk.Treeview(
            frame, columns=columns, show="headings", yscrollcommand=scrollbar.set
        )

        # Configure columns
        for column in columns:
            self.trades_treeview.heading(column, text=column)
            width = 100 if column in ["Entry Time", "Exit Time"] else 80
            self.trades_treeview.column(column, width=width)

        # Configure scrollbar
        scrollbar.config(command=self.trades_treeview.yview)

        # Pack treeview
        self.trades_treeview.pack(fill=tk.BOTH, expand=True)

        # Configure tags for win/loss
        self.trades_treeview.tag_configure('profit', background='#d4edda')  # Light green
        self.trades_treeview.tag_configure('loss', background='#f8d7da')  # Light red

    def log(self, message):
        """Add a message to the log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Insert log message in a thread-safe way
        self.parent.after(0, lambda: self.log_text.insert(tk.END, log_message))
        self.parent.after(0, lambda: self.log_text.see(tk.END))

    def update_progress(self, message):
        """Update the progress indicator."""
        self.progress_var.set(message)

    def _get_selected_strategy(self):
        """Get the selected strategy function based on user options."""
        strategy_type = self.strategy_var.get()

        if strategy_type == "HH/HL":
            consecutive = self.hhhl_consecutive_var.get()
            return lambda candles: adapt_hhhl_strategy(candles, consecutive_count=consecutive)

        elif strategy_type == "Candlestick":
            # Get selected patterns
            patterns = [
                pattern for pattern, var in self.pattern_vars.items() if var.get()
            ]

            if not patterns:
                patterns = ['hammer']  # Default if none selected

            min_strength = self.cs_min_strength_var.get()
            volume_conf = self.volume_confirmation_var.get()
            prior_trend = self.prior_trend_var.get()

            return lambda candles: adapt_candlestick_strategy(
                candles, pattern_types=patterns,
                min_strength=min_strength,
                volume_confirmation=volume_conf,
                prior_trend=prior_trend
            )

        elif strategy_type == "Combined":
            # Create both strategies and combined them
            consecutive = self.hhhl_consecutive_var.get()
            hhhl_strategy = lambda candles: adapt_hhhl_strategy(candles, consecutive_count=consecutive)

            patterns = [
                pattern for pattern, var in self.pattern_vars.items() if var.get()
            ]

            if not patterns:
                patterns = ['hammer']  # Default if none selected

            min_strength = self.cs_min_strength_var.get()
            volume_conf = self.volume_confirmation_var.get()
            prior_trend = self.prior_trend_var.get()

            cs_strategy = lambda candles: adapt_candlestick_strategy(
                candles, pattern_types=patterns,
                min_strength=min_strength,
                volume_confirmation=volume_conf,
                prior_trend=prior_trend
            )

            # Return combined strategy with equal weights
            return lambda candles: combined_strategy(
                candles, strategies=[hhhl_strategy, cs_strategy], weights=[0.5, 0.5]
            )

        # Default to HH/HL if something goes wrong
        return adapt_hhhl_strategy

    def run_backtest(self):
        """Run the backtest with the selected settings."""
        if self.running:
            messagebox.showinfo("Running", "Backtest is already running. Please wait.")
            return

        # Clear previous results
        self.summary_text.delete(1.0, tk.END)
        self.clear_equity_chart()
        self.clear_trades_list()
        self.log_text.delete(1.0, tk.END)

        # Get settings
        symbol = self.symbol_var.get()
        timeframe = self.timeframe_var.get()
        days = self.days_var.get()
        initial_capital = self.capital_var.get()
        pos_size_pct = self.pos_size_var.get()
        tp_pct = self.tp_var.get()
        sl_pct = self.sl_var.get()

        # Validate inputs
        if not symbol or not timeframe:
            messagebox.showerror("Error", "Symbol and timeframe are required.")
            return

        # Set running state
        self.running = True
        self.update_progress("Running...")
        self.run_button.config(state=tk.DISABLED)

        # Start backtest thread
        self.backtest_thread = threading.Thread(
            target=self.backtest_task,
            args=(
                symbol, timeframe, days, initial_capital,
                pos_size_pct, tp_pct, sl_pct
            )
        )
        self.backtest_thread.daemon = True
        self.backtest_thread.start()

    def backtest_task(self, symbol, timeframe, days, initial_capital, pos_size_pct, tp_pct, sl_pct):
        """Run the backtest task in a separate thread."""
        try:
            self.log(f"Starting backtest for {symbol} ({timeframe}, {days} days)...")
            self.log(
                f"Capital: ${initial_capital:.2f}, Position size: {pos_size_pct:.1f}%, TP: {tp_pct:.1f}%, SL: {sl_pct:.1f}%")

            # Update progress
            self.update_progress(f"Fetching data for {symbol}...")

            # Fetch historical data
            historical_data = self.data_fetcher.fetch_data(
                symbol, timeframe=timeframe, days=days, logger=self.log
            )

            if not historical_data:
                self.log(f"No data available for {symbol}. Backtest aborted.")
                self.update_progress("Failed - No data")
                self.run_button.config(state=tk.NORMAL)
                self.running = False
                return

            # Get selected strategy
            self.update_progress("Preparing strategy...")
            strategy_func = self._get_selected_strategy()

            # Initialize backtest
            self.update_progress("Running backtest...")
            backtest = Backtest(
                symbol=symbol,
                historical_data=historical_data,
                initial_capital=initial_capital,
                position_size_pct=pos_size_pct,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                logger=self.log
            )

            # Run backtest
            results = backtest.run_strategy(strategy_func)

            # Store results
            self.last_results = results

            # Update UI with results
            self.update_backtest_results(results)

            # Done
            self.update_progress(f"Done - Return: {results.get('total_return_pct', 0):.2f}%")

        except Exception as e:
            self.log(f"Error during backtest: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            self.update_progress("Failed - Error")

        finally:
            # Reset running state
            self.run_button.config(state=tk.NORMAL)
            self.running = False

    def update_backtest_results(self, results):
        """Update the UI with backtest results."""
        if not results:
            return

        # Update summary
        summary_text = format_backtest_summary(results)
        self.parent.after(0, lambda: self.summary_text.delete(1.0, tk.END))
        self.parent.after(0, lambda: self.summary_text.insert(tk.END, summary_text))

        # Update equity chart
        self.parent.after(0, lambda: self.update_equity_chart(results))

        # Update trades list
        self.parent.after(0, lambda: self.update_trades_list(results))

    def clear_equity_chart(self):
        """Clear the equity chart."""
        for widget in self.equity_frame.winfo_children():
            widget.destroy()

    def update_equity_chart(self, results):
        """Update the equity chart with backtest results."""
        # Clear any existing chart
        self.clear_equity_chart()

        if not results or 'equity_curve' not in results:
            return

        # Convert equity curve to DataFrame
        equity_data = pd.DataFrame(results['equity_curve'])

        if equity_data.empty:
            return

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot equity curve
        ax.plot(range(len(equity_data)), equity_data['equity'], label='Equity', color='blue')

        # Add initial capital line
        ax.axhline(y=results['initial_capital'], color='gray', linestyle='--', label='Initial Capital')

        # Customize chart
        ax.set_title(f"Equity Curve - {results['symbol']}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis labels for readability
        if len(equity_data) > 20:
            step = len(equity_data) // 10
            ax.set_xticks(range(0, len(equity_data), step))

            # If timestamp is available, use it for labels
            if 'timestamp' in equity_data.columns:
                # Use first 10 characters if timestamp is string (date only)
                if isinstance(equity_data['timestamp'].iloc[0], str):
                    labels = [str(equity_data['timestamp'].iloc[i])[:10] for i in range(0, len(equity_data), step)]
                    ax.set_xticklabels(labels, rotation=45)

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=self.equity_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_trades_list(self):
        """Clear the trades list."""
        for item in self.trades_treeview.get_children():
            self.trades_treeview.delete(item)

    def update_trades_list(self, results):
        """Update the trades list with backtest results."""
        self.clear_trades_list()

        if not results or 'trades' not in results:
            return

        trades = results['trades']

        for trade in trades:
            # Format values
            entry_time = trade.get('entry_time', '')
            side = trade.get('side', '')
            entry_price = f"${trade.get('entry_price', 0):.4f}"
            exit_time = trade.get('exit_time', '')
            exit_price = f"${trade.get('exit_price', 0):.4f}"
            profit_pct = f"{trade.get('profit_pct', 0):.2f}%"
            profit_amount = f"${trade.get('profit_amount', 0):.2f}"
            pattern = trade.get('pattern', 'Unknown')
            exit_reason = trade.get('exit_reason', '')

            # Determine tag based on profit
            tag = 'profit' if trade.get('profit_amount', 0) > 0 else 'loss'

            # Insert into treeview
            self.trades_treeview.insert(
                "", tk.END,
                values=(entry_time, side, entry_price, exit_time, exit_price,
                        profit_pct, profit_amount, pattern, exit_reason),
                tags=(tag,)
            )


def create_backtest_tab(notebook, client=None):
    """
    Create and add a backtesting tab to a notebook.

    Args:
        notebook: ttk.Notebook widget
        client: OKXClient instance (optional)

    Returns:
        BacktestTab instance
    """
    backtest_frame = ttk.Frame(notebook)
    backtest_tab = BacktestTab(backtest_frame, client=client)
    notebook.add(backtest_frame, text="Backtest")
    return backtest_tab