"""
Visualization utilities for Fibonacci levels in the trading GUI.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.dates as mdates
from datetime import datetime

from src.strategies.fibonacci_levels import FibonacciAnalyzer, find_fibonacci_zones


class FibonacciVisualizer:
    """
    Component for visualizing Fibonacci levels in the trading GUI.
    """

    def __init__(self, parent_frame):
        """
        Initialize the Fibonacci visualizer.

        Args:
            parent_frame: Parent Tkinter frame
        """
        self.parent = parent_frame
        self.fig = None
        self.canvas = None
        self.fig_agg = None
        self.current_zones = None

        # Create frame for the visualization
        self.frame = ttk.Frame(parent_frame)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create control frame
        self.create_controls()

        # Create the figure
        self.create_figure()

    def create_controls(self):
        """Create control widgets for the visualizer."""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)

        # Price range controls
        ttk.Label(control_frame, text="Start Index:").pack(side=tk.LEFT, padx=5)
        self.start_var = tk.IntVar(value=0)
        ttk.Spinbox(control_frame, from_=0, to=1000, increment=10,
                    textvariable=self.start_var, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="End Index:").pack(side=tk.LEFT, padx=5)
        self.end_var = tk.IntVar(value=100)
        ttk.Spinbox(control_frame, from_=10, to=1000, increment=10,
                    textvariable=self.end_var, width=8).pack(side=tk.LEFT, padx=5)

        # Update button
        ttk.Button(control_frame, text="Update View",
                   command=self.update_plot).pack(side=tk.LEFT, padx=20)

        # Information label for current Fibonacci level
        self.info_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.info_var, font=('TkDefaultFont', 10, 'bold')).pack(
            side=tk.RIGHT, padx=10)

    def create_figure(self):
        """Create matplotlib figure for visualization."""
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.tight_layout()

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Initial plot
        self.ax.set_title("Price Chart with Fibonacci Levels")
        self.ax.set_ylabel("Price")
        self.ax.set_xlabel("Candle Index")
        self.ax.grid(True, alpha=0.3)

        # Draw the canvas
        self.canvas.draw()

    def plot_fibonacci_levels(self, prices: List[float],
                              start_idx: Optional[int] = None,
                              end_idx: Optional[int] = None):
        """
        Plot price data with Fibonacci retracement and extension levels.

        Args:
            prices: List of price values
            start_idx: Starting index (default is 0)
            end_idx: Ending index (default is len(prices))
        """
        # Clear the previous plot
        self.ax.clear()

        # Determine start and end indices
        if start_idx is None:
            start_idx = 0
        if end_idx is None or end_idx > len(prices):
            end_idx = len(prices)

        # Get the price slice to display
        display_prices = prices[start_idx:end_idx]
        display_indices = list(range(start_idx, end_idx))

        # Plot the price data
        self.ax.plot(display_indices, display_prices, 'b-', label='Price')

        # Find Fibonacci zones for the entire price series
        fib_zones = find_fibonacci_zones(prices)
        self.current_zones = fib_zones

        if fib_zones["has_zones"]:
            # Get the trend info
            trend = fib_zones["zones"]["trend"]
            is_uptrend = trend == "uptrend"

            # Highlight key swing points if visible
            swing_high = fib_zones["zones"]["swing_high"]
            swing_low = fib_zones["zones"]["swing_low"]

            # Add horizontal lines for Fibonacci retracement levels
            self.plot_retracement_levels(fib_zones["zones"]["retracement"],
                                         display_indices, is_uptrend)

            # Add horizontal lines for Fibonacci extension levels
            self.plot_extension_levels(fib_zones["zones"]["extension"],
                                       display_indices, is_uptrend)

            # Add current price marker
            current_price = fib_zones["current_price"]
            self.ax.axhline(y=current_price, color='k', linestyle='-', alpha=0.5, linewidth=1)

            # Add warning if current price is at key level
            if fib_zones["zones"]["is_at_key_level"]:
                level = fib_zones["zones"]["closest_level"]
                level_price = fib_zones["zones"]["retracement"][level]

                # Highlight the key level
                self.ax.axhline(y=level_price, color='r', linestyle='-', alpha=0.7, linewidth=2)

                # Update info label
                self.info_var.set(f"Price at key Fibonacci level: {level}")
            else:
                self.info_var.set(f"Trend: {trend.capitalize()}")
        else:
            self.info_var.set("No clear Fibonacci levels detected")

        # Configure axes
        self.ax.set_title("Price Chart with Fibonacci Levels")
        self.ax.set_ylabel("Price")
        self.ax.set_xlabel("Candle Index")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

        # Refresh the canvas
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_retracement_levels(self, levels: Dict[float, float],
                                indices: List[int], is_uptrend: bool = True):
        """
        Plot Fibonacci retracement levels.

        Args:
            levels: Dictionary of Fibonacci ratios to price levels
            indices: List of indices to span the lines
            is_uptrend: Whether the market is in an uptrend
        """
        colors = ['#ffa726', '#ffcc80', '#ffe0b2', '#fff3e0', '#fff8e1', '#fffde7']
        labels = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']

        # Get x-axis limits for horizontal lines
        x_min, x_max = min(indices), max(indices)

        # Sort levels by price (ascending or descending based on trend)
        sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=not is_uptrend)

        # Plot each level with a unique color and label
        for i, (ratio, price) in enumerate(sorted_levels):
            color = colors[i % len(colors)]
            label = labels[list(levels.keys()).index(ratio)]

            self.ax.axhline(y=price, color=color, linestyle='--', alpha=0.7,
                            label=f"Fib {label}")

            # Add text label at the right edge
            self.ax.text(x_max, price, f"{label} ({price:.2f})",
                         verticalalignment='center', fontsize=8)

    def plot_extension_levels(self, levels: Dict[float, float],
                              indices: List[int], is_uptrend: bool = True):
        """
        Plot Fibonacci extension levels.

        Args:
            levels: Dictionary of Fibonacci ratios to price levels
            indices: List of indices to span the lines
            is_uptrend: Whether the market is in an uptrend
        """
        colors = ['#ba68c8', '#ce93d8', '#e1bee7', '#f3e5f5']

        # Only plot a few key extension levels to avoid cluttering
        key_extensions = [0.618, 1.0, 1.618, 2.618]

        # Get x-axis limits
        x_min, x_max = min(indices), max(indices)

        # Plot each key extension level
        for ext in key_extensions:
            if ext in levels:
                price = levels[ext]
                color = colors[key_extensions.index(ext) % len(colors)]

                self.ax.axhline(y=price, color=color, linestyle='-.', alpha=0.5,
                                label=f"Ext {ext}")

                # Add text label at the right edge
                self.ax.text(x_max, price, f"Ext {ext} ({price:.2f})",
                             verticalalignment='center', fontsize=8)

    def update_plot(self):
        """Update the plot with the current settings."""
        # Get the current price data (placeholder)
        # In a real implementation, this would get data from the actual price source
        prices = self.get_current_prices()

        # Get the start and end indices
        start_idx = self.start_var.get()
        end_idx = self.end_var.get()

        # Plot with the new range
        self.plot_fibonacci_levels(prices, start_idx, end_idx)

    def get_current_prices(self) -> List[float]:
        """
        Get the current price data.

        Returns:
            List of price values
        """
        # This is a placeholder. In a real implementation, this would get data from the actual source.
        # For demo purposes, we'll generate some dummy data
        np.random.seed(42)
        x = np.arange(200)
        prices = 100 + 0.5 * x + 10 * np.sin(x / 10) + np.random.normal(0, 3, 200)
        return prices.tolist()

    def update_with_real_data(self, symbol: str, timeframe: str = "1h",
                              candles: Optional[List[Dict]] = None):
        """
        Update the visualization with real market data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            candles: Optional list of candle data (if not provided, will be fetched)
        """
        if candles is None:
            # This would normally fetch data from the exchange
            # For now, just use dummy data
            prices = self.get_current_prices()
        else:
            # Extract close prices from candles
            prices = [candle["close"] for candle in candles]

        # Update the plot with the new data
        self.plot_fibonacci_levels(prices)

        # Update info text
        self.info_var.set(f"Updated: {symbol} ({timeframe})")


def create_fibonacci_tab(notebook, client=None):
    """
    Create and add a Fibonacci analysis tab to a notebook.

    Args:
        notebook: ttk.Notebook widget
        client: OKXClient instance (optional)

    Returns:
        FibonacciVisualizer instance
    """
    # Create the tab frame
    fib_frame = ttk.Frame(notebook)
    notebook.add(fib_frame, text="Fibonacci Analysis")

    # Create the main layout
    main_frame = ttk.Frame(fib_frame)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Create settings frame at top
    settings_frame = ttk.LabelFrame(main_frame, text="Fibonacci Settings")
    settings_frame.pack(fill=tk.X, pady=5)

    # Symbol selection
    symbol_frame = ttk.Frame(settings_frame)
    symbol_frame.pack(fill=tk.X, pady=5, padx=5)

    ttk.Label(symbol_frame, text="Symbol:").pack(side=tk.LEFT, padx=5)
    symbol_var = tk.StringVar(value="BTC-USDT")
    symbol_entry = ttk.Entry(symbol_frame, textvariable=symbol_var, width=15)
    symbol_entry.pack(side=tk.LEFT, padx=5)

    ttk.Label(symbol_frame, text="Timeframe:").pack(side=tk.LEFT, padx=5)
    timeframe_var = tk.StringVar(value="1h")
    timeframe_combo = ttk.Combobox(symbol_frame, textvariable=timeframe_var,
                                   values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], width=5)
    timeframe_combo.pack(side=tk.LEFT, padx=5)

    ttk.Label(symbol_frame, text="Lookback:").pack(side=tk.LEFT, padx=5)
    lookback_var = tk.IntVar(value=100)
    lookback_spin = ttk.Spinbox(symbol_frame, from_=20, to=500, increment=10,
                                textvariable=lookback_var, width=8)
    lookback_spin.pack(side=tk.LEFT, padx=5)

    # Add fetch button
    fetch_button = ttk.Button(symbol_frame, text="Analyze",
                              command=lambda: update_fibonacci_viz(client, fib_viz, symbol_var.get(),
                                                                   timeframe_var.get(), lookback_var.get()))
    fetch_button.pack(side=tk.LEFT, padx=20)

    # Create info frame
    info_frame = ttk.LabelFrame(main_frame, text="Fibonacci Analysis Results")
    info_frame.pack(fill=tk.X, pady=5)

    # Create result text
    result_text = tk.Text(info_frame, wrap=tk.WORD, height=6)
    result_text.pack(fill=tk.X, padx=5, pady=5)

    # Create the Fibonacci visualizer
    viz_frame = ttk.LabelFrame(main_frame, text="Fibonacci Visualization")
    viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    fib_viz = FibonacciVisualizer(viz_frame)

    # Function to update with real data
    def update_fibonacci_viz(client, visualizer, symbol, timeframe, lookback):
        try:
            # Fetch data (would use real client in actual implementation)
            if client:
                candles = client.get_klines(symbol, interval=timeframe, limit=lookback)
            else:
                # Use dummy data
                candles = None

            # Update the visualization
            visualizer.update_with_real_data(symbol, timeframe, candles)

            # Update result text
            if visualizer.current_zones and visualizer.current_zones["has_zones"]:
                fib_zones = visualizer.current_zones["zones"]
                current_price = visualizer.current_zones["current_price"]

                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Symbol: {symbol} ({timeframe})\n")
                result_text.insert(tk.END, f"Current Price: ${current_price:.4f}\n")
                result_text.insert(tk.END, f"Trend: {fib_zones['trend'].capitalize()}\n")

                if fib_zones["is_at_key_level"]:
                    level = fib_zones["closest_level"]
                    result_text.insert(tk.END, f"Price at key Fibonacci level: {level}\n")
                    result_text.insert(tk.END, "POTENTIAL TRADE OPPORTUNITY\n")

                    # Get trade direction based on trend
                    if fib_zones["trend"] == "uptrend":
                        direction = "BUY"
                        stop_loss = fib_zones["swing_low"]
                        take_profit = current_price + (current_price - stop_loss) * 1.5
                    else:
                        direction = "SELL"
                        stop_loss = fib_zones["swing_high"]
                        take_profit = current_price - (stop_loss - current_price) * 1.5

                    result_text.insert(tk.END,
                                       f"Signal: {direction} at {current_price:.4f}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
            else:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, "No clear Fibonacci levels detected")

        except Exception as e:
            print(f"Error updating Fibonacci visualization: {str(e)}")
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Error: {str(e)}")

    return fib_viz


# Standalone test
if __name__ == "__main__":
    # Create a standalone application for testing
    root = tk.Tk()
    root.title("Fibonacci Level Visualizer")
    root.geometry("1000x800")

    # Create a notebook
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Create the Fibonacci tab
    fib_viz = create_fibonacci_tab(notebook)

    # Run the application
    root.mainloop()
