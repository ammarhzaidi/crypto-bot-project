"""
Example showing how to integrate Fibonacci analysis into the main trading GUI.
This demonstrates extending the existing TabbedTradingBotGUI with Fibonacci capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from typing import List, Dict, Any, Optional

# Add src to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.gui.trading_gui_tabbed import TabbedTradingBotGUI
from src.gui.fibonacci_visualizer import create_fibonacci_tab
from src.strategies.fibonacci_levels import FibonacciAnalyzer
from src.strategies.fibonacci_integration import enhanced_hh_hl_with_fibonacci
from src.risk_management.position_sizer import calculate_take_profit, calculate_stop_loss, format_trade_summary


class FibonacciEnhancedGUI(TabbedTradingBotGUI):
    """
    Extended version of the TabbedTradingBotGUI with Fibonacci analysis capabilities.
    """

    def __init__(self, root):
        """
        Initialize the enhanced GUI with Fibonacci analysis.

        Args:
            root: Tkinter root window
        """
        # Call parent's init method
        super().__init__(root)

        # Add Fibonacci tab
        self.fib_viz = create_fibonacci_tab(self.notebook, client=self.client)

        # Enhance HH/HL tab with Fibonacci settings
        self.add_fibonacci_to_hhhl_tab()

    def add_fibonacci_to_hhhl_tab(self):
        """Add Fibonacci analysis options to the HH/HL tab."""
        # Create a new frame for Fibonacci options
        fib_frame = ttk.LabelFrame(self.hhhl_tab, text="Fibonacci Enhancement", padding="10")
        fib_frame.pack(fill=tk.X, pady=5)

        # Add option to enable Fibonacci analysis
        self.use_fibonacci = tk.BooleanVar(value=True)
        ttk.Checkbutton(fib_frame, text="Enhance signals with Fibonacci analysis",
                        variable=self.use_fibonacci).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)

        # Add option to filter by Fibonacci confluence
        self.require_fib_confluence = tk.BooleanVar(value=False)
        ttk.Checkbutton(fib_frame, text="Only show signals with Fibonacci confluence",
                        variable=self.require_fib_confluence).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)

        # Additional controls for Fibonacci levels
        ttk.Label(fib_frame, text="Key Levels:").grid(row=0, column=1, sticky=tk.W, padx=20, pady=2)

        # Create a frame for level checkboxes
        levels_frame = ttk.Frame(fib_frame)
        levels_frame.grid(row=1, column=1, sticky=tk.W, padx=20, pady=2)

        # Variables for each level
        self.fib_levels = {
            '0.236': tk.BooleanVar(value=False),
            '0.382': tk.BooleanVar(value=True),
            '0.5': tk.BooleanVar(value=True),
            '0.618': tk.BooleanVar(value=True),
            '0.786': tk.BooleanVar(value=False)
        }

        # Create checkboxes for each level
        ttk.Checkbutton(levels_frame, text="0.236",
                        variable=self.fib_levels['0.236']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(levels_frame, text="0.382",
                        variable=self.fib_levels['0.382']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(levels_frame, text="0.5",
                        variable=self.fib_levels['0.5']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(levels_frame, text="0.618",
                        variable=self.fib_levels['0.618']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(levels_frame, text="0.786",
                        variable=self.fib_levels['0.786']).pack(side=tk.LEFT, padx=5)

        # Add info tooltip
        info_text = ("Fibonacci analysis enhances the HH/HL strategy by:\n"
                     "1. Identifying key Fibonacci retracement levels\n"
                     "2. Finding confluence between trend and Fibonacci levels\n"
                     "3. Providing enhanced price targets and stop losses")

        info_label = ttk.Label(fib_frame, text=info_text, foreground="gray50",
                               font=("TkDefaultFont", 8, "italic"), wraplength=400)
        info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

    def hhhl_analysis_task(self, n_symbols, tp_percent, sl_percent):
        """
        Override the parent method to add Fibonacci analysis.

        Args:
            n_symbols: Number of top symbols to analyze
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
        """
        try:
            # Use parent implementation when Fibonacci is disabled
            if not self.use_fibonacci.get():
                return super().hhhl_analysis_task(n_symbols, tp_percent, sl_percent)

            self.update_hhhl_status("Fetching symbols with Fibonacci analysis...")
            self.log_hhhl(f"Starting enhanced HH/HL analysis with Fibonacci for top {n_symbols} symbols")

            # Get available symbols
            symbols = self.client.get_top_volume_symbols(limit=n_symbols)

            if not symbols:
                self.log_hhhl("No symbols found. Analysis failed.")
                self.update_hhhl_status("Analysis failed - no symbols found")
                self.running = False
                return

            # Results containers
            uptrends = []
            downtrends = []
            no_trends = []

            # Create Fibonacci analyzer
            fib_analyzer = FibonacciAnalyzer()

            # Analyze each symbol
            for i, symbol in enumerate(symbols):
                self.update_hhhl_status(f"Analyzing {symbol} with Fibonacci ({i + 1}/{len(symbols)})...")
                self.log_hhhl(f"Analyzing {symbol}...")

                # Get historical price data
                klines = self.client.get_klines(symbol, interval="1h", limit=48)

                if not klines:
                    self.log_hhhl(f"No data available for {symbol}, skipping")
                    continue

                # Extract close prices
                close_prices = [candle["close"] for candle in klines]

                # Apply enhanced HH/HL with Fibonacci analysis
                result = enhanced_hh_hl_with_fibonacci(close_prices, smoothing=1, consecutive_count=2)

                # Get current price
                ticker = self.client.get_ticker(symbol)
                if not ticker or "last_price" not in ticker:
                    self.log_hhhl(f"Could not get current price for {symbol}, skipping")
                    continue

                current_price = ticker["last_price"]

                # Process results based on trend
                if result["trend"] == "uptrend":
                    # Check if we're requiring Fibonacci confluence
                    if self.require_fib_confluence.get() and not result["has_fib_confluence"]:
                        no_trends.append(symbol)
                        self.log_hhhl(f"➖ NO FIB CONFLUENCE: {symbol}")
                        continue

                    # Use standard HH/HL data
                    hh_count = result["uptrend_analysis"]["consecutive_hh"]
                    hl_count = result["uptrend_analysis"]["consecutive_hl"]
                    pattern = f"{hh_count} HH, {hl_count} HL"

                    # Calculate pattern strength (minimum of HH and HL count)
                    pattern_strength = min(hh_count, hl_count)

                    # Create base signal data
                    signal_data = {
                        'symbol': symbol,
                        'pattern': pattern,
                        'price': current_price,
                        'side': 'BUY',
                        'strength': pattern_strength,
                        'volume': ticker.get('volume_24h', 0) * current_price,
                        'volume_formatted': self.format_volume(ticker.get('volume_24h', 0) * current_price)
                    }

                    # Check if we have Fibonacci confluence and signals
                    if result["has_fib_confluence"] and result["signals"]:
                        fib_signal = result["signals"][0]

                        # Add Fibonacci info
                        signal_data['pattern'] = fib_signal["pattern"]
                        signal_data['tp'] = fib_signal["tp"]
                        signal_data['sl'] = fib_signal["sl"]
                        signal_data['fib_level'] = fib_signal.get("fibonacci_level", "")

                        self.log_hhhl(
                            f"✅🔷 UPTREND w/FIB CONFLUENCE: {symbol} - {pattern}, Fib {signal_data['fib_level']}")
                    else:
                        # Calculate standard TP/SL
                        signal_data['tp'] = calculate_take_profit(current_price, tp_percent)
                        signal_data['sl'] = calculate_stop_loss(current_price, sl_percent)

                        self.log_hhhl(f"✅ UPTREND: {symbol} - {pattern}")

                    uptrends.append(signal_data)

                elif result["trend"] == "downtrend":
                    # Check if we're requiring Fibonacci confluence
                    if self.require_fib_confluence.get() and not result["has_fib_confluence"]:
                        no_trends.append(symbol)
                        self.log_hhhl(f"➖ NO FIB CONFLUENCE: {symbol}")
                        continue

                    # Use standard HH/HL data
                    lh_count = result["downtrend_analysis"]["consecutive_lh"]
                    ll_count = result["downtrend_analysis"]["consecutive_ll"]
                    pattern = f"{lh_count} LH, {ll_count} LL"

                    # Calculate pattern strength (minimum of HH and HL count)
                    pattern_strength = min(lh_count, ll_count)

                    # Create base signal data
                    signal_data = {
                        'symbol': symbol,
                        'pattern': pattern,
                        'price': current_price,
                        'side': 'SELL',
                        'strength': pattern_strength,
                        'volume': ticker.get('volume_24h', 0) * current_price,
                        'volume_formatted': self.format_volume(ticker.get('volume_24h', 0) * current_price)
                    }

                    # Check if we have Fibonacci confluence and signals
                    if result["has_fib_confluence"] and result["signals"]:
                        fib_signal = result["signals"][0]

                        # Add Fibonacci info
                        signal_data['pattern'] = fib_signal["pattern"]
                        signal_data['tp'] = fib_signal["tp"]
                        signal_data['sl'] = fib_signal["sl"]
                        signal_data['fib_level'] = fib_signal.get("fibonacci_level", "")

                        self.log_hhhl(
                            f"🔻🔷 DOWNTREND w/FIB CONFLUENCE: {symbol} - {pattern}, Fib {signal_data['fib_level']}")
                    else:
                        # For shorts, TP is lower and SL is higher
                        signal_data['tp'] = current_price * (1 - tp_percent / 100)
                        signal_data['sl'] = current_price * (1 + sl_percent / 100)

                        self.log_hhhl(f"🔻 DOWNTREND: {symbol} - {pattern}")

                    downtrends.append(signal_data)

                else:
                    # No clear trend
                    no_trends.append(symbol)
                    self.log_hhhl(f"➖ NO TREND: {symbol}")

                # Small delay to avoid hammering the API
                import time
                time.sleep(0.1)

            # Create a class to mimic the HHHLResult structure expected by update_hhhl_results
            class FibonacciHHHLResult:
                def __init__(self, uptrends, downtrends, no_trends, execution_time):
                    self.uptrends = uptrends
                    self.downtrends = downtrends
                    self.no_trends = no_trends
                    self.execution_time = execution_time

            # Calculate execution time
            import time
            execution_time = time.time() - self.start_time

            # Create the result object
            result = FibonacciHHHLResult(uptrends, downtrends, no_trends, execution_time)

            # Update UI with results
            self.update_hhhl_results(result)

        except Exception as e:
            self.log_hhhl(f"Error during analysis: {str(e)}")
            self.update_hhhl_status("Analysis failed")

        finally:
            # Ensure timer shows final time
            elapsed_time = time.time() - self.start_time
            self.hhhl_timer_var.set(f"Time: {self.format_time(elapsed_time)}")
            self.running = False

    def format_volume(self, volume: float) -> str:
        """Format volume for display."""
        if volume >= 1000000000:  # Billions
            return f"${volume / 1000000000:.2f}B"
        elif volume >= 1000000:  # Millions
            return f"${volume / 1000000:.2f}M"
        elif volume >= 1000:  # Thousands
            return f"${volume / 1000:.2f}K"
        else:
            return f"${volume:.2f}"


def main():
    """Main entry point for the enhanced GUI application."""
    root = tk.Tk()
    app = FibonacciEnhancedGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()