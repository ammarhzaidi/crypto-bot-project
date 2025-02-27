import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import os
import time
from datetime import datetime

# Add src to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.market_data.okx_client import OKXClient
from src.strategies.hh_hl_strategy import analyze_price_action
from src.risk_management.position_sizer import calculate_take_profit, calculate_stop_loss


class TradingBotGUI:
    """
    Simple GUI for the trading bot.
    """

    def __init__(self, root):
        """
        Initialize the GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Crypto Trading Bot")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # Initialize the OKX client
        self.client = OKXClient()

        # Set default values
        self.symbols_count = tk.IntVar(value=10)
        self.tp_percent = tk.DoubleVar(value=1.0)
        self.sl_percent = tk.DoubleVar(value=1.0)

        # Create the main frame
        self.create_widgets()

        # For tracking running state
        self.running = False
        self.analysis_thread = None
        self.start_time = 0

    def create_widgets(self):
        """Create all GUI widgets."""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create top control frame
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Create grid layout for controls
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)

        # Add number of symbols control
        ttk.Label(control_frame, text="Top N Symbols:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        symbol_count = ttk.Spinbox(control_frame, from_=5, to=50, increment=5, textvariable=self.symbols_count,
                                   width=10)
        symbol_count.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Add TP/SL percentages
        ttk.Label(control_frame, text="TP %:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        tp_entry = ttk.Spinbox(control_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.tp_percent, width=10)
        tp_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(control_frame, text="SL %:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        sl_entry = ttk.Spinbox(control_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.sl_percent, width=10)
        sl_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Add button to run analysis
        run_button = ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis)
        run_button.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Create notebook for results
        results_notebook = ttk.Notebook(main_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabs for uptrends, downtrends, no trends, and log
        self.uptrends_frame = ttk.Frame(results_notebook, padding="10")
        self.downtrends_frame = ttk.Frame(results_notebook, padding="10")
        self.notrends_frame = ttk.Frame(results_notebook, padding="10")
        self.log_frame = ttk.Frame(results_notebook, padding="10")

        results_notebook.add(self.uptrends_frame, text="Uptrends")
        results_notebook.add(self.downtrends_frame, text="Downtrends")
        results_notebook.add(self.notrends_frame, text="No Clear Trend")
        results_notebook.add(self.log_frame, text="Log")

        # Create uptrends treeview
        self.create_trends_treeview(self.uptrends_frame, "uptrends")

        # Create downtrends treeview
        self.create_trends_treeview(self.downtrends_frame, "downtrends")

        # Create no trends list
        self.create_notrends_list(self.notrends_frame)

        # Create log text area
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar with two parts
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        status_frame.columnconfigure(0, weight=3)
        status_frame.columnconfigure(1, weight=1)

        # Status message (left side)
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=0, column=0, sticky=tk.W + tk.E)

        # Timer display (right side)
        self.timer_var = tk.StringVar()
        self.timer_var.set("Time: 0s")
        timer_label = ttk.Label(status_frame, textvariable=self.timer_var, relief=tk.SUNKEN, anchor=tk.E)
        timer_label.grid(row=0, column=1, sticky=tk.E)

    def create_trends_treeview(self, parent, trend_type):
        """
        Create a treeview to display trend information.

        Args:
            parent: Parent frame
            trend_type: Type of trend ('uptrends' or 'downtrends')
        """
        # Create frame for treeview and scrollbar
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create treeview
        columns = ("Symbol", "Pattern", "Entry", "TP", "SL", "Side")
        treeview = ttk.Treeview(frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)

        # Configure columns
        treeview.heading("Symbol", text="Symbol")
        treeview.heading("Pattern", text="Pattern")
        treeview.heading("Entry", text="Entry")
        treeview.heading("TP", text="Take Profit")
        treeview.heading("SL", text="Stop Loss")
        treeview.heading("Side", text="Side")

        treeview.column("Symbol", width=100)
        treeview.column("Pattern", width=150)
        treeview.column("Entry", width=100)
        treeview.column("TP", width=100)
        treeview.column("SL", width=100)
        treeview.column("Side", width=50)

        # Configure scrollbar
        scrollbar.config(command=treeview.yview)

        # Pack treeview
        treeview.pack(fill=tk.BOTH, expand=True)

        # Store reference to treeview
        if trend_type == "uptrends":
            self.uptrends_treeview = treeview
        else:
            self.downtrends_treeview = treeview

    def create_notrends_list(self, parent):
        """
        Create a listbox to display symbols with no clear trend.

        Args:
            parent: Parent frame
        """
        # Create frame for listbox and scrollbar
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        # Add explanation label
        explanation = ttk.Label(frame, text="Symbols that don't show a clear trend pattern based on current analysis:")
        explanation.pack(anchor=tk.W, padx=5, pady=5)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create listbox
        self.notrends_listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("TkDefaultFont", 10))
        self.notrends_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure scrollbar
        scrollbar.config(command=self.notrends_listbox.yview)

    def log(self, message):
        """
        Add a message to the log.

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Insert log message in a thread-safe way
        self.root.after(0, lambda: self.log_text.insert(tk.END, log_message))
        self.root.after(0, lambda: self.log_text.see(tk.END))

    def update_status(self, message):
        """
        Update the status bar.

        Args:
            message: Status message
        """
        self.status_var.set(message)

    def update_timer(self):
        """Update the timer display while analysis is running."""
        if not self.running:
            return

        elapsed = time.time() - self.start_time
        self.timer_var.set(f"Time: {self.format_time(elapsed)}")
        self.root.after(100, self.update_timer)  # Update every 100ms

    def format_time(self, seconds):
        """
        Format time in seconds to a readable string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (e.g., "5s" or "2m 30s")
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"

    def run_analysis(self):
        """Run the trading analysis in a separate thread."""
        if self.running:
            messagebox.showinfo("Running", "Analysis is already running. Please wait.")
            return

        # Clear previous results
        for item in self.uptrends_treeview.get_children():
            self.uptrends_treeview.delete(item)

        for item in self.downtrends_treeview.get_children():
            self.downtrends_treeview.delete(item)

        self.notrends_listbox.delete(0, tk.END)

        self.log_text.delete(1.0, tk.END)

        # Get parameters
        n_symbols = self.symbols_count.get()
        tp_percent = self.tp_percent.get()
        sl_percent = self.sl_percent.get()

        # Start timer
        self.start_time = time.time()
        self.timer_var.set("Time: 0s")
        self.update_timer()

        # Start analysis thread
        self.running = True
        self.analysis_thread = threading.Thread(target=self.analysis_task,
                                                args=(n_symbols, tp_percent, sl_percent))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def analysis_task(self, n_symbols, tp_percent, sl_percent):
        """
        Perform analysis in a separate thread.

        Args:
            n_symbols: Number of top symbols to analyze
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
        """
        try:
            self.update_status("Fetching symbols...")
            self.log(f"Starting analysis with top {n_symbols} symbols, TP: {tp_percent}%, SL: {sl_percent}%")

            # Get top symbols
            self.log("Fetching top volume symbols...")
            symbols = self.client.get_top_volume_symbols(limit=n_symbols)

            if not symbols:
                self.log("No symbols found. Analysis failed.")
                self.update_status("Analysis failed - no symbols found")
                self.running = False
                return

            self.log(f"Found {len(symbols)} symbols")

            # Analyze each symbol
            uptrends = []
            downtrends = []
            no_trends = []

            for i, symbol in enumerate(symbols):
                self.update_status(f"Analyzing {symbol} ({i + 1}/{len(symbols)})...")
                self.log(f"Analyzing {symbol}...")

                # Get historical price data
                klines = self.client.get_klines(symbol, interval="1h", limit=48)

                if not klines:
                    self.log(f"No data available for {symbol}, skipping")
                    continue

                # Extract close prices
                close_prices = [candle["close"] for candle in klines]

                # Apply HH/HL strategy
                result = analyze_price_action(close_prices, smoothing=1, consecutive_count=2)
                trend = result["trend"]

                # Get current price
                ticker = self.client.get_ticker(symbol)
                if not ticker or "last_price" not in ticker:
                    self.log(f"Could not get current price for {symbol}, skipping")
                    continue

                current_price = ticker["last_price"]

                # Process results based on trend
                if trend == "uptrend":
                    hh_count = result["uptrend_analysis"]["consecutive_hh"]
                    hl_count = result["uptrend_analysis"]["consecutive_hl"]
                    pattern = f"{hh_count} HH, {hl_count} HL"

                    # Calculate pattern strength (minimum of HH and HL count)
                    pattern_strength = min(hh_count, hl_count)

                    # Calculate TP/SL
                    tp = calculate_take_profit(current_price, tp_percent)
                    sl = calculate_stop_loss(current_price, sl_percent)

                    uptrends.append((symbol, pattern, current_price, tp, sl, "BUY", pattern_strength))

                    # Add strength indicator to log
                    if pattern_strength >= 3:
                        self.log(f"✅✅✅ STRONG UPTREND: {symbol} - {pattern}")
                    else:
                        self.log(f"✅ UPTREND: {symbol} - {pattern}")

                elif trend == "downtrend":
                    lh_count = result["downtrend_analysis"]["consecutive_lh"]
                    ll_count = result["downtrend_analysis"]["consecutive_ll"]
                    pattern = f"{lh_count} LH, {ll_count} LL"

                    # Calculate pattern strength (minimum of LH and LL count)
                    pattern_strength = min(lh_count, ll_count)

                    # For shorts, TP is lower and SL is higher
                    tp = current_price * (1 - tp_percent / 100)
                    sl = current_price * (1 + sl_percent / 100)

                    downtrends.append((symbol, pattern, current_price, tp, sl, "SELL", pattern_strength))

                    # Add strength indicator to log
                    if pattern_strength >= 3:
                        self.log(f"🔻🔻🔻 STRONG DOWNTREND: {symbol} - {pattern}")
                    else:
                        self.log(f"🔻 DOWNTREND: {symbol} - {pattern}")

                else:
                    # No clear trend
                    no_trends.append(symbol)
                    self.log(f"➖ NO TREND: {symbol}")

                # Small delay to avoid hammering the API
                time.sleep(0.1)

            # Sort by pattern strength (3 consecutive patterns first, then 2)
            uptrends.sort(key=lambda x: x[6], reverse=True)
            downtrends.sort(key=lambda x: x[6], reverse=True)

            # Configure tags first
            self.root.after(0, lambda: self.uptrends_treeview.tag_configure("strength3",
                                                                            background="#d4ffcc"))  # Light green
            self.root.after(0, lambda: self.uptrends_treeview.tag_configure("strength2", background="white"))
            self.root.after(0, lambda: self.downtrends_treeview.tag_configure("strength3",
                                                                              background="#ffcccc"))  # Light red
            self.root.after(0, lambda: self.downtrends_treeview.tag_configure("strength2", background="white"))

            # Update treeviews and listbox
            for data in uptrends:
                # Remove pattern_strength from display data
                display_data = data[:-1]  # Everything except the last item
                strength = data[6]  # Pattern strength
                tag = f"strength{min(strength, 3)}"  # Use strength3 tag for any strength >= 3

                # Insert with appropriate tag
                self.root.after(0, lambda d=display_data, t=tag:
                self.uptrends_treeview.insert("", tk.END, values=self.format_values(d), tags=(t,)))

            for data in downtrends:
                # Remove pattern_strength from display data
                display_data = data[:-1]  # Everything except the last item
                strength = data[6]  # Pattern strength
                tag = f"strength{min(strength, 3)}"  # Use strength3 tag for any strength >= 3

                # Insert with appropriate tag
                self.root.after(0, lambda d=display_data, t=tag:
                self.downtrends_treeview.insert("", tk.END, values=self.format_values(d), tags=(t,)))

            for symbol in no_trends:
                self.root.after(0, lambda s=symbol: self.notrends_listbox.insert(tk.END, s))

            # Final timing
            elapsed_time = time.time() - self.start_time
            formatted_time = self.format_time(elapsed_time)

            # Update status and log
            self.update_status(
                f"Analysis complete in {formatted_time}. Found {len(uptrends)} uptrends, {len(downtrends)} downtrends, {len(no_trends)} no trends")
            self.log(
                f"Analysis complete in {formatted_time}. Found {len(uptrends)} uptrends, {len(downtrends)} downtrends, {len(no_trends)} no trends")

        except Exception as e:
            self.log(f"Error during analysis: {str(e)}")
            self.update_status("Analysis failed")

        finally:
            # Ensure timer shows final time
            elapsed_time = time.time() - self.start_time
            self.timer_var.set(f"Time: {self.format_time(elapsed_time)}")
            self.running = False

    def format_values(self, data):
        """
        Format values for display in treeview.

        Args:
            data: Tuple of (symbol, pattern, price, tp, sl, side)

        Returns:
            Tuple of formatted values
        """
        symbol, pattern, price, tp, sl, side = data

        # Determine decimal places based on price magnitude
        if price < 0.0001:
            decimals = 10
        elif price < 0.01:
            decimals = 8
        elif price < 1:
            decimals = 6
        elif price < 100:
            decimals = 4
        else:
            decimals = 2

        # Format strings
        price_str = f"${price:.{decimals}f}"
        tp_str = f"${tp:.{decimals}f}"
        sl_str = f"${sl:.{decimals}f}"

        return symbol, pattern, price_str, tp_str, sl_str, side


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()