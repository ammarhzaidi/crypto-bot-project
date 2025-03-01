import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import os
import time
from datetime import datetime
from src.analysis.analysis_service import (
    AnalysisService,
    HHHLAnalysisParams,
    CandlestickAnalysisParams,
    HHHLResult
)

# Add src to path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.market_data.okx_client import OKXClient
from src.strategies.hh_hl_strategy import analyze_price_action
from src.risk_management.position_sizer import calculate_take_profit, calculate_stop_loss
from src.strategies.candlestick_patterns import CandlestickPatternFinder


class TabbedTradingBotGUI:
    """
    Enhanced GUI for the trading bot with tabbed interface.
    """

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

    def create_candlestick_treeview(self, parent):
        """
        Create a treeview to display candlestick pattern results.

        Args:
            parent: Parent frame
        """
        # Create frame for treeview and scrollbar
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create treeview with added Volume column
        columns = ("Symbol", "Pattern", "Price", "Time (PST)", "Is Bullish", "Quality", "Volume 24h")
        self.candlestick_treeview = ttk.Treeview(frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)

        # Configure columns
        self.candlestick_treeview.heading("Symbol", text="Symbol")
        self.candlestick_treeview.heading("Pattern", text="Pattern")
        self.candlestick_treeview.heading("Price", text="Price")
        self.candlestick_treeview.heading("Time (PST)", text="Time (PST)")
        self.candlestick_treeview.heading("Is Bullish", text="Is Bullish")
        self.candlestick_treeview.heading("Quality", text="Quality")
        self.candlestick_treeview.heading("Volume 24h", text="Volume 24h ($)")

        self.candlestick_treeview.column("Symbol", width=100)
        self.candlestick_treeview.column("Pattern", width=120)
        self.candlestick_treeview.column("Price", width=100)
        self.candlestick_treeview.column("Time (PST)", width=150)
        self.candlestick_treeview.column("Is Bullish", width=80)
        self.candlestick_treeview.column("Quality", width=80)
        self.candlestick_treeview.column("Volume 24h", width=120)

        # Configure scrollbar
        scrollbar.config(command=self.candlestick_treeview.yview)

        # Pack treeview
        self.candlestick_treeview.pack(fill=tk.BOTH, expand=True)

        # Configure tags for quality levels
        self.candlestick_treeview.tag_configure('good', background='#c8e6c9')  # Light green
        self.candlestick_treeview.tag_configure('moderate', background='#fff9c4')  # Light yellow
        self.candlestick_treeview.tag_configure('poor', background='#ffcdd2')  # Light red

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

        # Initialize the analysis service
        self.analysis_service = AnalysisService(client=self.client, logger=self.log_hhhl)

        # Initialize the candlestick pattern finder
        self.pattern_finder = CandlestickPatternFinder()

        # Set default values for HH/HL tab
        self.symbols_count = tk.IntVar(value=10)
        self.tp_percent = tk.DoubleVar(value=1.0)
        self.sl_percent = tk.DoubleVar(value=1.0)

        # Set default values for Candlestick Finder tab
        self.cs_symbols_count = tk.IntVar(value=10)
        self.cs_timeframe = tk.StringVar(value="1h")
        self.cs_candles_count = tk.IntVar(value=48)
        self.check_freshness = tk.BooleanVar(value=False)

        # Dictionary to store selected candlestick patterns
        self.selected_patterns = {
            'hammer': tk.BooleanVar(value=True),
            'bullish_engulfing': tk.BooleanVar(value=False),
            'piercing': tk.BooleanVar(value=False),
            'morning_star': tk.BooleanVar(value=False),
            'doji': tk.BooleanVar(value=False)
        }

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

        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.hhhl_tab = ttk.Frame(self.notebook)
        self.candlestick_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.hhhl_tab, text="HH/HL Strategy")
        self.notebook.add(self.candlestick_tab, text="Candlestick Finder")

        # Initialize both tabs
        self.init_hhhl_tab()
        self.init_candlestick_tab()

    def init_hhhl_tab(self):
        """Initialize the HH/HL Strategy tab."""
        # Create top control frame
        control_frame = ttk.LabelFrame(self.hhhl_tab, text="HH/HL Analysis Controls", padding="10")
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

        ttk.Checkbutton(control_frame, text="Prioritize Fresh Signals", variable=self.check_freshness).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Add button to run analysis
        run_button = ttk.Button(control_frame, text="Run HH/HL Analysis", command=self.run_hhhl_analysis)
        run_button.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Create notebook for results
        results_notebook = ttk.Notebook(self.hhhl_tab)
        results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabs for uptrends, downtrends, no trends, and log
        self.uptrends_frame = ttk.Frame(results_notebook, padding="10")
        self.downtrends_frame = ttk.Frame(results_notebook, padding="10")
        self.notrends_frame = ttk.Frame(results_notebook, padding="10")
        self.hhhl_log_frame = ttk.Frame(results_notebook, padding="10")

        results_notebook.add(self.uptrends_frame, text="Uptrends")
        results_notebook.add(self.downtrends_frame, text="Downtrends")
        results_notebook.add(self.notrends_frame, text="No Clear Trend")
        results_notebook.add(self.hhhl_log_frame, text="Log")

        # Create uptrends treeview
        self.create_trends_treeview(self.uptrends_frame, "uptrends")

        # Create downtrends treeview
        self.create_trends_treeview(self.downtrends_frame, "downtrends")

        # Create no trends list
        self.create_notrends_list(self.notrends_frame)

        # Create log text area
        self.hhhl_log_text = scrolledtext.ScrolledText(self.hhhl_log_frame, wrap=tk.WORD)
        self.hhhl_log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar for HH/HL tab
        status_frame = ttk.Frame(self.hhhl_tab)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        status_frame.columnconfigure(0, weight=3)
        status_frame.columnconfigure(1, weight=1)

        # Status message (left side)
        self.hhhl_status_var = tk.StringVar()
        self.hhhl_status_var.set("Ready")
        status_label = ttk.Label(status_frame, textvariable=self.hhhl_status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=0, column=0, sticky=tk.W + tk.E)

        # Timer display (right side)
        self.hhhl_timer_var = tk.StringVar()
        self.hhhl_timer_var.set("Time: 0s")
        timer_label = ttk.Label(status_frame, textvariable=self.hhhl_timer_var, relief=tk.SUNKEN, anchor=tk.E)
        timer_label.grid(row=0, column=1, sticky=tk.E)

    def init_candlestick_tab(self):
        """Initialize the Candlestick Finder tab."""
        # Create top control frame for candlestick finder
        cs_control_frame = ttk.LabelFrame(self.candlestick_tab, text="Candlestick Pattern Finder", padding="10")
        cs_control_frame.pack(fill=tk.X, pady=5)

        # Create patterns frame
        patterns_frame = ttk.LabelFrame(cs_control_frame, text="Select Patterns")
        patterns_frame.pack(fill=tk.X, pady=5, padx=5)

        # Create grid for pattern checkboxes (2 columns)
        patterns_frame.columnconfigure(0, weight=1)
        patterns_frame.columnconfigure(1, weight=1)

        # Add pattern checkboxes
        ttk.Checkbutton(patterns_frame, text="Hammer", variable=self.selected_patterns['hammer']).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(patterns_frame, text="Bullish Engulfing",
                        variable=self.selected_patterns['bullish_engulfing']).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(patterns_frame, text="Piercing Pattern", variable=self.selected_patterns['piercing']).grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(patterns_frame, text="Morning Star", variable=self.selected_patterns['morning_star']).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(patterns_frame, text="Doji", variable=self.selected_patterns['doji']).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Create settings frame
        settings_frame = ttk.Frame(cs_control_frame)
        settings_frame.pack(fill=tk.X, pady=5, padx=5)

        # Add settings controls
        ttk.Label(settings_frame, text="Top N Symbols:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(settings_frame, from_=5, to=50, increment=5, textvariable=self.cs_symbols_count,
                    width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(settings_frame, text="Timeframe:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(settings_frame, textvariable=self.cs_timeframe,
                     values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                     width=10).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(settings_frame, text="Candles:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(settings_frame, from_=20, to=200, increment=10, textvariable=self.cs_candles_count,
                    width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Add button to run analysis
        run_button = ttk.Button(settings_frame, text="Find Patterns", command=self.run_candlestick_analysis)
        run_button.grid(row=1, column=2, columnspan=2, sticky=tk.E, padx=5, pady=5)

        # Create results frame
        results_frame = ttk.LabelFrame(self.candlestick_tab, text="Pattern Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabbed interface for results and log
        cs_results_notebook = ttk.Notebook(results_frame)
        cs_results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabs for results and log
        self.patterns_results_frame = ttk.Frame(cs_results_notebook, padding="10")
        self.cs_log_frame = ttk.Frame(cs_results_notebook, padding="10")

        cs_results_notebook.add(self.patterns_results_frame, text="Results")
        cs_results_notebook.add(self.cs_log_frame, text="Log")

        # Create results treeview
        self.create_candlestick_treeview(self.patterns_results_frame)

        # Create log text area
        self.cs_log_text = scrolledtext.ScrolledText(self.cs_log_frame, wrap=tk.WORD)
        self.cs_log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar for Candlestick tab
        cs_status_frame = ttk.Frame(self.candlestick_tab)
        cs_status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        cs_status_frame.columnconfigure(0, weight=3)
        cs_status_frame.columnconfigure(1, weight=1)

        # Status message (left side)
        self.cs_status_var = tk.StringVar()
        self.cs_status_var.set("Ready")
        cs_status_label = ttk.Label(cs_status_frame, textvariable=self.cs_status_var, relief=tk.SUNKEN, anchor=tk.W)
        cs_status_label.grid(row=0, column=0, sticky=tk.W + tk.E)

        # Timer display (right side)
        self.cs_timer_var = tk.StringVar()
        self.cs_timer_var.set("Time: 0s")
        cs_timer_label = ttk.Label(cs_status_frame, textvariable=self.cs_timer_var, relief=tk.SUNKEN, anchor=tk.E)
        cs_timer_label.grid(row=0, column=1, sticky=tk.E)

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

        # Create treeview with columns based on whether freshness is checked
        if self.check_freshness.get():
            columns = ("Symbol", "Pattern", "Entry", "TP", "SL", "Side", "Volume 24h", "Freshness")
        else:
            columns = ("Symbol", "Pattern", "Entry", "TP", "SL", "Side", "Volume 24h")

        treeview = ttk.Treeview(frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)

        # Configure basic columns
        treeview.heading("Symbol", text="Symbol")
        treeview.heading("Pattern", text="Pattern")
        treeview.heading("Entry", text="Entry")
        treeview.heading("TP", text="Take Profit")
        treeview.heading("SL", text="Stop Loss")
        treeview.heading("Side", text="Side")
        treeview.heading("Volume 24h", text="Volume 24h ($)")

        treeview.column("Symbol", width=100)
        treeview.column("Pattern", width=150)
        treeview.column("Entry", width=100)
        treeview.column("TP", width=100)
        treeview.column("SL", width=100)
        treeview.column("Side", width=50)
        treeview.column("Volume 24h", width=120)

        # Add freshness column if needed
        if self.check_freshness.get():
            treeview.heading("Freshness", text="Candles Since")
            treeview.column("Freshness", width=100)

        # Configure scrollbar
        scrollbar.config(command=treeview.yview)

        # Pack treeview
        treeview.pack(fill=tk.BOTH, expand=True)

        # Store reference to treeview
        if trend_type == "uptrends":
            self.uptrends_treeview = treeview
        else:
            self.downtrends_treeview = treeview

    def log_hhhl(self, message):
        """
        Add a message to the HH/HL log.

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Insert log message in a thread-safe way
        self.root.after(0, lambda: self.hhhl_log_text.insert(tk.END, log_message))
        self.root.after(0, lambda: self.hhhl_log_text.see(tk.END))

    def log_cs(self, message):
        """
        Add a message to the Candlestick log.

        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Insert log message in a thread-safe way
        self.root.after(0, lambda: self.cs_log_text.insert(tk.END, log_message))
        self.root.after(0, lambda: self.cs_log_text.see(tk.END))

    def update_hhhl_status(self, message):
        """
        Update the HH/HL status bar.

        Args:
            message: Status message
        """
        self.hhhl_status_var.set(message)

    def update_cs_status(self, message):
        """
        Update the Candlestick status bar.

        Args:
            message: Status message
        """
        self.cs_status_var.set(message)

    def update_hhhl_timer(self):
        """Update the HH/HL timer display while analysis is running."""
        if not self.running:
            return

        elapsed = time.time() - self.start_time
        self.hhhl_timer_var.set(f"Time: {self.format_time(elapsed)}")
        self.root.after(100, self.update_hhhl_timer)  # Update every 100ms

    def update_cs_timer(self):
        """Update the Candlestick timer display while analysis is running."""
        if not self.running:
            return

        elapsed = time.time() - self.start_time
        self.cs_timer_var.set(f"Time: {self.format_time(elapsed)}")
        self.root.after(100, self.update_cs_timer)  # Update every 100ms

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

    def run_hhhl_analysis(self):
        """Run the HH/HL trading analysis in a separate thread."""
        if self.running:
            messagebox.showinfo("Running", "Analysis is already running. Please wait.")
            return

        # Clear previous results
        for item in self.uptrends_treeview.get_children():
            self.uptrends_treeview.delete(item)

        for item in self.downtrends_treeview.get_children():
            self.downtrends_treeview.delete(item)

        self.notrends_listbox.delete(0, tk.END)

        self.hhhl_log_text.delete(1.0, tk.END)

        # Get parameters
        n_symbols = self.symbols_count.get()
        tp_percent = self.tp_percent.get()
        sl_percent = self.sl_percent.get()

        # Start timer
        self.start_time = time.time()
        self.hhhl_timer_var.set("Time: 0s")
        self.update_hhhl_timer()

        # Start analysis thread
        self.running = True
        self.analysis_thread = threading.Thread(target=self.hhhl_analysis_task,
                                                args=(n_symbols, tp_percent, sl_percent))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def hhhl_analysis_task(self, n_symbols, tp_percent, sl_percent):
        """
        Perform HH/HL analysis in a separate thread.

        Args:
            n_symbols: Number of top symbols to analyze
            tp_percent: Take profit percentage
            sl_percent: Stop loss percentage
        """
        try:
            self.update_hhhl_status("Fetching symbols...")
            self.log_hhhl(f"Starting HH/HL analysis with top {n_symbols} symbols, TP: {tp_percent}%, SL: {sl_percent}%")

            # Create analysis parameters
            params = HHHLAnalysisParams(
                symbols_count=n_symbols,
                tp_percent=tp_percent,
                sl_percent=sl_percent,
                check_freshness=self.check_freshness.get()
            )

            # Run analysis through the service
            result = self.analysis_service.run_hhhl_analysis(params)

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

    def update_hhhl_results(self, result: HHHLResult):
        """
        Update the HH/HL results in the UI.

        Args:
            result: HH/HL analysis result
        """
        # Configure tags first
        self.root.after(0, lambda: self.uptrends_treeview.tag_configure("strength3",
                                                                        background="#d4ffcc"))  # Light green
        self.root.after(0, lambda: self.uptrends_treeview.tag_configure("strength2", background="white"))
        self.root.after(0, lambda: self.downtrends_treeview.tag_configure("strength3",
                                                                          background="#ffcccc"))  # Light red
        self.root.after(0, lambda: self.downtrends_treeview.tag_configure("strength2", background="white"))

        # Update uptrends treeview
        for data in result.uptrends:
            # Format display data
            symbol = data['symbol']
            pattern = data['pattern']
            price = data['price']
            tp = data['tp']
            sl = data['sl']
            side = data['side']
            strength = data['strength']
            volume_formatted = data['volume_formatted']

            # Create tuple for treeview
            display_data = [symbol, pattern, price, tp, sl, side, volume_formatted]

            # Add freshness if present
            if 'freshness' in data and self.check_freshness.get():
                display_data.append(str(data['freshness']))

            tag = f"strength{min(strength, 3)}"  # Use strength3 tag for any strength >= 3

            # Insert with appropriate tag
            self.root.after(0, lambda d=display_data, t=tag:
            self.uptrends_treeview.insert("", tk.END, values=self.format_values_with_freshness(d), tags=(t,)))

        # Update downtrends treeview
        for data in result.downtrends:
            # Format display data
            symbol = data['symbol']
            pattern = data['pattern']
            price = data['price']
            tp = data['tp']
            sl = data['sl']
            side = data['side']
            strength = data['strength']
            volume_formatted = data['volume_formatted']

            # Create tuple for treeview
            display_data = [symbol, pattern, price, tp, sl, side, volume_formatted]

            # Add freshness if present
            if 'freshness' in data and self.check_freshness.get():
                display_data.append(str(data['freshness']))

            tag = f"strength{min(strength, 3)}"  # Use strength3 tag for any strength >= 3

            # Insert with appropriate tag
            self.root.after(0, lambda d=display_data, t=tag:
            self.downtrends_treeview.insert("", tk.END, values=self.format_values_with_freshness(d), tags=(t,)))

        # Update no trends listbox
        for symbol in result.no_trends:
            self.root.after(0, lambda s=symbol: self.notrends_listbox.insert(tk.END, s))

        # Update status and log
        formatted_time = self.format_time(result.execution_time)

        self.update_hhhl_status(
            f"Analysis complete in {formatted_time}. Found {len(result.uptrends)} uptrends, {len(result.downtrends)} downtrends, {len(result.no_trends)} no trends")
        self.log_hhhl(
            f"Analysis complete in {formatted_time}. Found {len(result.uptrends)} uptrends, {len(result.downtrends)} downtrends, {len(result.no_trends)} no trends")

    def run_candlestick_analysis(self):
        """Run the candlestick pattern analysis in a separate thread."""
        if self.running:
            messagebox.showinfo("Running", "Analysis is already running. Please wait.")
            return

        # Check if at least one pattern is selected
        if not any(var.get() for var in self.selected_patterns.values()):
            messagebox.showinfo("No Patterns Selected", "Please select at least one candlestick pattern.")
            return

        # Clear previous results
        for item in self.candlestick_treeview.get_children():
            self.candlestick_treeview.delete(item)

        self.cs_log_text.delete(1.0, tk.END)

        # Get parameters
        n_symbols = self.cs_symbols_count.get()
        timeframe = self.cs_timeframe.get()
        candles_count = self.cs_candles_count.get()

        # Start timer
        self.start_time = time.time()
        self.cs_timer_var.set("Time: 0s")
        self.update_cs_timer()

        # Start analysis thread
        self.running = True
        self.analysis_thread = threading.Thread(target=self.candlestick_analysis_task,
                                                args=(n_symbols, timeframe, candles_count))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def candlestick_analysis_task(self, n_symbols, timeframe, candles_count):
        """
        Perform candlestick pattern analysis in a separate thread.

        Args:
            n_symbols: Number of top symbols to analyze
            timeframe: Candlestick timeframe (e.g., '1h', '4h')
            candles_count: Number of candles to analyze
        """
        try:
            self.update_cs_status("Fetching symbols...")
            self.log_cs(f"Starting candlestick pattern analysis with top {n_symbols} symbols")

            # Get selected patterns
            selected_pattern_names = [name for name, var in self.selected_patterns.items() if var.get()]
            self.log_cs(f"Selected patterns: {', '.join(selected_pattern_names)}")

            # Get top symbols
            self.log_cs("Fetching top volume symbols...")
            symbols = self.client.get_top_volume_symbols(limit=n_symbols)

            if not symbols:
                self.log_cs("No symbols found. Analysis failed.")
                self.update_cs_status("Analysis failed - no symbols found")
                self.running = False
                return

            self.log_cs(f"Found {len(symbols)} symbols")

            # Store all results
            all_patterns = []

            # Analyze each symbol
            for i, symbol in enumerate(symbols):
                self.update_cs_status(f"Analyzing {symbol} ({i + 1}/{len(symbols)})...")
                self.log_cs(f"Analyzing {symbol}...")

                try:
                    # Get historical price data
                    klines = self.client.get_klines(symbol, interval=timeframe, limit=candles_count)

                    if not klines:
                        self.log_cs(f"No data available for {symbol}, skipping")
                        continue

                    # Convert to DataFrame
                    import pandas as pd
                    df = pd.DataFrame(klines)

                    # Add this after getting the klines data in candlestick_analysis_task
                    ticker = self.client.get_ticker(symbol)
                    usd_volume = 0
                    if ticker and "volume_24h" in ticker and "last_price" in ticker:
                        usd_volume = ticker["volume_24h"] * ticker["last_price"]

                    # Then when adding hammer patterns
                    if 'hammer' in selected_pattern_names:
                        hammers = self.pattern_finder.find_hammers(df)
                        if hammers:
                            patterns_found = True
                            self.log_cs(f"Found {len(hammers)} hammer patterns for {symbol}")
                            for hammer in hammers:
                                # Add symbol, pattern type and volume
                                hammer['symbol'] = symbol
                                hammer['pattern_type'] = 'Hammer'
                                hammer['usd_volume'] = usd_volume
                                all_patterns.append(hammer)

                    # Check if data has required columns
                    if not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close']):
                        self.log_cs(f"Data for {symbol} is missing required columns, skipping")
                        self.log_cs(f"Available columns: {df.columns.tolist()}")
                        continue

                    # Find patterns
                    patterns_found = False

                    # Hammer pattern detection
                    if 'hammer' in selected_pattern_names:
                        hammers = self.pattern_finder.find_hammers(df)
                        if hammers:
                            patterns_found = True
                            self.log_cs(f"Found {len(hammers)} hammer patterns for {symbol}")
                            for hammer in hammers:
                                # Add symbol and pattern type
                                hammer['symbol'] = symbol
                                hammer['pattern_type'] = 'Hammer'
                                all_patterns.append(hammer)
                        else:
                            self.log_cs(f"No hammer patterns found for {symbol}")

                    # Add more pattern detection here as they're implemented
                    # For now, we have only hammer pattern, other patterns will be added later

                    if not patterns_found:
                        self.log_cs(f"No selected patterns found for {symbol}")

                except Exception as e:
                    self.log_cs(f"Error analyzing {symbol}: {str(e)}")

                # Small delay to avoid hammering the API
                time.sleep(0.2)

            # Sort patterns by timestamp (newest first)
            all_patterns.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)

            # Update results in GUI
            if all_patterns:
                self.update_candlestick_results(all_patterns)
                self.log_cs(f"Found {len(all_patterns)} patterns across all symbols")
            else:
                self.log_cs("No patterns found across all analyzed symbols")

            # Final timing
            elapsed_time = time.time() - self.start_time
            formatted_time = self.format_time(elapsed_time)

            # Update status
            pattern_count = len(all_patterns)
            self.update_cs_status(f"Analysis complete in {formatted_time}. Found {pattern_count} patterns.")

        except Exception as e:
            self.log_cs(f"Error during analysis: {str(e)}")
            self.update_cs_status("Analysis failed")

        finally:
            # Ensure timer shows final time
            elapsed_time = time.time() - self.start_time
            self.cs_timer_var.set(f"Time: {self.format_time(elapsed_time)}")
            self.running = False

    def update_candlestick_results(self, patterns):
        """
        Update the candlestick treeview with found patterns.

        Args:
            patterns: List of pattern dictionaries
        """
        for pattern in patterns:
            # Format values
            symbol = pattern['symbol']
            pattern_type = pattern['pattern_type']
            price = f"${pattern['close']:.4f}"

            # Format timestamp
            timestamp = pattern['timestamp'].strftime("%Y-%m-%d %H:%M") if pattern['timestamp'] else "N/A"

            is_bullish = "Yes" if pattern['is_bullish'] else "No"

            # Determine quality based on strength
            strength = pattern['strength']
            if strength >= 0.7:
                quality = "Good"
                tag = "good"
            elif strength >= 0.4:
                quality = "Moderate"
                tag = "moderate"
            else:
                quality = "Poor"
                tag = "poor"

            # Format volume
            usd_volume = pattern.get('usd_volume', 0)
            volume_formatted = f"${usd_volume / 1000000:.2f}M" if usd_volume >= 1000000 else f"${usd_volume / 1000:.2f}K"

            # Insert into treeview
            values = (symbol, pattern_type, price, timestamp, is_bullish, quality, volume_formatted)
            self.root.after(0, lambda v=values, t=tag:
            self.candlestick_treeview.insert("", tk.END, values=v, tags=(t,)))

    def format_values(self, data):
        """
        Format values for display in treeview.

        Args:
            data: Tuple of (symbol, pattern, price, tp, sl, side)

        Returns:
            Tuple of formatted values
        """
        symbol, pattern, price, tp, sl, side, volume_formatted = data

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

        return symbol, pattern, price_str, tp_str, sl_str, side, volume_formatted

    def format_values_with_freshness(self, data):
        """
        Format values for display in treeview, including freshness if present.

        Args:
            data: List of data values

        Returns:
            Tuple of formatted values
        """
        # Check if freshness is included in the data
        has_freshness = len(data) > 7

        if has_freshness:
            symbol, pattern, price, tp, sl, side, volume_formatted, freshness = data
        else:
            symbol, pattern, price, tp, sl, side, volume_formatted = data

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

        if has_freshness:
            # Return with freshness
            return symbol, pattern, price_str, tp_str, sl_str, side, volume_formatted, freshness
        else:
            # Return without freshness
            return symbol, pattern, price_str, tp_str, sl_str, side, volume_formatted

def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = TabbedTradingBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
