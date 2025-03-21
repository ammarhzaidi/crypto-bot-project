"""
Tab for displaying and tracking top gainers and losers in the cryptocurrency market.
Integrates with the database for historical data storage and retrieval.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.okx_tb_10 import OKXMarketAnalyzer
from src.analysis.db_analysis_service import DBAnalysisService
from src.market_data.okx_client import OKXClient
from src.repositories.movers_repository import MoversRepository
from src.utils.database import DatabaseManager


class TopMoversTab:
    """
    Tab for displaying and analyzing top gainers and losers with database persistence.
    """

    def test_okx_connection(self):
        """Test connection to OKX API."""
        try:
            self.log("Testing connection to OKX API...")

            import requests  # Import here if not already at the top of the file

            # Use the first base URL from the list
            base_url = "https://www.okx.com"
            url = f"{base_url}/api/v5/market/tickers"
            params = {"instType": "SPOT"}

            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json'
            }

            self.log(f"Sending request to {url}")
            response = requests.get(url, params=params, headers=headers, timeout=15)

            self.log(f"Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                self.log(f"API response code: {data.get('code')}")

                if data.get("code") == "0" and data.get("data"):
                    item_count = len(data["data"])
                    self.log(f"Received {item_count} items from API")

                    # Count USDT pairs
                    usdt_pairs = [item for item in data["data"] if item["instId"].endswith("-USDT")]
                    self.log(f"Found {len(usdt_pairs)} USDT trading pairs")

                    # Display first few pairs
                    if usdt_pairs:
                        sample = usdt_pairs[:5]
                        for pair in sample:
                            self.log(
                                f"Sample pair: {pair['instId']}, price: {pair.get('last')}, 24h change: {pair.get('change24h')}")

                    return True
                else:
                    self.log(f"API error: {data.get('msg', 'Unknown error')}")
            else:
                self.log(f"HTTP error: {response.status_code}")

            return False
        except Exception as e:
            self.log(f"Connection test error: {str(e)}")
            return False

    def __init__(self, parent, client=None, db_path="data/market_data.db"):
        """
        Initialize the top movers tab.

        Args:
            parent: Parent frame
            client: OKXClient instance (optional)
            db_path: Database file path
        """
        self.parent = parent
        self.client = client or OKXClient()
        self.db_path = db_path

        # Initialize analyzers
        self.market_analyzer = OKXMarketAnalyzer(min_volume=1000000)  # $1M minimum volume

        # Initialize database components
        self.db = DatabaseManager(db_path)
        self.movers_repo = MoversRepository(self.db)

        # Initialize DB-enabled analysis service
        self.db_analysis_service = DBAnalysisService(db_path=db_path)

        # Track running state
        self.running = False
        self.analysis_thread = None
        self.start_time = 0

        # Schedule variables
        self.scheduled_task = None
        self.is_scheduled = False
        self.schedule_interval = 60  # Default to 60 minutes

        # Import the tooltip manager
        from src.gui.components import get_tooltip_manager
        self.tooltip_manager = get_tooltip_manager()

        # Store timestamp data for tooltips
        self.comparison_timestamps = {
            "current": None,
            "previous": None,
            "symbols": {}  # Will store per-symbol timestamps
        }

        self.log("DEBUG: Tooltip manager initialized")

        # Create main frame
        self.frame = ttk.Frame(parent, padding="10")

        # Create main frame
        self.frame = ttk.Frame(parent, padding="10")

        # Create widgets
        self.create_widgets()

        # Pack the main frame
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Initial data load
        self.load_latest_data()

    def create_widgets(self):
        """Create all widgets for the tab."""
        # Create control panel
        self.create_control_panel()

        # Create results panel
        self.create_results_panel()

    def create_control_panel(self):
        """Create the control panel with buttons and options."""
        control_frame = ttk.LabelFrame(self.frame, text="Top Movers Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Configure grid
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)

        # Minimum volume filter
        ttk.Label(control_frame, text="Min Volume ($):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.min_volume_var = tk.DoubleVar(value=1000000)  # Default $1M
        volume_spinbox = ttk.Spinbox(
            control_frame, from_=100000, to=100000000, increment=100000,
            textvariable=self.min_volume_var, width=15
        )
        volume_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Number of movers to display
        ttk.Label(control_frame, text="Top N:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.top_n_var = tk.IntVar(value=10)
        top_n_spinbox = ttk.Spinbox(
            control_frame, from_=5, to=50, increment=5,
            textvariable=self.top_n_var, width=10
        )
        top_n_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Compare with previous timeframe
        ttk.Label(control_frame, text="Compare with (hours ago):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.compare_hours_var = tk.DoubleVar(value=4.0)
        compare_spinbox = ttk.Spinbox(
            control_frame, from_=1, to=24, increment=1,
            textvariable=self.compare_hours_var, width=10
        )
        compare_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Scheduling options
        ttk.Label(control_frame, text="Auto-update (minutes):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.schedule_var = tk.IntVar(value=60)
        schedule_spinbox = ttk.Spinbox(
            control_frame, from_=15, to=1440, increment=15,
            textvariable=self.schedule_var, width=10
        )
        schedule_spinbox.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W + tk.E, pady=10)

        # Run button
        self.run_button = ttk.Button(
            buttons_frame, text="Analyze Now",
            command=self.run_analysis
        )
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Schedule button
        self.schedule_button = ttk.Button(
            buttons_frame, text="Start Scheduling",
            command=self.toggle_scheduling
        )
        self.schedule_button.pack(side=tk.LEFT, padx=5)

        # Save to DB button
        self.save_button = ttk.Button(
            buttons_frame, text="Save to Database",
            command=self.save_to_database
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Refresh from DB button
        self.refresh_button = ttk.Button(
            buttons_frame, text="Load from Database",
            command=self.load_latest_data
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(buttons_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=10)

        # Next update label (for scheduling)
        self.next_update_var = tk.StringVar(value="")
        next_update_label = ttk.Label(buttons_frame, textvariable=self.next_update_var)
        next_update_label.pack(side=tk.RIGHT, padx=10)

        test_connection = ttk.Button(
            buttons_frame, text="Test API",
            command=self.test_okx_connection
        )
        test_connection.pack(side=tk.LEFT, padx=5)

    def create_results_panel(self):
        """Create the panel for displaying movers results."""
        # Create notebook for results tabs
        self.results_notebook = ttk.Notebook(self.frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabs
        self.gainers_frame = ttk.Frame(self.results_notebook, padding="10")
        self.losers_frame = ttk.Frame(self.results_notebook, padding="10")
        self.comparison_frame = ttk.Frame(self.results_notebook, padding="10")
        self.history_frame = ttk.Frame(self.results_notebook, padding="10")
        self.log_frame = ttk.Frame(self.results_notebook, padding="10")

        self.results_notebook.add(self.gainers_frame, text="Top Gainers")
        self.results_notebook.add(self.losers_frame, text="Top Losers")
        self.results_notebook.add(self.comparison_frame, text="Comparison")
        self.results_notebook.add(self.history_frame, text="History")
        self.results_notebook.add(self.log_frame, text="Log")

        # Create treeviews for each tab
        self.create_gainers_treeview()
        self.create_losers_treeview()
        self.create_comparison_treeviews()
        self.create_history_view()

        # Create log area
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def create_gainers_treeview(self):
        """Create treeview for top gainers."""
        # Create frame for treeview and scrollbar
        frame = ttk.Frame(self.gainers_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create treeview
        columns = ("Rank", "Symbol", "Price", "Change 24h", "Volume 24h", "Last Updated")
        self.gainers_treeview = ttk.Treeview(
            frame, columns=columns, show="headings",
            yscrollcommand=scrollbar.set
        )

        # Configure columns
        self.gainers_treeview.heading("Rank", text="#")
        self.gainers_treeview.heading("Symbol", text="Symbol")
        self.gainers_treeview.heading("Price", text="Price ($)")
        self.gainers_treeview.heading("Change 24h", text="Change 24h (%)")
        self.gainers_treeview.heading("Volume 24h", text="Volume 24h ($)")
        self.gainers_treeview.heading("Last Updated", text="Last Updated")

        self.gainers_treeview.column("Rank", width=40, anchor=tk.CENTER)
        self.gainers_treeview.column("Symbol", width=100)
        self.gainers_treeview.column("Price", width=100, anchor=tk.E)
        self.gainers_treeview.column("Change 24h", width=100, anchor=tk.E)
        self.gainers_treeview.column("Volume 24h", width=150, anchor=tk.E)
        self.gainers_treeview.column("Last Updated", width=150, anchor=tk.CENTER)

        # Configure scrollbar
        scrollbar.config(command=self.gainers_treeview.yview)

        # Pack treeview
        self.gainers_treeview.pack(fill=tk.BOTH, expand=True)

        # Configure tags for alternate row colors
        self.gainers_treeview.tag_configure('oddrow', background='#F5F5F5')
        self.gainers_treeview.tag_configure('evenrow', background='#FFFFFF')

        # Configure tag for highlighting
        self.gainers_treeview.tag_configure('highlight', background='#E6F3FF')

    def create_losers_treeview(self):
        """Create treeview for top losers."""
        # Create frame for treeview and scrollbar
        frame = ttk.Frame(self.losers_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create treeview
        columns = ("Rank", "Symbol", "Price", "Change 24h", "Volume 24h", "Last Updated")
        self.losers_treeview = ttk.Treeview(
            frame, columns=columns, show="headings",
            yscrollcommand=scrollbar.set
        )

        # Configure columns
        self.losers_treeview.heading("Rank", text="#")
        self.losers_treeview.heading("Symbol", text="Symbol")
        self.losers_treeview.heading("Price", text="Price ($)")
        self.losers_treeview.heading("Change 24h", text="Change 24h (%)")
        self.losers_treeview.heading("Volume 24h", text="Volume 24h ($)")
        self.losers_treeview.heading("Last Updated", text="Last Updated")

        self.losers_treeview.column("Rank", width=40, anchor=tk.CENTER)
        self.losers_treeview.column("Symbol", width=100)
        self.losers_treeview.column("Price", width=100, anchor=tk.E)
        self.losers_treeview.column("Change 24h", width=100, anchor=tk.E)
        self.losers_treeview.column("Volume 24h", width=150, anchor=tk.E)
        self.losers_treeview.column("Last Updated", width=150, anchor=tk.CENTER)

        # Configure scrollbar
        scrollbar.config(command=self.losers_treeview.yview)

        # Pack treeview
        self.losers_treeview.pack(fill=tk.BOTH, expand=True)

        # Configure tags for alternate row colors
        self.losers_treeview.tag_configure('oddrow', background='#F5F5F5')
        self.losers_treeview.tag_configure('evenrow', background='#FFFFFF')

        # Configure tag for highlighting
        self.losers_treeview.tag_configure('highlight', background='#FFEBEE')  # Light red for losers

    def create_comparison_treeviews(self):
        """Create treeviews for comparison tab."""
        # Create separate frames for gainers and losers comparisons
        gainers_comp_frame = ttk.LabelFrame(self.comparison_frame, text="Gainers Movement")
        gainers_comp_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        losers_comp_frame = ttk.LabelFrame(self.comparison_frame, text="Losers Movement")
        losers_comp_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create treeviews for each comparison
        # Gainers comparison treeview
        columns = ("Symbol", "Price", "Current %", "Previous %", "Acceleration", "Volume 24h")
        self.gainers_comp_treeview = ttk.Treeview(
            gainers_comp_frame, columns=columns, show="headings"
        )

        # Configure columns
        self.gainers_comp_treeview.heading("Symbol", text="Symbol")
        self.gainers_comp_treeview.heading("Price", text="Price ($)")
        self.gainers_comp_treeview.heading("Current %", text="Current %")
        self.gainers_comp_treeview.heading("Previous %", text="Previous %")
        self.gainers_comp_treeview.heading("Acceleration", text="Acceleration")
        self.gainers_comp_treeview.heading("Volume 24h", text="Volume 24h ($)")

        # Set column widths
        self.gainers_comp_treeview.column("Symbol", width=100)
        self.gainers_comp_treeview.column("Price", width=100, anchor=tk.E)
        self.gainers_comp_treeview.column("Current %", width=100, anchor=tk.E)
        self.gainers_comp_treeview.column("Previous %", width=100, anchor=tk.E)
        self.gainers_comp_treeview.column("Acceleration", width=100, anchor=tk.E)
        self.gainers_comp_treeview.column("Volume 24h", width=150, anchor=tk.E)

        # Add scrollbar
        gainers_scroll = ttk.Scrollbar(gainers_comp_frame, orient="vertical", command=self.gainers_comp_treeview.yview)
        self.gainers_comp_treeview.configure(yscrollcommand=gainers_scroll.set)

        # Pack elements
        gainers_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.gainers_comp_treeview.pack(fill=tk.BOTH, expand=True)

        # Losers comparison treeview
        self.losers_comp_treeview = ttk.Treeview(
            losers_comp_frame, columns=columns, show="headings"
        )

        # Configure columns (same as gainers)
        self.losers_comp_treeview.heading("Symbol", text="Symbol")
        self.losers_comp_treeview.heading("Price", text="Price ($)")
        self.losers_comp_treeview.heading("Current %", text="Current %")
        self.losers_comp_treeview.heading("Previous %", text="Previous %")
        self.losers_comp_treeview.heading("Acceleration", text="Acceleration")
        self.losers_comp_treeview.heading("Volume 24h", text="Volume 24h ($)")

        # Set column widths
        self.losers_comp_treeview.column("Symbol", width=100)
        self.losers_comp_treeview.column("Price", width=100, anchor=tk.E)
        self.losers_comp_treeview.column("Current %", width=100, anchor=tk.E)
        self.losers_comp_treeview.column("Previous %", width=100, anchor=tk.E)
        self.losers_comp_treeview.column("Acceleration", width=100, anchor=tk.E)
        self.losers_comp_treeview.column("Volume 24h", width=150, anchor=tk.E)

        # Add scrollbar
        losers_scroll = ttk.Scrollbar(losers_comp_frame, orient="vertical", command=self.losers_comp_treeview.yview)
        self.losers_comp_treeview.configure(yscrollcommand=losers_scroll.set)

        # Pack elements
        losers_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.losers_comp_treeview.pack(fill=tk.BOTH, expand=True)

        # Configure tags for positive/negative acceleration
        self.gainers_comp_treeview.tag_configure('positive', background='#E8F5E9')  # Light green
        self.gainers_comp_treeview.tag_configure('negative', background='#FFEBEE')  # Light red
        self.losers_comp_treeview.tag_configure('positive', background='#E8F5E9')  # Light green
        self.losers_comp_treeview.tag_configure('negative', background='#FFEBEE')  # Light red

        self.log("DEBUG: Adding tooltips to comparison treeviews")
        self.tooltip_manager.add_treeview_tooltip(
            self.gainers_comp_treeview,
            callback=self.get_comparison_tooltip_text
        )

        self.tooltip_manager.add_treeview_tooltip(
            self.gainers_comp_treeview,
            callback=self.get_comparison_tooltip_text
        )

        self.tooltip_manager.add_treeview_tooltip(
            self.losers_comp_treeview,
            callback=self.get_comparison_tooltip_text
        )
        self.log("DEBUG: Tooltips added to comparison treeviews")

    def create_history_view(self):
        """Create view for historical data."""
        # Create a sub-notebook for history types
        self.history_notebook = ttk.Notebook(self.history_frame)
        self.history_notebook.pack(fill=tk.BOTH, expand=True)

        # Create frames for different history views
        self.frequent_frame = ttk.Frame(self.history_notebook)
        self.persistent_frame = ttk.Frame(self.history_notebook)

        self.history_notebook.add(self.frequent_frame, text="Frequent Movers")
        self.history_notebook.add(self.persistent_frame, text="Persistent Movers")

        def load_persistent_movers(self):
            """Load and display persistent movers data."""
            try:
                # Get parameters
                min_days = self.history_days_var.get()

                self.log(f"Loading persistent movers data (min_days={min_days})...")

                # Get persistent symbols
                persistent_symbols = self.movers_repo.get_persistent_movers(min_days)

                # Clear current items
                for item in self.persistent_treeview.get_children():
                    self.persistent_treeview.delete(item)

                # Update treeview
                for symbol_data in persistent_symbols:
                    # Format values
                    symbol = symbol_data['symbol']
                    trend = "Uptrend" if symbol_data['trend'] == 'up' else "Downtrend"
                    days = str(symbol_data['days'])
                    start_price = f"${symbol_data['start_price']:.4f}"
                    current_price = f"${symbol_data['current_price']:.4f}"
                    total_change = f"{symbol_data['total_change']:+.2f}%"
                    avg_volume = self.format_volume(symbol_data['avg_volume'])

                    # Determine tag based on trend
                    tag = 'uptrend' if symbol_data['trend'] == 'up' else 'downtrend'

                    # Insert into treeview
                    self.persistent_treeview.insert(
                        "", tk.END,
                        values=(symbol, trend, days, start_price, current_price, total_change, avg_volume),
                        tags=(tag,)
                    )

                self.log(f"Loaded {len(persistent_symbols)} persistent movers")

            except Exception as e:
                self.log(f"Error loading persistent movers data: {str(e)}")

        def create_history_view(self):
            # Existing code remains the same until persistent_frame creation

            # Create treeview for persistent movers
            columns = ("Symbol", "Trend", "Days in Trend", "Start Price", "Current Price", "Total Change", "Avg Volume")
            self.persistent_treeview = ttk.Treeview(
                self.persistent_frame, columns=columns, show="headings"
            )

            # Configure columns
            self.persistent_treeview.heading("Symbol", text="Symbol")
            self.persistent_treeview.heading("Trend", text="Trend Direction")
            self.persistent_treeview.heading("Days in Trend", text="Consecutive Days")
            self.persistent_treeview.heading("Start Price", text="Start Price ($)")
            self.persistent_treeview.heading("Current Price", text="Current Price ($)")
            self.persistent_treeview.heading("Total Change", text="Total Change (%)")
            self.persistent_treeview.heading("Avg Volume", text="Avg Volume ($)")

            # Set column widths
            self.persistent_treeview.column("Symbol", width=100)
            self.persistent_treeview.column("Trend", width=100)
            self.persistent_treeview.column("Days in Trend", width=120, anchor=tk.CENTER)
            self.persistent_treeview.column("Start Price", width=100, anchor=tk.E)
            self.persistent_treeview.column("Current Price", width=100, anchor=tk.E)
            self.persistent_treeview.column("Total Change", width=120, anchor=tk.E)
            self.persistent_treeview.column("Avg Volume", width=120, anchor=tk.E)

            # Add scrollbar
            persistent_scroll = ttk.Scrollbar(self.persistent_frame, orient="vertical",
                                              command=self.persistent_treeview.yview)
            self.persistent_treeview.configure(yscrollcommand=persistent_scroll.set)

            # Pack elements
            persistent_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.persistent_treeview.pack(fill=tk.BOTH, expand=True)

            # Configure tags for up/down trends
            self.persistent_treeview.tag_configure('uptrend', background='#E8F5E9')  # Light green
            self.persistent_treeview.tag_configure('downtrend', background='#FFEBEE')  # Light red

        # Create treeview for frequent movers
        columns = ("Symbol", "Total Count", "Gainer Count", "Loser Count", "Latest Price", "Latest Change")
        self.frequent_treeview = ttk.Treeview(
            self.frequent_frame, columns=columns, show="headings"
        )

        # Configure columns
        self.frequent_treeview.heading("Symbol", text="Symbol")
        self.frequent_treeview.heading("Total Count", text="Total Appearances")
        self.frequent_treeview.heading("Gainer Count", text="As Gainer")
        self.frequent_treeview.heading("Loser Count", text="As Loser")
        self.frequent_treeview.heading("Latest Price", text="Latest Price ($)")
        self.frequent_treeview.heading("Latest Change", text="Latest Change (%)")

        # Set column widths
        self.frequent_treeview.column("Symbol", width=100)
        self.frequent_treeview.column("Total Count", width=120, anchor=tk.CENTER)
        self.frequent_treeview.column("Gainer Count", width=100, anchor=tk.CENTER)
        self.frequent_treeview.column("Loser Count", width=100, anchor=tk.CENTER)
        self.frequent_treeview.column("Latest Price", width=120, anchor=tk.E)
        self.frequent_treeview.column("Latest Change", width=120, anchor=tk.E)

        # Add scrollbar
        frequent_scroll = ttk.Scrollbar(self.frequent_frame, orient="vertical", command=self.frequent_treeview.yview)
        self.frequent_treeview.configure(yscrollcommand=frequent_scroll.set)

        # Pack elements
        frequent_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.frequent_treeview.pack(fill=tk.BOTH, expand=True)

        # Create history controls
        history_controls = ttk.Frame(self.history_frame)
        history_controls.pack(fill=tk.X, pady=5)

        # Days selector
        ttk.Label(history_controls, text="Days to look back:").pack(side=tk.LEFT, padx=5)
        self.history_days_var = tk.IntVar(value=7)
        history_days = ttk.Spinbox(
            history_controls, from_=1, to=30, increment=1,
            textvariable=self.history_days_var, width=5
        )
        history_days.pack(side=tk.LEFT, padx=5)

        # Minimum count selector
        ttk.Label(history_controls, text="Min. appearances:").pack(side=tk.LEFT, padx=5)
        self.min_count_var = tk.IntVar(value=3)
        min_count = ttk.Spinbox(
            history_controls, from_=2, to=20, increment=1,
            textvariable=self.min_count_var, width=5
        )
        min_count.pack(side=tk.LEFT, padx=5)

        # Refresh button
        refresh_history = ttk.Button(
            history_controls, text="Refresh History",
            command=self.load_historical_data
        )
        refresh_history.pack(side=tk.LEFT, padx=10)

    def log(self, message):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Insert log message in a thread-safe way
        self.parent.after(0, lambda: self.log_text.insert(tk.END, log_message))
        self.parent.after(0, lambda: self.log_text.see(tk.END))

    def update_status(self, message):
        """Update the status label."""
        self.status_var.set(message)

    def format_volume(self, volume: float) -> str:
        """Format volume value for display."""
        if volume >= 1000000000:  # Billions
            return f"${volume / 1000000000:.2f}B"
        elif volume >= 1000000:  # Millions
            return f"${volume / 1000000:.2f}M"
        elif volume >= 1000:  # Thousands
            return f"${volume / 1000:.2f}K"
        else:
            return f"${volume:.2f}"

    def run_analysis(self):
        """Run the market analysis in a separate thread."""
        if self.running:
            messagebox.showinfo("Running", "Analysis is already running. Please wait.")
            return

        # Clear previous results
        self.clear_treeviews()

        # Get parameters
        min_volume = self.min_volume_var.get()
        compare_hours = self.compare_hours_var.get()

        # Set running state
        self.running = True
        self.update_status("Analyzing market data...")
        self.run_button.config(state=tk.DISABLED)

        # Start analysis thread
        self.analysis_thread = threading.Thread(
            target=self.analysis_task,
            args=(min_volume, compare_hours)
        )
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def analysis_task(self, min_volume, compare_hours):
        """Run the analysis in a background thread."""
        try:
            self.log(f"Starting market analysis with min_volume=${min_volume:.2f}, compare_hours={compare_hours}")

            # Initialize the market analyzer with the specified minimum volume
            analyzer = OKXMarketAnalyzer(min_volume=min_volume)

            # Get top gainers and losers
            self.log("Fetching current market data...")
            top_n = self.top_n_var.get()
            self.log(f"Using Top N value: {top_n}")
            top_gainers, top_losers = analyzer.get_top_gainers_losers(limit=top_n)

            # Store results for later saving to database
            self.current_gainers = top_gainers
            self.current_losers = top_losers

            # Update UI with results
            self.parent.after(0, lambda: self.update_gainers_treeview(top_gainers))
            self.parent.after(0, lambda: self.update_losers_treeview(top_losers))

            # Compare with previous period if requested
            # Compare with previous period if requested
            if compare_hours > 0:
                self.log(f"Comparing with data from {compare_hours} hours ago...")

                # Store current timestamp
                current_time = datetime.now()
                self.comparison_timestamps["current"] = current_time

                # For each symbol in the comparison, store its timestamp
                gainers_comparison, losers_comparison = analyzer.compare_with_previous(hours_ago=compare_hours)

                # Store timestamps for each symbol
                for _, row in gainers_comparison.iterrows():
                    symbol = row['symbol']
                    if symbol not in self.comparison_timestamps["symbols"]:
                        self.comparison_timestamps["symbols"][symbol] = {"previous": None, "current": None}

                    # Set previous timestamp if it exists, otherwise use current time - compare_hours
                    prev_time = self.comparison_timestamps["symbols"].get(symbol, {}).get("current")
                    if prev_time is None:
                        prev_time = current_time - timedelta(hours=compare_hours)

                    # Update timestamps
                    self.comparison_timestamps["symbols"][symbol] = {
                        "previous": prev_time,
                        "current": current_time
                    }

                # Do the same for losers
                for _, row in losers_comparison.iterrows():
                    symbol = row['symbol']
                    if symbol not in self.comparison_timestamps["symbols"]:
                        self.comparison_timestamps["symbols"][symbol] = {"previous": None, "current": None}

                    prev_time = self.comparison_timestamps["symbols"].get(symbol, {}).get("current")
                    if prev_time is None:
                        prev_time = current_time - timedelta(hours=compare_hours)

                    # Update timestamps
                    self.comparison_timestamps["symbols"][symbol] = {
                        "previous": prev_time,
                        "current": current_time
                    }

                # Update comparison treeviews
                self.parent.after(0, lambda: self.update_comparison_treeviews(gainers_comparison, losers_comparison))

        except Exception as e:
            self.log(f"Error during analysis: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            self.update_status("Analysis failed.")

        finally:
            # Reset running state
            self.running = False
            self.run_button.config(state=tk.NORMAL)

    def clear_treeviews(self):
        """Clear all treeviews."""
        for treeview in [self.gainers_treeview, self.losers_treeview,
                         self.gainers_comp_treeview, self.losers_comp_treeview]:
            for item in treeview.get_children():
                treeview.delete(item)

    def update_gainers_treeview(self, gainers_df):
        """Update the gainers treeview with data."""
        # Clear existing items
        for item in self.gainers_treeview.get_children():
            self.gainers_treeview.delete(item)

        if gainers_df.empty:
            return

        # Format and insert data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        for i, (_, row) in enumerate(gainers_df.iterrows(), 1):
            # Format values
            symbol = row['symbol']
            price = f"${row['price']:.4f}"
            change = f"+{row['change_24h']:.2f}%" if 'change_24h' in row else "N/A"
            volume = self.format_volume(row['volume_24h']) if 'volume_24h' in row else "N/A"

            # Alternate row colors
            tag = 'oddrow' if i % 2 == 1 else 'evenrow'

            # Insert into treeview
            self.gainers_treeview.insert("", tk.END, values=(i, symbol, price, change, volume, timestamp), tags=(tag,))

    def update_losers_treeview(self, losers_df):
        """Update the losers treeview with data."""
        # Clear existing items
        for item in self.losers_treeview.get_children():
            self.losers_treeview.delete(item)

        if losers_df.empty:
            return

        # Format and insert data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        for i, (_, row) in enumerate(losers_df.iterrows(), 1):
            # Format values
            symbol = row['symbol']
            price = f"${row['price']:.4f}"
            change = f"{row['change_24h']:.2f}%" if 'change_24h' in row else "N/A"
            volume = self.format_volume(row['volume_24h']) if 'volume_24h' in row else "N/A"

            # Alternate row colors
            tag = 'oddrow' if i % 2 == 1 else 'evenrow'

            # Insert into treeview
            self.losers_treeview.insert("", tk.END, values=(i, symbol, price, change, volume, timestamp), tags=(tag,))

    def update_comparison_treeviews(self, gainers_comparison, losers_comparison):
        """Update the comparison treeviews with data."""
        # Clear existing items
        for treeview in [self.gainers_comp_treeview, self.losers_comp_treeview]:
            for item in treeview.get_children():
                treeview.delete(item)

        # Update gainers comparison
        if not gainers_comparison.empty:
            for _, row in gainers_comparison.iterrows():
                # Format values
                symbol = row['symbol']
                price = f"${row['price']:.4f}"
                current = f"+{row['change_24h_current']:.2f}%"
                previous = f"+{row['change_24h_previous']:.2f}%"

                # Handle acceleration which might be a string or a number
                if isinstance(row['acceleration'], (int, float)):
                    acceleration = f"{row['acceleration']:+.2f}%"
                else:
                    acceleration = str(row['acceleration'])  # Handle string case

                volume = self.format_volume(row['volume_24h'])

                # Determine tag based on acceleration
                if isinstance(row['acceleration'], (int, float)):
                    tag = 'positive' if row['acceleration'] >= 0 else 'negative'
                else:
                    tag = 'positive'  # Default tag for non-numeric values

                # Insert into treeview
                self.gainers_comp_treeview.insert(
                    "", tk.END,
                    values=(symbol, price, current, previous, acceleration, volume),
                    tags=(tag,)
                )

        # Update losers comparison
        if not losers_comparison.empty:
            for _, row in losers_comparison.iterrows():
                # Format values
                symbol = row['symbol']
                price = f"${row['price']:.4f}"
                current = f"{row['change_24h_current']:.2f}%"
                previous = f"{row['change_24h_previous']:.2f}%"

                # Handle acceleration which might be a string or a number
                if isinstance(row['acceleration'], (int, float)):
                    acceleration = f"{row['acceleration']:+.2f}%"
                else:
                    acceleration = str(row['acceleration'])  # Handle string case

                volume = self.format_volume(row['volume_24h'])

                # Determine tag based on acceleration
                if isinstance(row['acceleration'], (int, float)):
                    tag = 'negative' if row['acceleration'] <= 0 else 'positive'
                else:
                    tag = 'negative'  # Default tag for non-numeric values

                # Insert into treeview
                self.losers_comp_treeview.insert(
                    "", tk.END,
                    values=(symbol, price, current, previous, acceleration, volume),
                    tags=(tag,)
                )

    def get_comparison_tooltip_text(self, item_id, column_id, value):
        """Generate tooltip text for comparison data cells with real timestamp data."""
        # Get the symbol for this row
        values = self.gainers_comp_treeview.item(item_id, "values")
        if not values or len(values) < 1:
            # Try the losers treeview
            values = self.losers_comp_treeview.item(item_id, "values")
            if not values or len(values) < 1:
                return "Symbol not found"

        symbol = values[0]

        # Get timestamps for this symbol
        symbol_timestamps = self.comparison_timestamps["symbols"].get(symbol, {})

        if not symbol_timestamps:
            return "No timestamp data available for this symbol"

        current_ts = symbol_timestamps.get("current")
        previous_ts = symbol_timestamps.get("previous")

        # Format based on column
        if "Current" in str(column_id):
            if current_ts:
                formatted_ts = current_ts.strftime("%d-%m-%Y, %H:%M:%S")
                return f"Current data recorded at:\n{formatted_ts}\n(Pakistan time)"
            else:
                return "Timestamp not available"

        elif "Previous" in str(column_id):
            if previous_ts:
                formatted_ts = previous_ts.strftime("%d-%m-%Y, %H:%M:%S")
                return f"Previous data recorded at:\n{formatted_ts}\n(Pakistan time)"
            else:
                # For new symbols
                if "New" in str(value):
                    return "This symbol is new in the comparison"
                return "Previous timestamp not available"

        elif "Accel" in str(column_id):
            # Calculate time difference between current and previous
            if current_ts and previous_ts:
                time_diff = current_ts - previous_ts

                # Format time difference nicely
                days = time_diff.days
                seconds = time_diff.seconds
                hours, remainder = divmod(seconds, 3600)
                minutes, seconds = divmod(remainder, 60)

                time_str = ""
                if days > 0:
                    time_str += f"{days}d "
                if hours > 0 or days > 0:
                    time_str += f"{hours}h "
                if minutes > 0 or hours > 0 or days > 0:
                    time_str += f"{minutes}m "
                time_str += f"{seconds}s"

                return f"Time elapsed between measurements:\n{time_str}"
            else:
                return "Time difference cannot be calculated"

        # No tooltip for other columns
        return None

    def save_to_database(self):
        """Save current analysis results to the database."""
        if not hasattr(self, 'current_gainers') or not hasattr(self, 'current_losers'):
            messagebox.showinfo("No Data", "No current data to save. Run an analysis first.")
            return

        try:
            self.log("Saving analysis results to database...")

            # Convert to DataFrames if needed
            gainers_df = self.current_gainers if isinstance(self.current_gainers,
                                                            pd.DataFrame) else pd.DataFrame(
                self.current_gainers)
            losers_df = self.current_losers if isinstance(self.current_losers, pd.DataFrame) else pd.DataFrame(
                self.current_losers)

            # Save to database using repository
            success = self.movers_repo.save_top_movers(gainers_df, losers_df)

            # Also save timestamp data to a file for persistence
            import json
            import os
            from datetime import datetime

            # Convert timestamps to strings for JSON serialization
            timestamp_data = {}
            for symbol, ts_data in self.comparison_timestamps["symbols"].items():
                timestamp_data[symbol] = {
                    "current": ts_data["current"].isoformat() if ts_data["current"] else None,
                    "previous": ts_data["previous"].isoformat() if ts_data["previous"] else None
                }

            # Make sure the directory exists
            os.makedirs("data", exist_ok=True)

            # Save to file
            with open("data/comparison_timestamps.json", "w") as f:
                json.dump(timestamp_data, f)

            self.log("Timestamp data saved to file")

            if success:
                self.log("Data saved successfully to database")
                self.update_status("Data saved to database.")

                # Reload frequent movers data to reflect new additions
                self.load_historical_data()
            else:
                self.log("Failed to save data to database")
                self.update_status("Failed to save to database.")

        except Exception as e:
            self.log(f"Error saving to database: {str(e)}")
            self.update_status("Error saving to database.")

    def load_latest_data(self):
        """Load the most recent data from the database."""
        try:
            self.log("Loading latest data from database...")

            # Use repository to get latest data
            gainers_df, losers_df = self.movers_repo.get_latest_movers()

            if gainers_df.empty and losers_df.empty:
                self.log("No data found in database")
                self.update_status("No data in database.")
                return

            # Update treeviews with loaded data
            self.update_gainers_treeview(gainers_df)
            self.update_losers_treeview(losers_df)

            # Store the loaded data
            self.current_gainers = gainers_df
            self.current_losers = losers_df

            # Log success
            timestamp = gainers_df['timestamp'].iloc[0] if not gainers_df.empty else (
                losers_df['timestamp'].iloc[0] if not losers_df.empty else "Unknown"
            )
            self.log(f"Loaded data from {timestamp}")
            self.update_status(f"Loaded data from {timestamp}")

        except Exception as e:
            self.log(f"Error loading from database: {str(e)}")
            self.update_status("Error loading from database.")

    def load_historical_data(self):
        """Load historical data from the database."""
        try:
            # Get parameters
            days = self.history_days_var.get()
            min_count = self.min_count_var.get()

            self.log(f"Loading historical data (days={days}, min_count={min_count})...")

            # Get frequent symbols
            frequent_symbols = self.movers_repo.get_frequent_symbols(days, min_count)


            # Clear current items
            for item in self.frequent_treeview.get_children():
                self.frequent_treeview.delete(item)

            # Update treeview
            for symbol_data in frequent_symbols:
                symbol = symbol_data['symbol']
                total_count = symbol_data['total_count']
                gainer_count = symbol_data.get('gainers_count', 0)
                loser_count = symbol_data.get('losers_count', 0)

                # Format price and change if available
                price = f"${symbol_data.get('latest_price', 0):.4f}" if 'latest_price' in symbol_data else "N/A"
                change = f"{symbol_data.get('latest_change', 0):.2f}%" if 'latest_change' in symbol_data else "N/A"

                # Insert into treeview
                self.frequent_treeview.insert(
                    "", tk.END,
                    values=(symbol, total_count, gainer_count, loser_count, price, change)
                )

            # Log results
            self.log(f"Loaded {len(frequent_symbols)} frequent movers")

            self.load_persistent_movers()

        except Exception as e:
            self.log(f"Error loading historical data: {str(e)}")

    def toggle_scheduling(self):
        """Toggle scheduled analysis on/off."""
        if self.is_scheduled:
            # Turn off scheduling
            self.is_scheduled = False
            self.schedule_button.config(text="Start Scheduling")
            self.next_update_var.set("")

            # Cancel any pending task
            if self.scheduled_task is not None:
                self.parent.after_cancel(self.scheduled_task)
                self.scheduled_task = None

            self.log("Scheduled analysis stopped")
        else:
            # Turn on scheduling
            self.is_scheduled = True
            self.schedule_button.config(text="Stop Scheduling")

            # Get interval
            self.schedule_interval = self.schedule_var.get()

            # Schedule first run
            self.schedule_next_run()

            self.log(f"Scheduled analysis started (every {self.schedule_interval} minutes)")

    def schedule_next_run(self):
        """Schedule the next analysis run."""
        if not self.is_scheduled:
            return

        # Calculate next run time
        next_run = datetime.now() + timedelta(minutes=self.schedule_interval)
        next_run_str = next_run.strftime("%H:%M:%S")

        # Update next update label
        self.next_update_var.set(f"Next update: {next_run_str}")

        # Convert minutes to milliseconds for the after call
        interval_ms = self.schedule_interval * 60 * 1000

        # Schedule the task
        self.scheduled_task = self.parent.after(interval_ms, self.scheduled_run)

        self.log(f"Next analysis scheduled for {next_run_str}")

    def scheduled_run(self):
        """Run analysis as part of scheduled task."""
        if not self.is_scheduled:
            return

        # Run the analysis if not already running
        if not self.running:
            self.log("Running scheduled analysis...")
            self.run_analysis()
        else:
            self.log("Previous analysis still running, skipping this scheduled run")
            # Schedule next run anyway
            self.schedule_next_run()

    def create_top_movers_tab(notebook, client=None, db_path="data/market_data.db"):
        """
        Create and add a top movers tab to a notebook.

        Args:
            notebook: ttk.Notebook widget
            client: OKXClient instance (optional)
            db_path: Database file path

        Returns:
            TopMoversTab instance
        """
        top_movers_frame = ttk.Frame(notebook)
        top_movers_tab = TopMoversTab(top_movers_frame, client=client, db_path=db_path)
        notebook.add(top_movers_frame, text="Top Movers")
        return top_movers_tab

        # Testing code
    if __name__ == "__main__":
        # Create a standalone application for testing
        root = tk.Tk()
        root.title("Top Movers Tracker")
        root.geometry("1000x700")

        # Create a notebook
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create the top movers tab
        top_movers_tab = create_top_movers_tab(notebook)

        # Run the application
        root.mainloop()