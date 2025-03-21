import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime

class SchedulingTab:
    """Tab for controlling application-wide scheduling settings."""

    def __init__(self, parent, top_movers_tab):
        """Initialize the scheduling control tab."""
        self.parent = parent
        self.top_movers_tab = top_movers_tab
        self.scheduled_task = None

        # Create main frame
        self.frame = ttk.Frame(parent, padding="10")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create scheduling controls
        self.create_control_panel()

        # Add status label
        self.status_var = tk.StringVar(value="Scheduler: Inactive")
        self.status_label = ttk.Label(
            self.frame,
            textvariable=self.status_var,
            font=("TkDefaultFont", 9, "bold")
        )
        self.status_label.pack(pady=5)

        # Add log text area
        self.log_frame = ttk.LabelFrame(self.frame, text="Scheduler Log", padding="5")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def create_control_panel(self):
        """Create the scheduling control panel."""
        # Main scheduling controls
        control_frame = ttk.LabelFrame(self.frame, text="Scheduling Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Re-run frequency control
        ttk.Label(control_frame, text="Re-run every:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        frequency_frame = ttk.Frame(control_frame)
        frequency_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        self.frequency_var = tk.IntVar(value=60)
        frequency_spinbox = ttk.Spinbox(
            frequency_frame,
            from_=15,
            to=1440,
            increment=15,
            textvariable=self.frequency_var,
            width=10
        )
        frequency_spinbox.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(frequency_frame, text="minutes").pack(side=tk.LEFT)

        # Schedule button
        self.schedule_button = ttk.Button(
            control_frame,
            text="Start Scheduling",
            command=self.toggle_scheduling
        )
        self.schedule_button.grid(row=0, column=2, padx=10, pady=5)

        # Top Movers Parameters
        movers_frame = ttk.LabelFrame(self.frame, text="Top Movers Parameters", padding="10")
        movers_frame.pack(fill=tk.X, pady=5)

        # Top N setting
        ttk.Label(movers_frame, text="Top N:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.top_n_var = tk.IntVar(value=10)
        top_n_spinbox = ttk.Spinbox(
            movers_frame,
            from_=5,
            to=50,
            increment=5,
            textvariable=self.top_n_var,
            width=10
        )
        top_n_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Minimum Volume
        ttk.Label(movers_frame, text="Min Volume ($):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.min_volume_var = tk.DoubleVar(value=1000000)
        volume_spinbox = ttk.Spinbox(
            movers_frame,
            from_=100000,
            to=100000000,
            increment=100000,
            textvariable=self.min_volume_var,
            width=15
        )
        volume_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Compare Hours
        ttk.Label(movers_frame, text="Compare Hours:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.compare_hours_var = tk.DoubleVar(value=4.0)
        hours_spinbox = ttk.Spinbox(
            movers_frame,
            from_=1,
            to=24,
            increment=1,
            textvariable=self.compare_hours_var,
            width=10
        )
        hours_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

    def toggle_scheduling(self):
        if not hasattr(self, 'is_scheduled'):
            self.is_scheduled = False

        if self.is_scheduled:
            self.is_scheduled = False
            self.schedule_button.config(text="Start Scheduling")
            self.status_var.set("Scheduler: Inactive")
            self.log_message("Scheduler stopped")
            if self.scheduled_task:
                self.parent.after_cancel(self.scheduled_task)
                self.scheduled_task = None
        else:
            self.is_scheduled = True
            self.schedule_button.config(text="Stop Scheduling")
            self.status_var.set("Scheduler: Active")
            self.log_message("Scheduler started")
            self.schedule_next_run()

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    def schedule_next_run(self):
        """Schedule the next analysis run."""
        if not self.is_scheduled:
            return

        # Get frequency in minutes and convert to milliseconds
        frequency_ms = self.frequency_var.get() * 60 * 1000

        # Run top movers analysis with scheduled parameters
        self.run_scheduled_analysis()

        # Schedule next run
        self.scheduled_task = self.parent.after(frequency_ms, self.schedule_next_run)

    def run_scheduled_analysis(self):
        """Run analysis with scheduled parameters."""
        self.log_message("Running scheduled Top Movers analysis...")
        # Set parameters in top movers tab
        self.top_movers_tab.top_n_var.set(self.top_n_var.get())
        self.top_movers_tab.min_volume_var.set(self.min_volume_var.get())
        self.top_movers_tab.compare_hours_var.set(self.compare_hours_var.get())

        # Run the analysis
        self.top_movers_tab.run_analysis()
        self.log_message("Analysis completed")

