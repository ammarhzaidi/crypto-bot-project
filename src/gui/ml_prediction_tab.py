"""
Machine Learning prediction tab for the cryptocurrency trading bot GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import threading
import time
from datetime import datetime, timedelta
import os
import sys

# Add project root to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import ML components
from src.ml.price_predictor import MLModelManager
from src.market_data.okx_client import OKXClient


class MLPredictionTab:
    """
    Tab for ML-based price prediction and analysis.
    """

    def __init__(self, parent, client=None):
        """
        Initialize the ML prediction tab.

        Args:
            parent: Parent frame
            client: OKXClient instance or None to create a new one
        """
        self.parent = parent
        self.client = client or OKXClient()

        # Initialize ML model manager
        self.model_manager = MLModelManager()

        # Track running state
        self.running = False
        self.training_thread = None

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
        """Create the control panel with buttons and options."""
        control_frame = ttk.LabelFrame(self.frame, text="ML Prediction Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Configure grid
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)

        # Symbol selection
        ttk.Label(control_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.symbol_var = tk.StringVar(value="BTC-USDT")
        self.symbol_entry = ttk.Combobox(control_frame, textvariable=self.symbol_var, width=15)
        self.symbol_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Populate symbols dropdown
        self.populate_symbols()

        # Timeframe selection
        ttk.Label(control_frame, text="Timeframe:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.timeframe_var = tk.StringVar(value="1h")
        self.timeframe_combo = ttk.Combobox(
            control_frame,
            textvariable=self.timeframe_var,
            values=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            width=10
        )
        self.timeframe_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Model type selection
        ttk.Label(control_frame, text="Model Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_type_var = tk.StringVar(value="random_forest")
        self.model_type_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_type_var,
            values=["random_forest", "gradient_boosting", "linear", "svr"],
            width=15
        )
        self.model_type_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Data points selection
        ttk.Label(control_frame, text="Data Points:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.data_points_var = tk.IntVar(value=500)
        data_points_spinbox = ttk.Spinbox(
            control_frame, from_=100, to=5000, increment=100,
            textvariable=self.data_points_var, width=10
        )
        data_points_spinbox.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Prediction periods selection
        ttk.Label(control_frame, text="Prediction Periods:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.prediction_periods_var = tk.IntVar(value=5)
        prediction_periods_spinbox = ttk.Spinbox(
            control_frame, from_=1, to=30, increment=1,
            textvariable=self.prediction_periods_var, width=10
        )
        prediction_periods_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Hyperparameter tuning checkbox
        self.hyperparameter_tuning_var = tk.BooleanVar(value=False)
        hyperparameter_tuning_check = ttk.Checkbutton(
            control_frame, text="Hyperparameter Tuning",
            variable=self.hyperparameter_tuning_var
        )
        hyperparameter_tuning_check.grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W + tk.E, pady=10)

        # Train button
        self.train_button = ttk.Button(
            buttons_frame, text="Train Model",
            command=self.train_model
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        # Predict button
        self.predict_button = ttk.Button(
            buttons_frame, text="Make Prediction",
            command=self.make_prediction
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        # Load model button
        self.load_button = ttk.Button(
            buttons_frame, text="Load Model",
            command=self.load_model
        )
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Evaluate model button
        self.evaluate_button = ttk.Button(
            buttons_frame, text="Evaluate Model",
            command=self.evaluate_model
        )
        self.evaluate_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(buttons_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=10)

    def create_results_panel(self):
        """Create the panel for displaying prediction results."""
        # Create notebook for results tabs
        self.results_notebook = ttk.Notebook(self.frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabs
        self.prediction_frame = ttk.Frame(self.results_notebook, padding="10")
        self.evaluation_frame = ttk.Frame(self.results_notebook, padding="10")
        self.feature_importance_frame = ttk.Frame(self.results_notebook, padding="10")
        self.log_frame = ttk.Frame(self.results_notebook, padding="10")

        self.results_notebook.add(self.prediction_frame, text="Predictions")
        self.results_notebook.add(self.evaluation_frame, text="Model Evaluation")
        self.results_notebook.add(self.feature_importance_frame, text="Feature Importance")
        self.results_notebook.add(self.log_frame, text="Log")

        # Set up prediction frame
        self.setup_prediction_frame()

        # Set up evaluation frame
        self.setup_evaluation_frame()

        # Set up feature importance frame
        self.setup_feature_importance_frame()

        # Create log area
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar to log
        log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

    def setup_prediction_frame(self):
        """Set up the prediction display frame."""

        # Create a frame for the plot
        self.prediction_plot_frame = ttk.Frame(self.prediction_frame, height=200)
        self.prediction_plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a frame for prediction details
        self.prediction_details_frame = ttk.LabelFrame(self.prediction_frame, text="Prediction Details")
        self.prediction_details_frame.pack(fill=tk.X, pady=5)

        # Create a treeview for prediction results
        columns = ("Period", "Date", "Predicted Price", "Lower Bound", "Upper Bound")
        self.prediction_treeview = ttk.Treeview(
            self.prediction_details_frame, columns=columns, show="headings",
            height=5
        )

        # Configure columns
        self.prediction_treeview.heading("Period", text="Period")
        self.prediction_treeview.heading("Date", text="Date")
        self.prediction_treeview.heading("Predicted Price", text="Predicted Price")
        self.prediction_treeview.heading("Lower Bound", text="Lower Bound")
        self.prediction_treeview.heading("Upper Bound", text="Upper Bound")

        self.prediction_treeview.column("Period", width=50, anchor=tk.CENTER)
        self.prediction_treeview.column("Date", width=150, anchor=tk.CENTER)
        self.prediction_treeview.column("Predicted Price", width=100, anchor=tk.E)
        self.prediction_treeview.column("Lower Bound", width=100, anchor=tk.E)
        self.prediction_treeview.column("Upper Bound", width=100, anchor=tk.E)

        # Add scrollbar
        prediction_scrollbar = ttk.Scrollbar(self.prediction_details_frame, orient="vertical",
                                             command=self.prediction_treeview.yview)
        self.prediction_treeview.configure(yscrollcommand=prediction_scrollbar.set)

        # Pack elements
        prediction_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.prediction_treeview.pack(fill=tk.BOTH, expand=True)

    def setup_evaluation_frame(self):
        """Set up the model evaluation frame."""
        # Create a frame for the plot
        self.evaluation_plot_frame = ttk.Frame(self.evaluation_frame,height=200)
        self.evaluation_plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a frame for metrics
        self.metrics_frame = ttk.LabelFrame(self.evaluation_frame, text="Model Metrics")
        self.metrics_frame.pack(fill=tk.X, pady=5)

        # Create labels for metrics
        metrics_grid = ttk.Frame(self.metrics_frame)
        metrics_grid.pack(fill=tk.X, padx=10, pady=10)

        # Configure grid
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
        metrics_grid.columnconfigure(2, weight=1)
        metrics_grid.columnconfigure(3, weight=1)

        # Create metric labels
        ttk.Label(metrics_grid, text="RMSE:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.rmse_var = tk.StringVar(value="-")
        ttk.Label(metrics_grid, textvariable=self.rmse_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(metrics_grid, text="MAE:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.mae_var = tk.StringVar(value="-")
        ttk.Label(metrics_grid, textvariable=self.mae_var).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        ttk.Label(metrics_grid, text="R²:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.r2_var = tk.StringVar(value="-")
        ttk.Label(metrics_grid, textvariable=self.r2_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(metrics_grid, text="MSE:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.mse_var = tk.StringVar(value="-")
        ttk.Label(metrics_grid, textvariable=self.mse_var).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

    def setup_feature_importance_frame(self):
        """Set up the feature importance frame."""
        # Create a frame for the plot
        self.feature_plot_frame = ttk.Frame(self.feature_importance_frame)
        self.feature_plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a treeview for feature importance
        columns = ("Feature", "Importance")
        self.feature_treeview = ttk.Treeview(
            self.feature_importance_frame, columns=columns, show="headings",
            height=10
        )

        # Configure columns
        self.feature_treeview.heading("Feature", text="Feature")
        self.feature_treeview.heading("Importance", text="Importance")

        self.feature_treeview.column("Feature", width=300, anchor=tk.W)
        self.feature_treeview.column("Importance", width=100, anchor=tk.E)

        # Add scrollbar
        feature_scrollbar = ttk.Scrollbar(self.feature_importance_frame, orient="vertical",
                                          command=self.feature_treeview.yview)
        self.feature_treeview.configure(yscrollcommand=feature_scrollbar.set)

        # Pack elements
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_treeview.pack(fill=tk.X, expand=False)

    def populate_symbols(self):
        """Populate the symbols dropdown with available symbols."""
        try:
            # Get top volume symbols
            symbols = self.client.get_top_volume_symbols(limit=50)

            if symbols:
                self.symbol_entry['values'] = symbols
            else:
                self.log("No symbols found. Using default.")
                self.symbol_entry['values'] = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
        except Exception as e:
            self.log(f"Error fetching symbols: {str(e)}")
            self.symbol_entry['values'] = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

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

    def fetch_data(self):
        """Fetch historical price data for the selected symbol."""
        if not self.client:
            self.log("Client not initialized. Please initialize the client first.")
            return

        symbol = self.symbol_var.get()
        timeframe = self.timeframe_var.get()
        data_points = self.data_points_var.get()

        self.log(f"Fetching data for {symbol} ({timeframe}, {data_points} points)...")
        self.update_status("Fetching data...")
        self.fetch_button.config(state=tk.DISABLED)

        # Run data fetching in a background thread
        def fetch_thread():
            try:
                # Fetch klines with error handling and logging
                klines = self.client.get_klines(symbol, interval=timeframe, limit=data_points)

                # Log detailed information about fetched data
                self.log(f"Fetched {len(klines)} data points")
                if klines:
                    # Print first few rows to verify data
                    for i, kline in enumerate(klines[:5], 1):
                        self.log(f"Kline {i}: {kline}")

                self.root.after(0, lambda: self.data_fetched(klines))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Error fetching data: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Data fetch failed"))
                self.root.after(0, lambda: self.fetch_button.config(state=tk.NORMAL))

        threading.Thread(target=fetch_thread, daemon=True).start()

    def data_fetched(self, klines):
        """Called when data is fetched."""
        if klines:
            # Convert to DataFrame with explicit column names
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Log DataFrame details
            self.log(f"DataFrame shape: {df.shape}")
            self.log(f"DataFrame columns: {df.columns}")
            self.log(f"DataFrame info:\n{df.info()}")

            # Check if we have enough data
            if len(df) < 100:
                self.log("Insufficient data for training (need at least 100 points)")
                self.update_status("Not enough data points")
                return

            # Store the data
            self.klines_data = klines
            self.df = df

            # Enable buttons
            self.predict_button.config(state=tk.NORMAL)
            self.backtest_button.config(state=tk.NORMAL)

            # Display chart
            self.display_data_chart()

            self.update_status("Data fetched successfully")
        else:
            self.log("No data available")
            self.update_status("No data available")

        self.fetch_button.config(state=tk.NORMAL)

    def make_prediction(self):
        """Make a prediction using the fetched data."""
        if not hasattr(self, 'df') or self.df is None or len(self.df) < 100:
            self.log("Insufficient data for prediction")
            messagebox.showerror("Error", "Please fetch more data before making a prediction")
            return

        self.log("Starting prediction process...")
        self.update_status("Making prediction...")
        self.predict_button.config(state=tk.DISABLED)

        # Run prediction in a background thread
        def predict_thread():
            try:
                # Import ML components here to avoid early loading
                from src.ml.price_predictor import PricePredictor, FeatureGenerator

                self.log("Generating features...")
                feature_generator = FeatureGenerator()

                # Log DataFrame details before feature generation
                self.log(f"DataFrame for feature generation shape: {self.df.shape}")

                features_df = feature_generator.generate_features(self.df)

                self.log(f"Features DataFrame shape: {features_df.shape}")

                # Check features DataFrame
                if len(features_df) < 100:
                    raise ValueError("Not enough features generated")

                self.log("Running prediction model...")
                predictor = PricePredictor()

                # Get prediction periods from UI
                periods = self.prediction_periods_var.get()

                # Make prediction
                predictions = predictor.predict_next_n_periods(features_df, n=periods)

                # Store predictions
                self.predictions = predictions

                self.root.after(0, self.prediction_complete)

            except Exception as e:
                self.root.after(0, lambda: self.log(f"Prediction error: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
                self.root.after(0, lambda: self.update_status("Prediction failed"))
                self.root.after(0, lambda: self.predict_button.config(state=tk.NORMAL))

        threading.Thread(target=predict_thread, daemon=True).start()

    def train_model(self):
        """Train a machine learning model for the selected symbol and timeframe."""
        if self.running:
            messagebox.showinfo("Running", "A task is already running. Please wait.")
            return

        # Set running state
        self.running = True
        self.update_status("Training model...")
        self.train_button.config(state=tk.DISABLED)

        # Start training thread
        self.training_thread = threading.Thread(target=self.training_task)
        self.training_thread.daemon = True
        self.training_thread.start()

    # Modify the training_task method in the MLPredictionTab class
    def training_task(self):
        try:
            self.log("Starting training process...")

            # Get parameters from UI
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            data_points = self.data_points_var.get()
            model_type = self.model_type_var.get()
            hyperparameter_tuning = self.hyperparameter_tuning_var.get()

            # Fetch data
            self.log(f"Fetching data for {symbol} ({timeframe}, {data_points} points)...")

            # Use client to fetch data
            klines = self.client.get_klines(symbol, interval=timeframe, limit=data_points)

            if not klines or len(klines) < 100:
                self.parent.after(0, lambda: self.log("Insufficient data for training (need at least 100 points)"))
                self.parent.after(0, lambda: self.update_status("Training failed: Not enough data"))
                return

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Log progress
            self.parent.after(0, lambda: self.log(f"Data prepared, starting model training..."))
            self.parent.after(0, lambda: self.update_status("Training model..."))

            # Train the model
            self.parent.after(0, lambda: self.log(f"Training {model_type} model for {symbol} ({timeframe})..."))

            # Get predictor from model manager
            predictor = self.model_manager.get_predictor(symbol, timeframe)

            # Train model with hyperparameter tuning if selected
            # Pass symbol and timeframe as required by the method
            metrics = predictor.train_model(
                df,
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                hyperparameter_tuning=hyperparameter_tuning
            )

            # Update UI with results
            self.parent.after(0, lambda: self.log(f"Model training completed with metrics:"))
            self.parent.after(0, lambda: self.log(f"  RMSE: {metrics['rmse']:.4f}"))
            self.parent.after(0, lambda: self.log(f"  MAE: {metrics['mae']:.4f}"))
            self.parent.after(0, lambda: self.log(f"  R²: {metrics['r2']:.4f}"))

            # Update metrics display
            self.parent.after(0, lambda m=metrics: self.update_metrics(m))

            # Update feature importance if available
            if model_type in predictor.models and hasattr(predictor.models[model_type]['model'],
                                                          'feature_importances_'):
                feature_importance = predictor._get_feature_importance(
                    predictor.models[model_type]['model'],
                    predictor.models[model_type]['feature_names']
                )
                self.parent.after(0, lambda f=feature_importance: self.update_feature_importance(f))

            # Update status
            self.parent.after(0, lambda: self.update_status("Training completed"))

        except Exception as e:
            import traceback
            self.parent.after(0, lambda: self.log(f"Error during training: {str(e)}"))
            self.parent.after(0, lambda: self.log(f"Traceback: {traceback.format_exc()}"))
            self.parent.after(0, lambda: self.update_status("Training failed"))

        finally:
            # Reset running state
            self.running = False
            self.parent.after(0, lambda: self.train_button.config(state=tk.NORMAL))

    def make_prediction(self):
        """Make a prediction using a trained model."""
        if self.running:
            messagebox.showinfo("Running", "A task is already running. Please wait.")
            return

        # Set running state
        self.running = True
        self.update_status("Making prediction...")
        self.predict_button.config(state=tk.DISABLED)

        # Start prediction thread
        prediction_thread = threading.Thread(target=self.prediction_task)
        prediction_thread.daemon = True
        prediction_thread.start()

    def prediction_task(self):
        """Run the prediction in a background thread."""
        try:
            # Get parameters
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            model_type = self.model_type_var.get()

            # Fetch data
            self.log(f"Fetching data for {symbol} ({timeframe})...")

            # Use client to fetch data
            data_points = self.data_points_var.get()
            klines = self.client.get_klines(symbol, interval=timeframe, limit=data_points)

            if not klines or len(klines) < 100:
                self.parent.after(0, lambda: self.log("Insufficient data for prediction (need at least 100 points)"))
                self.parent.after(0, lambda: self.update_status("Prediction failed: Not enough data"))
                return

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Make prediction with the data
            self.make_prediction_with_model(df, symbol, timeframe, model_type)

        except Exception as e:
            import traceback
            self.log(f"Error during prediction: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            self.update_status("Prediction failed")

        finally:
            # Reset running state
            self.running = False
            self.predict_button.config(state=tk.NORMAL)

    def make_prediction_with_model(self, df, symbol, timeframe, model_type):
        """Make a prediction with a specific model."""
        try:
            # Get prediction periods
            prediction_periods = self.prediction_periods_var.get()

            self.log(f"Making prediction for {symbol} ({timeframe}) using {model_type} model...")

            # Get predictions
            predictions = self.model_manager.predict_for_symbol(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                prediction_periods=prediction_periods
            )

            if predictions is None or len(predictions) == 0:
                self.log("No predictions generated")
                return

            # Update prediction display
            self.update_prediction_display(df, predictions, symbol, timeframe, model_type)

            # Log success
            self.log(f"Generated {len(predictions)} predictions")
            self.update_status("Prediction completed")

        except Exception as e:
            self.log(f"Error making prediction: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

    def update_prediction_display(self, df, predictions, symbol, timeframe, model_type):
        """Update the prediction display with new predictions."""
        # Clear existing treeview items
        for item in self.prediction_treeview.get_children():
            self.prediction_treeview.delete(item)

        # Add new predictions to treeview
        for i, (_, row) in enumerate(predictions.iterrows()):
            # Format values
            period = f"P{i + 1}"
            date = row['timestamp'].strftime("%Y-%m-%d %H:%M") if isinstance(row['timestamp'], datetime) else str(
                row['timestamp'])
            predicted_price = f"${row['predicted_close']:.4f}"
            lower_bound = f"${row['lower_bound']:.4f}"
            upper_bound = f"${row['upper_bound']:.4f}"

            # Insert into treeview
            self.prediction_treeview.insert("", tk.END,
                                            values=(period, date, predicted_price, lower_bound, upper_bound))

        # Get predictor
        predictor = self.model_manager.get_predictor(symbol, timeframe)

        # Create prediction plot in a thread-safe way
        def create_plot():
            # Create and display prediction plot
            fig = predictor.plot_predictions(df, predictions,
                                             title=f"{symbol} Price Prediction ({timeframe}, {model_type})")
            return fig

        # Execute plot creation and update UI on the main thread
        fig = create_plot()

        # Clear existing plot
        for widget in self.prediction_plot_frame.winfo_children():
            widget.destroy()

        # Create canvas for the plot with proper event handling
        canvas = FigureCanvasTkAgg(fig, master=self.prediction_plot_frame)
        canvas.draw()

        # Disable the default matplotlib key bindings
        canvas.mpl_connect('key_press_event', lambda event: None)

        # Add the canvas to the frame
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add a simple toolbar (optional)
        toolbar = NavigationToolbar2Tk(canvas, self.prediction_plot_frame)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

    def load_model(self):
        """Load a previously trained model."""
        # Get list of available models
        available_models = self.model_manager.list_available_models()

        if not available_models:
            messagebox.showinfo("No Models", "No trained models found.")
            return

        # Create model selection dialog
        dialog = tk.Toplevel(self.parent)
        dialog.title("Select Model")
        dialog.geometry("500x300")
        dialog.transient(self.parent)
        dialog.grab_set()

        # Create treeview for model selection
        columns = ("Symbol", "Timeframe", "Model Type")
        model_treeview = ttk.Treeview(dialog, columns=columns, show="headings", height=10)

        # Configure columns
        model_treeview.heading("Symbol", text="Symbol")
        model_treeview.heading("Timeframe", text="Timeframe")
        model_treeview.heading("Model Type", text="Model Type")

        model_treeview.column("Symbol", width=150, anchor=tk.W)
        model_treeview.column("Timeframe", width=100, anchor=tk.W)
        model_treeview.column("Model Type", width=150, anchor=tk.W)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=model_treeview.yview)
        model_treeview.configure(yscrollcommand=scrollbar.set)

        # Pack elements
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        model_treeview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Populate treeview
        for model in available_models:
            model_treeview.insert("", tk.END, values=(model['symbol'], model['timeframe'], model['model_type']))

        # Function to handle model selection
        def on_select():
            selected_items = model_treeview.selection()
            if not selected_items:
                messagebox.showinfo("No Selection", "Please select a model.")
                return

            # Get selected model
            item = selected_items[0]
            values = model_treeview.item(item, "values")

            symbol = values[0]
            timeframe = values[1]
            model_type = values[2]

            # Update UI
            self.symbol_var.set(symbol)
            self.timeframe_var.set(timeframe)
            self.model_type_var.set(model_type)

            # Load the model
            predictor = self.model_manager.get_predictor(symbol, timeframe)
            success = predictor.load_model(symbol, timeframe, model_type)

            if success:
                self.log(f"Loaded model: {symbol} ({timeframe}, {model_type})")

                # Update metrics if available
                if model_type in predictor.models and 'metrics' in predictor.models[model_type]:
                    self.update_metrics(predictor.models[model_type]['metrics'])

                # Update feature importance if available
                if hasattr(predictor.models[model_type]['model'], 'feature_importances_'):
                    feature_importance = predictor._get_feature_importance(
                        predictor.models[model_type]['model'],
                        predictor.models[model_type]['feature_names']
                    )
                    self.update_feature_importance(feature_importance)

                # Make a prediction with the loaded model
                self.make_prediction()
            else:
                self.log(f"Failed to load model: {symbol} ({timeframe}, {model_type})")

            # Close dialog
            dialog.destroy()

        # Add buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        select_button = ttk.Button(button_frame, text="Select", command=on_select)
        select_button.pack(side=tk.RIGHT, padx=5)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)

    def evaluate_model(self):
        """Evaluate the currently loaded model."""
        if self.running:
            messagebox.showinfo("Running", "A task is already running. Please wait.")
            return

        # Set running state
        self.running = True
        self.update_status("Evaluating model...")
        self.evaluate_button.config(state=tk.DISABLED)

        # Start evaluation thread
        evaluation_thread = threading.Thread(target=self.evaluation_task)
        evaluation_thread.daemon = True
        evaluation_thread.start()

    def evaluation_task(self):
        """Run the model evaluation in a background thread."""
        try:
            # Get parameters
            symbol = self.symbol_var.get()
            timeframe = self.timeframe_var.get()
            model_type = self.model_type_var.get()

            # Fetch data directly instead of using fetch_data()
            self.log(f"Fetching data for {symbol} ({timeframe})...")

            # Use client to fetch data
            data_points = self.data_points_var.get()
            klines = self.client.get_klines(symbol, interval=timeframe, limit=data_points)

            if not klines or len(klines) < 100:
                self.log("Insufficient data for evaluation (need at least 100 points)")
                self.update_status("Evaluation failed: Not enough data")
                return

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Get predictor
            predictor = self.model_manager.get_predictor(symbol, timeframe)

            # Check if model is loaded
            if model_type not in predictor.models:
                success = predictor.load_model(symbol, timeframe, model_type)
                if not success:
                    self.log(f"Model not found: {symbol} ({timeframe}, {model_type})")
                    self.update_status("Evaluation failed: Model not found")
                    return

            # Evaluate model
            self.log(f"Evaluating {model_type} model for {symbol} ({timeframe})...")
            metrics, fig = predictor.evaluate_model(df, model_type=model_type)

            # Update metrics display
            self.update_metrics(metrics)

            # Display evaluation plot
            if fig:
                # Clear existing plot
                for widget in self.evaluation_plot_frame.winfo_children():
                    widget.destroy()

                # Create canvas for the plot with proper event handling
                canvas = FigureCanvasTkAgg(fig, master=self.evaluation_plot_frame)
                canvas.draw()

                # Disable the default matplotlib key bindings
                canvas.mpl_connect('key_press_event', lambda event: None)

                # Add the canvas to the frame
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill=tk.BOTH, expand=True)

                # Add a simple toolbar (optional)
                toolbar = NavigationToolbar2Tk(canvas, self.evaluation_plot_frame)
                toolbar.update()
                canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

            # Log results
            self.log(f"Evaluation completed with metrics:")
            self.log(f"  RMSE: {metrics['rmse']:.4f}")
            self.log(f"  MAE: {metrics['mae']:.4f}")
            self.log(f"  R²: {metrics['r2']:.4f}")

            # Update status
            self.update_status("Evaluation completed")

        except Exception as e:
            import traceback
            self.log(f"Error during evaluation: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            self.update_status("Evaluation failed")

        finally:
            # Reset running state
            self.running = False
            self.evaluate_button.config(state=tk.NORMAL)

    def update_metrics(self, metrics):
        """Update the metrics display."""
        self.rmse_var.set(f"{metrics['rmse']:.4f}")
        self.mae_var.set(f"{metrics['mae']:.4f}")
        self.r2_var.set(f"{metrics['r2']:.4f}")
        self.mse_var.set(f"{metrics['mse']:.4f}")

    def update_feature_importance(self, feature_importance):
        """Update the feature importance display."""
        if not feature_importance:
            return

        # Clear existing items
        for item in self.feature_treeview.get_children():
            self.feature_treeview.delete(item)

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Add to treeview
        for feature, importance in sorted_features:
            self.feature_treeview.insert("", tk.END, values=(feature, f"{importance:.4f}"))

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get top 15 features for readability
        top_features = sorted_features[:15]

        # Extract feature names and importance values
        features = [item[0] for item in top_features]
        importances = [item[1] for item in top_features]

        # Create horizontal bar chart
        bars = ax.barh(features, importances, color='skyblue')

        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')

        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                    ha='left', va='center')

        # Tight layout
        plt.tight_layout()

        # Clear existing plot
        for widget in self.feature_plot_frame.winfo_children():
            widget.destroy()

        # Create canvas for the plot
        canvas = FigureCanvasTkAgg(fig, master=self.feature_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_ml_prediction_tab(notebook, client=None):
        """
        Create and add an ML prediction tab to a notebook.

        Args:
            notebook: ttk.Notebook widget
            client: OKXClient instance (optional)

        Returns:
            MLPredictionTab instance
        """
        ml_frame = ttk.Frame(notebook)
        ml_tab = MLPredictionTab(ml_frame, client=client)
        notebook.add(ml_frame, text="ML Prediction")
        return ml_tab

    # Testing code
    if __name__ == "__main__":
        # Create a standalone application for testing
        root = tk.Tk()
        root.title("ML Price Prediction")
        root.geometry("1000x700")

        # Create a notebook
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create the ML prediction tab
        ml_tab = create_ml_prediction_tab(notebook)

        # Run the application
        root.mainloop()



