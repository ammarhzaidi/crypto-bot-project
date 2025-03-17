import tkinter as tk
from tkinter import ttk
import threading
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the ML tab module
from src.gui.ml_prediction_tab import MLPredictionTab
from src.market_data.okx_client import OKXClient


def main():
    # Create the main window
    root = tk.Tk()
    root.title("ML Prediction Test")
    root.geometry("1000x700")

    # Create a loading frame
    loading_frame = ttk.Frame(root)
    loading_frame.pack(fill=tk.BOTH, expand=True)

    loading_label = ttk.Label(loading_frame, text="Initializing application...", font=("TkDefaultFont", 14))
    loading_label.pack(expand=True)

    progress = ttk.Progressbar(loading_frame, mode="indeterminate")
    progress.pack(fill=tk.X, padx=20, pady=10)
    progress.start()

    # Function to initialize the client in a background thread
    def initialize_client():
        client = OKXClient()

        # Once client is initialized, create the ML tab on the main thread
        root.after(0, lambda: create_ml_tab(root, loading_frame, client))

    # Start initialization in a background thread
    threading.Thread(target=initialize_client, daemon=True).start()

    # Start the main loop
    root.mainloop()


def create_ml_tab(root, loading_frame, client):
    # Remove the loading frame
    loading_frame.destroy()

    # Create a frame for the ML tab
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create the ML tab with the initialized client
    ml_tab = MLPredictionTab(frame, client=client)


if __name__ == "__main__":
    main()
