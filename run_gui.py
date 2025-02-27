#!/usr/bin/env python
"""
Launcher for the Crypto Trading Bot GUI.
"""
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the GUI module
from src.gui.trading_gui import main

if __name__ == "__main__":
    # Run the GUI
    main()