"""
Integration module for adding ML capabilities to the main application.
"""

from src.gui.ml_prediction_tab import create_ml_prediction_tab


def add_ml_tab_to_gui(notebook, client=None):
    """
    Add the ML prediction tab to the main application.

    Args:
        notebook: ttk.Notebook widget from the main application
        client: OKXClient instance (optional)

    Returns:
        MLPredictionTab instance
    """
    ml_tab = create_ml_prediction_tab(notebook, client)
    # The notebook.add() call should be here, not in create_ml_prediction_tab
    notebook.add(ml_tab.frame, text="ML Prediction")
    return ml_tab

