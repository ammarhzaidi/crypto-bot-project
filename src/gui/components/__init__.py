"""
Reusable UI components for the crypto trading bot GUI.
"""

# Export main classes for easy importing
from src.gui.components.tooltip import EnhancedTooltip, TreeviewTooltip, TableCellTooltip
from src.gui.components.tooltip_manager import TooltipManager

# Create a convenience function to get a tooltip manager
_default_tooltip_manager = None


def get_tooltip_manager():
    """
    Get the default tooltip manager instance.

    This provides a singleton-like access to a shared tooltip manager.
    """
    global _default_tooltip_manager
    if _default_tooltip_manager is None:
        _default_tooltip_manager = TooltipManager()
    return _default_tooltip_manager


# Make these directly importable from the package
__all__ = [
    'EnhancedTooltip',
    'TreeviewTooltip',
    'TableCellTooltip',
    'TooltipManager',
    'get_tooltip_manager'
]