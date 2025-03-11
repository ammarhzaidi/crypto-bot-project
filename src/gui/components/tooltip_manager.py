"""
Tooltip manager for centralized creation and control of tooltips in the application.
"""

import tkinter as tk
from typing import Dict, Any, Optional, Callable, Union, List

# Import the tooltip implementations
from src.gui.components.tooltip import EnhancedTooltip, TreeviewTooltip, TableCellTooltip


class TooltipManager:
    """
    Centralized manager for all tooltips in the application.
    Provides a single point of control for tooltip creation, updating, and removal.
    """

    def __init__(self):
        """Initialize the tooltip manager."""
        # Store all active tooltips
        self.tooltips: Dict[str, EnhancedTooltip] = {}

        # Default tooltip configuration
        self.default_config = {
            'delay': 500,
            'bg_color': '#ffffcc',  # Light yellow
            'fg_color': 'black',
            'font': ('TkDefaultFont', 8, 'normal'),
            'padding': 3,
            'position': 'auto',
            'follow_mouse': False,
            'wrap_length': 300
        }

    def generate_widget_id(self, widget) -> str:
        """
        Generate a unique identifier for a widget.

        Args:
            widget: The widget to generate an ID for

        Returns:
            A string identifier for the widget
        """
        # Use the widget's string representation as an ID
        return str(widget)

    def add_tooltip(self, widget, text=None, callback=None, **config) -> str:
        """
        Add a tooltip to a widget.

        Args:
            widget: The widget to attach a tooltip to
            text: Static tooltip text
            callback: Function that returns tooltip text
            **config: Override default tooltip configuration

        Returns:
            Widget ID that can be used to reference this tooltip
        """
        # Generate a unique ID for this widget
        widget_id = self.generate_widget_id(widget)

        # Remove any existing tooltip for this widget
        if widget_id in self.tooltips:
            self.remove_tooltip(widget_id)

        # Combine default config with provided config
        tooltip_config = dict(self.default_config)
        tooltip_config.update(config)

        # Determine the tooltip text (static or dynamic)
        tooltip_text = callback if callback else text

        # Create the tooltip
        self.tooltips[widget_id] = EnhancedTooltip(
            widget,
            text=tooltip_text,
            **tooltip_config
        )

        return widget_id

    def add_treeview_tooltip(self, treeview, callback=None, column_tooltips=None, **config) -> str:
        """
        Add tooltips to a treeview that work at the cell level.

        Args:
            treeview: The ttk.Treeview widget
            callback: Function to call with (item_id, column_id, value) that returns tooltip text
            column_tooltips: Dict mapping column IDs to static tooltips
            **config: Override default tooltip configuration

        Returns:
            Widget ID that can be used to reference this tooltip
        """
        # Generate a unique ID for this widget
        widget_id = self.generate_widget_id(treeview)

        # Remove any existing tooltip for this widget
        if widget_id in self.tooltips:
            self.remove_tooltip(widget_id)

        # Combine default config with provided config
        tooltip_config = dict(self.default_config)
        tooltip_config.update(config)

        # Create the specialized treeview tooltip
        self.tooltips[widget_id] = TreeviewTooltip(
            treeview,
            callback=callback,
            column_tooltips=column_tooltips,
            **tooltip_config
        )

        return widget_id

    def add_table_tooltip(self, widget, cell_identifier, tooltip_provider, **config) -> str:
        """
        Add tooltips to a table-like widget.

        Args:
            widget: The widget containing the table
            cell_identifier: Function that returns (row, column) given mouse coords
            tooltip_provider: Function that returns tooltip text given (row, column)
            **config: Override default tooltip configuration

        Returns:
            Widget ID that can be used to reference this tooltip
        """
        # Generate a unique ID for this widget
        widget_id = self.generate_widget_id(widget)

        # Remove any existing tooltip for this widget
        if widget_id in self.tooltips:
            self.remove_tooltip(widget_id)

        # Combine default config with provided config
        tooltip_config = dict(self.default_config)
        tooltip_config.update(config)

        # Create the specialized table cell tooltip
        self.tooltips[widget_id] = TableCellTooltip(
            widget,
            cell_identifier=cell_identifier,
            text=tooltip_provider,
            **tooltip_config
        )

        return widget_id

    def update_tooltip(self, widget_id, text=None, callback=None) -> bool:
        """
        Update an existing tooltip's text or callback.

        Args:
            widget_id: The ID of the widget with the tooltip to update
            text: New static tooltip text
            callback: New function that returns tooltip text

        Returns:
            True if the tooltip was updated, False if not found
        """
        if widget_id not in self.tooltips:
            return False

        # Update the tooltip text or callback
        if callback is not None:
            self.tooltips[widget_id].text = callback
        elif text is not None:
            self.tooltips[widget_id].text = text

        return True

    def update_tooltip_config(self, widget_id, **config) -> bool:
        """
        Update an existing tooltip's configuration.

        Args:
            widget_id: The ID of the widget with the tooltip to update
            **config: New configuration options

        Returns:
            True if the tooltip was updated, False if not found
        """
        if widget_id not in self.tooltips:
            return False

        tooltip = self.tooltips[widget_id]

        # Update each provided config option
        for key, value in config.items():
            if hasattr(tooltip, key):
                setattr(tooltip, key, value)

        return True

    def remove_tooltip(self, widget_id) -> bool:
        """
        Remove a tooltip from a widget.

        Args:
            widget_id: The ID of the widget with the tooltip to remove

        Returns:
            True if the tooltip was removed, False if not found
        """
        if widget_id not in self.tooltips:
            return False

        # Hide the tooltip if it's visible
        self.tooltips[widget_id].hidetip()

        # Clean up bindings
        try:
            widget = self.tooltips[widget_id].widget
            widget.unbind("<Enter>", self.tooltips[widget_id].bind_id)
            widget.unbind("<Leave>", self.tooltips[widget_id].bind_id)
            widget.unbind("<Motion>", self.tooltips[widget_id].bind_id)
        except AttributeError:
            # If bindings are unavailable, just continue
            pass

        # Remove from our tracking
        del self.tooltips[widget_id]

        return True

    def remove_all_tooltips(self):
        """Remove all tooltips being managed."""
        widget_ids = list(self.tooltips.keys())
        for widget_id in widget_ids:
            self.remove_tooltip(widget_id)

    def get_tooltip(self, widget_id) -> Optional[EnhancedTooltip]:
        """
        Get the tooltip object for a widget.

        Args:
            widget_id: The ID of the widget with the tooltip

        Returns:
            The tooltip object, or None if not found
        """
        return self.tooltips.get(widget_id)

    def update_all_tooltips(self, **config):
        """
        Update configuration for all tooltips.

        Args:
            **config: Configuration parameters to update
        """
        for widget_id in self.tooltips:
            self.update_tooltip_config(widget_id, **config)

    # Convenience methods for common tooltip requirements

    def add_timestamp_tooltip(self, widget, timestamp_provider,
                              format_str="%Y-%m-%d %H:%M:%S",
                              prefix="Recorded at: ",
                              timezone_suffix=" (Pakistan time)",
                              **config) -> str:
        """
        Add a tooltip that displays a formatted timestamp.

        Args:
            widget: The widget to attach the tooltip to
            timestamp_provider: Function that returns a timestamp
            format_str: Timestamp format string
            prefix: Text to show before the timestamp
            timezone_suffix: Text to show after the timestamp
            **config: Additional tooltip configuration

        Returns:
            Widget ID for the tooltip
        """

        def get_timestamp_tooltip():
            timestamp = timestamp_provider()
            if not timestamp:
                return None

            # Format the timestamp if it's not already a string
            if not isinstance(timestamp, str):
                try:
                    # For datetime objects
                    if hasattr(timestamp, 'strftime'):
                        formatted_time = timestamp.strftime(format_str)
                    else:
                        # For numeric timestamps
                        from datetime import datetime
                        formatted_time = datetime.fromtimestamp(timestamp).strftime(format_str)
                except Exception:
                    # If formatting fails, use the string representation
                    formatted_time = str(timestamp)
            else:
                formatted_time = timestamp

            return f"{prefix}{formatted_time}{timezone_suffix}"

        return self.add_tooltip(widget, callback=get_timestamp_tooltip, **config)

    def add_data_tooltip(self, widget, data_provider, formatter=None, **config) -> str:
        """
        Add a tooltip that displays data with optional formatting.

        Args:
            widget: The widget to attach the tooltip to
            data_provider: Function that returns data to display
            formatter: Function that formats the data (or None to use str())
            **config: Additional tooltip configuration

        Returns:
            Widget ID for the tooltip
        """

        def get_data_tooltip():
            data = data_provider()
            if data is None:
                return None

            if formatter:
                return formatter(data)
            return str(data)

        return self.add_tooltip(widget, callback=get_data_tooltip, **config)