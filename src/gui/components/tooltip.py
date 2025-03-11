"""
Enhanced tooltip implementation for the crypto trading bot GUI.
Provides flexible tooltips that work with various Tkinter widgets.
"""

import tkinter as tk
from typing import Optional, Union, Callable, Dict, Any


class EnhancedTooltip:
    """
    Flexible tooltip system for displaying information on hover.
    Works with various Tkinter widgets including Treeviews.
    """

    def __init__(self, widget, text='widget info', delay=500,
                 bg_color='#ffffcc', fg_color='black',
                 font=None, padding=3, position='auto',
                 follow_mouse=False, wrap_length=300):
        """
        Initialize tooltip with customizable properties.

        Args:
            widget: The widget to attach the tooltip to
            text: Static text or callable that returns text
            delay: Delay in ms before showing tooltip
            bg_color: Background color
            fg_color: Text color
            font: Font specification (e.g., ('Tahoma', 8, 'normal'))
            padding: Padding in pixels
            position: 'auto', 'north', 'south', 'east', 'west'
            follow_mouse: Whether tooltip should follow mouse cursor
            wrap_length: Text wrapping width in pixels
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.font = font or ('TkDefaultFont', 8, 'normal')
        self.padding = padding
        self.position = position
        self.follow_mouse = follow_mouse
        self.wrap_length = wrap_length

        # Internal state
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self._bound_widget = None
        self._last_position = None

        # Bind events
        self.bind_widget()

    def bind_widget(self):
        """Bind mouse events to the widget."""
        if self._bound_widget == self.widget:
            return  # Already bound

        self._bound_widget = self.widget
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<Motion>", self.motion)

        # For treeview, we need special handling
        if isinstance(self.widget, tk.ttk.Treeview):
            self.widget.bind("<<TreeviewSelect>>", self.leave)

        # For scrolled widgets
        if hasattr(self.widget, 'yview'):
            # If widget supports scrolling, hide tooltip on scroll
            if hasattr(self.widget, 'bind_class') and hasattr(self.widget, 'winfo_class'):
                self.widget.bind_class(self.widget.winfo_class(), "<MouseWheel>",
                                       lambda e: self.hidetip())
                self.widget.bind_class(self.widget.winfo_class(), "<Button-4>",
                                       lambda e: self.hidetip())
                self.widget.bind_class(self.widget.winfo_class(), "<Button-5>",
                                       lambda e: self.hidetip())

    def enter(self, event=None):
        """Handle mouse entering widget."""
        self.schedule(event)

    def leave(self, event=None):
        """Handle mouse leaving widget."""
        self.unschedule()
        self.hidetip()

    def motion(self, event=None):
        """Handle mouse motion over widget."""
        self.x, self.y = event.x_root, event.y_root

        if self.tipwindow and self.follow_mouse:
            # Update position if follow_mouse is enabled
            self._place_tooltip()

    def schedule(self, event=None):
        """Schedule tooltip to appear after delay."""
        self.unschedule()
        if event:
            self.x, self.y = event.x_root, event.y_root
        self.id = self.widget.after(self.delay, self.showtip)

    def unschedule(self):
        """Cancel scheduled tooltip display."""
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def get_text(self):
        """Get tooltip text, supporting both static text and callable."""
        if callable(self.text):
            return self.text()
        return self.text

    def showtip(self):
        """Show the tooltip."""
        text = self.get_text()
        if not text:
            return  # Don't show empty tooltips

        # Create tooltip window
        self.tipwindow = tw = tk.Toplevel(self.widget)

        # Remove window decorations
        tw.wm_overrideredirect(True)

        # Create tooltip content
        label = tk.Label(tw, text=text, justify=tk.LEFT,
                         background=self.bg_color, foreground=self.fg_color,
                         relief=tk.SOLID, borderwidth=1,
                         font=self.font, wraplength=self.wrap_length)
        label.pack(ipadx=self.padding, ipady=self.padding)

        # Position the tooltip
        self._place_tooltip()

        # Store this position to help with detecting repositioning needs
        self._last_position = (self.x, self.y)

    def _place_tooltip(self):
        """Place the tooltip based on position setting."""
        if not self.tipwindow:
            return

        # Get tooltip size
        tw_width = self.tipwindow.winfo_reqwidth()
        tw_height = self.tipwindow.winfo_reqheight()

        # Get screen size and position
        screen_width = self.widget.winfo_screenwidth()
        screen_height = self.widget.winfo_screenheight()

        # Default position (below and to the right of cursor)
        x, y = self.x + 15, self.y + 10

        # Adjust position based on setting and available space
        if self.position == 'auto' or self.position not in ('north', 'south', 'east', 'west'):
            # Avoid going off screen
            if x + tw_width > screen_width:
                x = self.x - tw_width - 5
            if y + tw_height > screen_height:
                y = self.y - tw_height - 5
        elif self.position == 'north':
            x = self.x - (tw_width // 2)
            y = self.y - tw_height - 5
        elif self.position == 'south':
            x = self.x - (tw_width // 2)
            y = self.y + 15
        elif self.position == 'east':
            x = self.x + 15
            y = self.y - (tw_height // 2)
        elif self.position == 'west':
            x = self.x - tw_width - 5
            y = self.y - (tw_height // 2)

        # Make sure tooltip is on screen
        x = max(0, min(x, screen_width - tw_width))
        y = max(0, min(y, screen_height - tw_height))

        # Set the position
        self.tipwindow.wm_geometry(f"+{x}+{y}")

    def hidetip(self):
        """Hide the tooltip."""
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

    def update_text(self, text):
        """Update the tooltip text."""
        self.text = text
        # If tooltip is currently shown, update it
        if self.tipwindow:
            self.hidetip()
            self.showtip()


class TreeviewTooltip(EnhancedTooltip):
    """
    Specialized tooltip for Treeview widgets that need cell-level tooltips.
    """

    def __init__(self, treeview, callback=None, column_tooltips=None, **kwargs):
        """
        Initialize tooltip for a Treeview widget.

        Args:
            treeview: The ttk.Treeview widget
            callback: Function to call to get tooltip text.
                     Will be passed (item_id, column_id, value)
            column_tooltips: Dict mapping column IDs to static tooltips
            **kwargs: Additional arguments for the EnhancedTooltip base class
        """
        self.callback = callback
        self.column_tooltips = column_tooltips or {}
        self.last_item = None
        self.last_column = None

        # Initialize with a placeholder text function
        super().__init__(treeview, text=self._get_cell_tooltip, **kwargs)

    def motion(self, event):
        """Handle mouse motion to identify the cell under cursor."""
        if not self.widget:
            return

        # Get the item (row) and column under cursor
        item = self.widget.identify_row(event.y)
        column = self.widget.identify_column(event.x)

        # If we're not over a cell, hide the tooltip
        if not item or not column:
            self.hidetip()
            self.last_item = None
            self.last_column = None
            return

        # If we've moved to a different cell, hide and reschedule
        if item != self.last_item or column != self.last_column:
            self.hidetip()
            self.last_item = item
            self.last_column = column
            self.schedule(event)

        # Always update position for mouse following
        super().motion(event)

    def _get_cell_tooltip(self):
        """Get tooltip text for the current cell."""
        if not self.last_item or not self.last_column:
            return None

        # Get the column name from column ID (e.g., '#1' -> 'name')
        column_name = self.last_column
        if self.last_column.startswith('#'):
            column_index = int(self.last_column[1:]) - 1
            if column_index >= 0 and column_index < len(self.widget['columns']):
                column_name = self.widget['columns'][column_index]

        # Get the value in the cell
        value = None
        values = self.widget.item(self.last_item, 'values')
        if values:
            try:
                column_index = list(self.widget['columns']).index(column_name)
                if 0 <= column_index < len(values):
                    value = values[column_index]
            except (ValueError, IndexError):
                pass

        # Try to get tooltip from callback first
        if self.callback:
            tooltip_text = self.callback(self.last_item, column_name, value)
            if tooltip_text:
                return tooltip_text

        # Then fall back to column-specific tooltips
        if column_name in self.column_tooltips:
            return self.column_tooltips[column_name]

        # Default to None (no tooltip)
        return None


class TableCellTooltip(EnhancedTooltip):
    """
    Tooltip implementation for table cells in various table-like widgets.
    Works with Treeview, Listbox, Text, Entry, etc.
    """

    def __init__(self, widget, cell_identifier, **kwargs):
        """
        Initialize tooltip for a table cell.

        Args:
            widget: The widget containing the table
            cell_identifier: Function that returns (row, column) given mouse coords
            **kwargs: Additional arguments for the EnhancedTooltip base class
        """
        self.cell_identifier = cell_identifier
        self.current_cell = None

        # Initialize base class
        super().__init__(widget, text=self._get_cell_text, **kwargs)

    def motion(self, event):
        """Handle mouse motion to identify cell under cursor."""
        if not self.widget:
            return

        # Get cell under cursor
        cell = self.cell_identifier(event.x, event.y)

        # If we're not over a cell, hide tooltip
        if not cell:
            self.hidetip()
            self.current_cell = None
            return

        # If we've moved to a different cell, hide and reschedule
        if cell != self.current_cell:
            self.hidetip()
            self.current_cell = cell
            self.schedule(event)

        # Update position for mouse following
        super().motion(event)

    def _get_cell_text(self):
        """Get tooltip text for current cell."""
        if not self.current_cell:
            return None

        # If text is a callable, call it with current cell
        if callable(self.text):
            return self.text(self.current_cell)

        # Otherwise just return the text
        return self.text