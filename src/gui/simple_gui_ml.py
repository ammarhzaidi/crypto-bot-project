import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SimplifiedMLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Prediction Test")
        self.root.geometry("800x600")  # Set a minimum size

        # Create main frame with visible border for debugging
        self.frame = ttk.Frame(self.root, padding="10", relief="solid", borderwidth=1)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Add a simple label to verify something is visible
        self.test_label = ttk.Label(self.frame, text="Testing - If you can see this, the frame is working")
        self.test_label.pack(pady=20)

        # Try to create the actual widgets
        try:
            self.create_widgets()
        except Exception as e:
            print(f"Error creating widgets: {e}")
            import traceback
            traceback.print_exc()

            # Add an error message to the UI
            error_label = ttk.Label(
                self.frame,
                text=f"Error creating widgets: {str(e)}",
                foreground="red"
            )
            error_label.pack(pady=20)

    def create_widgets(self):
        # Create a simple control panel
        control_frame = ttk.LabelFrame(self.frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        # Add a button
        test_button = ttk.Button(control_frame, text="Test Button", command=self.test_function)
        test_button.pack(pady=10)

        # Create a log area
        log_frame = ttk.LabelFrame(self.frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Application started\n")

    def test_function(self):
        self.log_text.insert(tk.END, "Test button clicked\n")
        self.log_text.see(tk.END)


def main():
    root = tk.Tk()
    app = SimplifiedMLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
