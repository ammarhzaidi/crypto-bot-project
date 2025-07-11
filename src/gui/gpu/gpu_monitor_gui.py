import tkinter as tk
from tkinter import ttk, messagebox
import psutil
import threading
import time
import tensorflow as tf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GPUMonitorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPU Monitor for Crypto AI")
        self.root.geometry("800x600")

        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            messagebox.showerror("Error", "No GPU detected! Please check your NVIDIA drivers.")
            self.root.destroy()
            return

        self.gpu = gpus[0]
        tf.config.experimental.set_memory_growth(self.gpu, True)

        self.setup_ui()
        self.setup_graphs()
        threading.Thread(target=self.update_stats, daemon=True).start()

    def setup_ui(self):
        # Stats frame
        self.stats_frame = ttk.LabelFrame(self.root, text="GPU Statistics", padding=10)
        self.stats_frame.pack(fill="x", padx=5, pady=5)

        self.gpu_util = ttk.Label(self.stats_frame, text="GPU Utilization: 0%")
        self.gpu_util.pack()

        self.mem_util = ttk.Label(self.stats_frame, text="Memory Usage: 0/0 MB")
        self.mem_util.pack()

        self.temp = ttk.Label(self.stats_frame, text="Temperature: 0°C")
        self.temp.pack()

    def setup_graphs(self):
        self.fig = Figure(figsize=(8, 4))
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.utilization_data = []
        self.temperature_data = []
        self.time_data = []

    def update_stats(self):
        while True:
            # Get GPU memory info using TensorFlow
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            memory_used = gpu_memory['current'] / 1024 ** 2  # Convert to MB
            memory_total = gpu_memory['peak'] / 1024 ** 2  # Convert to MB

            # Get GPU utilization through TensorFlow operations
            with tf.device('/GPU:0'):
                # Create small tensor operation to measure utilization
                tf.random.normal([1000, 1000])

            self.gpu_util.config(text=f"GPU Memory Used: {memory_used:.0f} MB")
            self.mem_util.config(text=f"Memory Peak: {memory_total:.0f} MB")

            # Update graphs
            self.utilization_data.append(memory_used)
            self.temperature_data.append(memory_total)
            self.time_data.append(time.time())

            if len(self.time_data) > 100:
                self.utilization_data.pop(0)
                self.temperature_data.pop(0)
                self.time_data.pop(0)

            self.ax1.clear()
            self.ax1.plot(self.time_data, self.utilization_data, label='Memory Used (MB)')
            self.ax1.plot(self.time_data, self.temperature_data, label='Memory Peak (MB)')
            self.ax1.legend()
            self.canvas.draw()

            time.sleep(1)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    monitor = GPUMonitorUI()
    monitor.run()
