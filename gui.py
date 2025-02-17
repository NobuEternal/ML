import tkinter as tk
from tkinter import scrolledtext
import threading
import logging
from main import main

class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Ticketing System Pipeline")
        
        self.start_button = tk.Button(root, text="Start Pipeline", command=self.start_pipeline)
        self.start_button.pack(pady=10)
        
        self.log_text = scrolledtext.ScrolledText(root, width=100, height=30)
        self.log_text.pack(pady=10)
        
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(self)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
        
    def write(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        
    def flush(self):
        pass
        
    def start_pipeline(self):
        threading.Thread(target=main).start()

if __name__ == "__main__":
    root = tk.Tk()
    gui = PipelineGUI(root)
    root.mainloop()
