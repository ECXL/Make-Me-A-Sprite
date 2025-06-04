import tkinter as tk
from frontend import PoseDetectorApp

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseDetectorApp(root)
    root.mainloop()