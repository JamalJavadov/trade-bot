from __future__ import annotations
import tkinter as tk
from .ui.gui_tk import App


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
