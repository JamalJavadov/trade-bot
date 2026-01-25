from __future__ import annotations

import threading
import time
import queue
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from ..core import binance_data
from ..core import analyzer
from ..core.news import NewsAnalyzer
from ..core.planner import (
    load_settings,
    save_settings,
    run_scan_and_build_best_plan,
    format_report,
)
from ..core.position_monitor import PositionMonitor
from .deep_dashboard import DeepAnalysisDashboard, DeepAnalysisWindow


class ModernStyle:
    """Modern rəng və stil konfiqurasiyası"""
    
    # Rəng Paleti
    BG_DARK = "#121826"
    BG_MEDIUM = "#1f2736"
    BG_LIGHT = "#2d3850"
    BG_ELEVATED = "#253049"
    BG_PANEL = "#1a2234"
    ACCENT_PRIMARY = "#00d4ff"
    ACCENT_SUCCESS = "#00ff88"
    ACCENT_WARNING = "#ffaa00"
    ACCENT_ERROR = "#ff4444"
    ACCENT_INFO = "#6aa7ff"
    ACCENT_PURPLE = "#b28dff"
    TEXT_PRIMARY = "#f5f7ff"
    TEXT_SECONDARY = "#c5cbe6"
    TEXT_MUTED = "#8892b0"
    BORDER = "#3a475f"
    
    # Font Konfiqurasiyası
    FONT_MAIN = ("Segoe UI", 11)
    FONT_HEADER = ("Segoe UI Semibold", 12)
    FONT_TITLE = ("Segoe UI Bold", 14)
    FONT_HERO = ("Segoe UI Bold", 18)
    FONT_HERO_SUB = ("Segoe UI Semibold", 11)
    FONT_SUBTITLE = ("Segoe UI Semibold", 11)
    FONT_MONO = ("Consolas", 10)


class AnimatedProgressBar(ttk.Frame):
    """Animated və gradient progress bar"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.canvas = tk.Canvas(
            self,
            height=8,
            bg=ModernStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(fill="x", expand=True)
        
        self.max_value = 100
        self.current_value = 0
        self.bar_id = None
        self.glow_id = None
        
    def configure(self, maximum=None, value=None):
        if maximum is not None:
            self.max_value = maximum
        if value is not None:
            self.current_value = value
            self._update_bar()
    
    def _update_bar(self):
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1:
            return
        
        # Background
        self.canvas.create_rectangle(
            0, 0, width, height,
            fill=ModernStyle.BG_LIGHT,
            outline=""
        )
        
        # Progress bar
        if self.max_value > 0:
            progress_width = int((self.current_value / self.max_value) * width)
            
            # Gradient effect (simulyasiya)
            self.canvas.create_rectangle(
                0, 0, progress_width, height,
                fill=ModernStyle.ACCENT_PRIMARY,
                outline=""
            )
            
            # Glow effect
            if progress_width > 0:
                self.canvas.create_rectangle(
                    max(0, progress_width - 20), 0,
                    progress_width, height,
                    fill=ModernStyle.ACCENT_SUCCESS,
                    outline="",
                    stipple="gray50"
                )


class ConfidenceMeter(ttk.Frame):
    """Vizual ehtimal göstəricisi."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(
            self,
            height=28,
            bg=ModernStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="x", expand=True)
        self.value = 0.0
        self.target = 0.0
        self._animating = False

    def set_value(self, value: float) -> None:
        self.target = max(0.0, min(100.0, float(value)))
        if not self._animating:
            self._animating = True
            self._animate()

    def _animate(self) -> None:
        if abs(self.value - self.target) < 0.2:
            self.value = self.target
            self._animating = False
            self._draw()
            return
        step = 1.5 if self.target > self.value else -1.5
        self.value = max(0.0, min(100.0, self.value + step))
        self._draw()
        self.after(12, self._animate)

    def _draw(self) -> None:
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        self.canvas.create_rectangle(
            0, 0, width, height,
            fill=ModernStyle.BG_LIGHT,
            outline="",
        )
        fill_width = int((self.value / 100.0) * width)
        self.canvas.create_rectangle(
            0, 0, fill_width, height,
            fill=ModernStyle.ACCENT_SUCCESS if self.value >= 65 else ModernStyle.ACCENT_WARNING,
            outline="",
        )
        self.canvas.create_text(
            width / 2,
            height / 2,
            text=f"{self.value:.1f}%",
            fill=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_SUBTITLE,
        )


class StatusIndicator(tk.Canvas):
    """Animated status göstəricisi"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            width=12,
            height=12,
            bg=ModernStyle.BG_DARK,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.state = "idle"  # idle, active, success, error
        self.pulse_active = False
        self._draw()
    
    def set_state(self, state: str):
        self.state = state
        self._draw()
        if state == "active":
            self._start_pulse()
        else:
            self.pulse_active = False
    
    def _draw(self):
        self.delete("all")
        
        colors = {
            "idle": ModernStyle.TEXT_SECONDARY,
            "active": ModernStyle.ACCENT_PRIMARY,
            "success": ModernStyle.ACCENT_SUCCESS,
            "error": ModernStyle.ACCENT_ERROR
        }
        
        color = colors.get(self.state, ModernStyle.TEXT_SECONDARY)
        self.create_oval(2, 2, 10, 10, fill=color, outline="")
    
    def _start_pulse(self):
        if not self.pulse_active:
            self.pulse_active = True
            self._pulse()
    
    def _pulse(self):
        if not self.pulse_active or self.state != "active":
            return
        
        # Simple pulse effect
        self.delete("all")
        self.create_oval(1, 1, 11, 11, fill="", outline=ModernStyle.ACCENT_PRIMARY, width=2)
        self.create_oval(3, 3, 9, 9, fill=ModernStyle.ACCENT_PRIMARY, outline="")
        
        self.after(500, lambda: self._draw() if self.pulse_active else None)
        self.after(1000, self._pulse)


class SummaryChart(tk.Canvas):
    """Summary bar chart for scan results."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=120,
            bg=ModernStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.counts = {"ok": 0, "setup": 0, "no": 0}
        self.target_counts = dict(self.counts)
        self._animating = False
        self.bind("<Configure>", lambda _: self._draw())

    def set_counts(self, ok: int, setup: int, no: int) -> None:
        self.target_counts = {"ok": ok, "setup": setup, "no": no}
        if not self._animating:
            self._animating = True
            self._animate()

    def _animate(self):
        done = True
        for key in self.counts:
            current = self.counts[key]
            target = self.target_counts[key]
            if current != target:
                done = False
                step = 1 if target > current else -1
                self.counts[key] = current + step
        self._draw()
        if not done:
            self.after(15, self._animate)
        else:
            self._animating = False

    def _draw(self):
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return

        total = max(sum(self.counts.values()), 1)
        chart_top = 28
        chart_bottom = height - 18
        chart_height = max(1, chart_bottom - chart_top)
        bar_width = max(30, int(width / 6))
        gap = int((width - (bar_width * 3)) / 4)
        colors = {
            "ok": ModernStyle.ACCENT_SUCCESS,
            "setup": ModernStyle.ACCENT_WARNING,
            "no": ModernStyle.ACCENT_ERROR,
        }

        self.create_text(
            12,
            12,
            text="Signal Distribution",
            fill=ModernStyle.TEXT_MUTED,
            anchor="w",
            font=ModernStyle.FONT_SUBTITLE,
        )

        for idx, key in enumerate(["ok", "setup", "no"]):
            x0 = gap + idx * (bar_width + gap)
            bar_height = int((self.counts[key] / total) * (chart_height - 6))
            y0 = chart_bottom - bar_height
            self.create_rectangle(
                x0, y0, x0 + bar_width, chart_bottom,
                fill=colors[key],
                outline=""
            )
            self.create_text(
                x0 + bar_width / 2,
                chart_bottom + 12,
                text=key.upper(),
                fill=ModernStyle.TEXT_SECONDARY,
                font=ModernStyle.FONT_MAIN
            )
            self.create_text(
                x0 + bar_width / 2,
                y0 - 8,
                text=str(self.counts[key]),
                fill=ModernStyle.TEXT_PRIMARY,
                font=ModernStyle.FONT_MAIN
            )


class WorkflowDiagram(tk.Canvas):
    """Workflow diagram with active step highlight."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=200,
            bg=ModernStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.steps = [
            {"title": "1D", "subtitle": "Trend Bias"},
            {"title": "4H", "subtitle": "Market Bias"},
            {"title": "1H", "subtitle": "Impulse"},
            {"title": "15M", "subtitle": "Confirm"},
            {"title": "5M", "subtitle": "Trigger"},
            {"title": "Score", "subtitle": "Evaluate"},
            {"title": "Report", "subtitle": "Export"},
        ]
        self.active_index = -1
        self._pulse_active = False
        self._pulse_phase = 0
        self.bind("<Configure>", lambda _: self._draw())

    def set_stage(self, stage_text: str) -> None:
        stage = stage_text.lower()
        mapping = {
            "fetch 1d": 0,
            "fetch 4h": 1,
            "fetch 1h": 2,
            "fetch 15m": 3,
            "fetch 5m": 4,
        }
        idx = None
        for key, val in mapping.items():
            if key in stage:
                idx = val
                break
        if idx is None:
            if "analiz" in stage or "analysis" in stage:
                idx = 2
            elif "score" in stage:
                idx = 5
            elif "report" in stage or "tamamlandı" in stage:
                idx = 6
        if idx is not None:
            self.active_index = idx
            self._start_pulse()
            self._draw()

    def reset(self) -> None:
        self.active_index = -1
        self._pulse_active = False
        self._draw()

    def _start_pulse(self):
        if not self._pulse_active:
            self._pulse_active = True
            self._pulse()

    def _pulse(self):
        if not self._pulse_active or self.active_index < 0:
            return
        self._pulse_phase = (self._pulse_phase + 1) % 4
        self._draw()
        self.after(220, self._pulse)

    def _rounded_rect(self, x0, y0, x1, y1, radius=12, **kwargs):
        points = [
            x0 + radius, y0,
            x1 - radius, y0,
            x1, y0,
            x1, y0 + radius,
            x1, y1 - radius,
            x1, y1,
            x1 - radius, y1,
            x0 + radius, y1,
            x0, y1,
            x0, y1 - radius,
            x0, y0 + radius,
            x0, y0,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw(self):
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return

        padding_x = 24
        padding_y = 18
        gap = 16
        row_gap = 22
        card_height = 64
        card_radius = 14
        top_y = padding_y + 46

        card_min_width = 88
        card_max_width = 130
        usable_width = max(1, width - (padding_x * 2))
        max_cols = max(1, int((usable_width + gap) / (card_min_width + gap)))
        cols = min(len(self.steps), max_cols)
        rows = (len(self.steps) + cols - 1) // cols
        available_width = usable_width - (gap * (cols - 1))
        card_width = max(card_min_width, min(card_max_width, int(available_width / cols)))

        self.create_text(
            padding_x,
            padding_y,
            text="Workflow",
            fill=ModernStyle.TEXT_SECONDARY,
            anchor="w",
            font=ModernStyle.FONT_HEADER
        )

        legend_items = [
            ("Pending", ModernStyle.TEXT_MUTED),
            ("Active", ModernStyle.ACCENT_PRIMARY),
            ("Complete", ModernStyle.ACCENT_SUCCESS),
        ]
        legend_x = padding_x + 120
        legend_y = padding_y + 2
        for label, color in legend_items:
            self.create_oval(
                legend_x,
                legend_y,
                legend_x + 10,
                legend_y + 10,
                fill=color,
                outline="",
            )
            self.create_text(
                legend_x + 16,
                legend_y + 5,
                text=label,
                fill=ModernStyle.TEXT_MUTED,
                anchor="w",
                font=ModernStyle.FONT_MAIN,
            )
            legend_x += 80

        for idx, step in enumerate(self.steps):
            row = idx // cols
            col = idx % cols
            x0 = padding_x + col * (card_width + gap)
            x1 = x0 + card_width
            y0 = top_y + row * (card_height + row_gap)
            y1 = y0 + card_height
            is_active = idx == self.active_index
            is_complete = self.active_index >= 0 and idx < self.active_index

            shadow_color = "#0c111b"
            self._rounded_rect(
                x0 + 2,
                y0 + 2,
                x1 + 2,
                y1 + 2,
                radius=card_radius,
                fill=shadow_color,
                outline=""
            )

            base_color = ModernStyle.BG_LIGHT
            border_color = ModernStyle.BORDER
            text_color = ModernStyle.TEXT_SECONDARY
            subtitle_color = ModernStyle.TEXT_SECONDARY

            if is_complete:
                base_color = "#223349"
                border_color = ModernStyle.ACCENT_SUCCESS
                text_color = ModernStyle.TEXT_PRIMARY
                subtitle_color = ModernStyle.TEXT_SECONDARY

            if is_active:
                base_color = "#1c3551"
                border_color = ModernStyle.ACCENT_PRIMARY
                text_color = ModernStyle.TEXT_PRIMARY
                subtitle_color = ModernStyle.TEXT_SECONDARY

                glow_colors = [
                    ModernStyle.ACCENT_PRIMARY,
                    ModernStyle.ACCENT_SUCCESS,
                    ModernStyle.ACCENT_PRIMARY,
                    "#4fd3ff",
                ]
                glow_color = glow_colors[self._pulse_phase]
                self._rounded_rect(
                    x0 - 4,
                    y0 - 4,
                    x1 + 4,
                    y1 + 4,
                    radius=card_radius + 2,
                    fill="",
                    outline=glow_color,
                    width=2
                )

            self._rounded_rect(
                x0,
                y0,
                x1,
                y1,
                radius=card_radius,
                fill=base_color,
                outline=border_color,
                width=1
            )

            self.create_text(
                (x0 + x1) / 2,
                y0 + 24,
                text=step["title"],
                fill=text_color,
                font=ModernStyle.FONT_HEADER
            )
            self.create_text(
                (x0 + x1) / 2,
                y0 + 44,
                text=step["subtitle"],
                fill=subtitle_color,
                font=ModernStyle.FONT_MAIN
            )

            is_row_end = col == cols - 1
            is_last = idx == len(self.steps) - 1
            if not is_row_end and not is_last:
                line_x0 = x1 + 4
                line_x1 = x1 + gap - 4
                line_y = y0 + card_height + 10
                line_color = ModernStyle.TEXT_SECONDARY
                if is_complete:
                    line_color = ModernStyle.ACCENT_SUCCESS
                self.create_line(
                    line_x0,
                    line_y,
                    line_x1,
                    line_y,
                    fill=line_color,
                    width=3,
                    capstyle=tk.ROUND
                )
                arrow_x = line_x1
                self.create_polygon(
                    arrow_x - 6,
                    line_y - 5,
                    arrow_x,
                    line_y,
                    arrow_x - 6,
                    line_y + 5,
                    fill=line_color,
                    outline=""
                )
            if row < rows - 1 and is_row_end:
                connector_x = x0 + card_width / 2
                connector_y0 = y1 + 6
                connector_y1 = y1 + row_gap - 6
                self.create_line(
                    connector_x,
                    connector_y0,
                    connector_x,
                    connector_y1,
                    fill=ModernStyle.TEXT_SECONDARY,
                    width=2
                )

class ModernButton(tk.Canvas):
    """Hover effektli custom button"""
    
    def __init__(self, parent, text, command, **kwargs):
        self.text = text
        self.command = command
        self.hovered = False
        self.disabled = False
        
        # Extract custom colors if provided, otherwise None (use defaults in _draw)
        self.custom_bg = kwargs.pop("btn_bg", None)
        self.custom_fg = kwargs.pop("btn_fg", None)
        
        super().__init__(
            parent,
            height=36,
            bg=ModernStyle.BG_DARK,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            **kwargs
        )
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        
        self._draw()
    
    def _draw(self):
        self.delete("all")
        
        width = self.winfo_width() if self.winfo_width() > 1 else 120
        height = self.winfo_height() if self.winfo_height() > 1 else 36
        
        # Background
        if self.disabled:
            bg_color = ModernStyle.BG_PANEL
        else:
            if self.hovered:
                bg_color = ModernStyle.ACCENT_PRIMARY 
            else:
                # Use custom bg if provided, else default light bg
                bg_color = self.custom_bg if self.custom_bg else ModernStyle.BG_LIGHT
                
        self.create_rectangle(
            2, 2, width-2, height-2,
            fill=bg_color,
            outline=ModernStyle.ACCENT_PRIMARY,
            width=1
        )
        
        # Text
        if self.disabled:
            text_color = ModernStyle.TEXT_MUTED
        else:
            if self.hovered:
                text_color = ModernStyle.BG_DARK
            else:
                # Use custom fg if provided, else default primary text
                text_color = self.custom_fg if self.custom_fg else ModernStyle.TEXT_PRIMARY
                
        self.create_text(
            width // 2, height // 2,
            text=self.text,
            fill=text_color,
            font=ModernStyle.FONT_HEADER
        )
    
    def _on_enter(self, event):
        if not self.disabled:
            self.hovered = True
            self._draw()
    
    def _on_leave(self, event):
        if not self.disabled:
            self.hovered = False
            self._draw()
    
    def _on_click(self, event):
        if self.command and not self.disabled:
            self.command()

    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled
        self.hovered = False
        self._draw()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Trade Bot - Professional Edition")
        self.root.configure(bg=ModernStyle.BG_DARK)
        
        # Window konfiqurasiyası
        self.root.geometry("1200x880")
        self.root.minsize(1040, 760)
        
        # Queue və thread
        self._q: queue.Queue = queue.Queue()
        self._scan_thread: Optional[threading.Thread] = None
        self._spinning = False
        self._spinner_i = 0
        self._spinner_frames = ["◐", "◓", "◑", "◒"]
        self.latest_best_plan = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_event = threading.Event()
        self._monitoring = False
        
        # Settings
        self.settings = load_settings("settings.json")
        
        # News/Sentiment Analyzer
        self.news_analyzer = NewsAnalyzer()
        
        # Position Monitor
        self.position_monitor = PositionMonitor(self.settings, self._q)
        
        # Dashboard separate window
        self.dashboard_window = None

        # Style konfiqurasiyası
        self._setup_styles()

        # Scrollable əsas kontent
        self._create_scroll_container()
        
        # UI Elementləri
        self._create_header()
        self._create_parameters_section()
        self._create_symbols_section()
        self._create_scan_section()
        self._create_output_section()
        self._create_news_section()
        
        # Queue polling
        self.root.after(120, self._poll_queue)

    def _play_success_sound(self):
        """Play a 5-second sound signal in a separate thread."""
        def sound_worker():
            try:
                # Try macOS afplay first (Glass sound is pleasant)
                for _ in range(3): # Play 3 times (~5 seconds total duration for Glass.aiff)
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
                    time.sleep(0.5)
            except Exception:
                # Fallback to system bell loop
                for _ in range(5):
                    self.root.bell()
                    print('\a')
                    time.sleep(1)
        
        threading.Thread(target=sound_worker, daemon=True).start()

    def _create_scroll_container(self) -> None:
        self.page_container = ttk.Frame(self.root, style="Dark.TFrame")
        self.page_container.pack(fill="both", expand=True)

        self.page_canvas = tk.Canvas(
            self.page_container,
            bg=ModernStyle.BG_DARK,
            highlightthickness=0,
            bd=0,
        )
        self.page_scrollbar = ttk.Scrollbar(
            self.page_container,
            orient="vertical",
            command=self.page_canvas.yview,
        )
        self.page_canvas.configure(yscrollcommand=self.page_scrollbar.set)

        self.page_scrollbar.pack(side="right", fill="y")
        self.page_canvas.pack(side="left", fill="both", expand=True)

        self.page_frame = ttk.Frame(
            self.page_canvas,
            style="Dark.TFrame",
            padding=(0, 10),
        )
        self.page_window = self.page_canvas.create_window(
            (0, 0),
            window=self.page_frame,
            anchor="nw",
        )

        def on_frame_configure(_event):
            self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all"))

        def on_canvas_configure(event):
            self.page_canvas.itemconfigure(self.page_window, width=event.width)

        self.page_frame.bind("<Configure>", on_frame_configure)
        self.page_canvas.bind("<Configure>", on_canvas_configure)
        self._bind_page_scroll()

    def _bind_page_scroll(self) -> None:
        def on_mousewheel(event):
            if hasattr(event, "delta") and event.delta:
                self.page_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                self.page_canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                self.page_canvas.yview_scroll(3, "units")

        self.page_canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.page_canvas.bind_all("<Button-4>", on_mousewheel)
        self.page_canvas.bind_all("<Button-5>", on_mousewheel)
    
    def _setup_styles(self):
        """TTK stil konfiqurasiyası"""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TFrame",
            background=ModernStyle.BG_DARK
        )

        style.configure(
            "TLabel",
            background=ModernStyle.BG_MEDIUM,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_MAIN
        )
        
        # Frame stil
        style.configure(
            "Dark.TFrame",
            background=ModernStyle.BG_DARK
        )
        
        style.configure(
            "Card.TFrame",
            background=ModernStyle.BG_MEDIUM,
            relief="flat"
        )
        
        # Label stil
        style.configure(
            "Header.TLabel",
            background=ModernStyle.BG_DARK,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_TITLE
        )

        style.configure(
            "SubHeader.TLabel",
            background=ModernStyle.BG_DARK,
            foreground=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SUBTITLE
        )
        
        style.configure(
            "Normal.TLabel",
            background=ModernStyle.BG_MEDIUM,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_MAIN
        )
        
        style.configure(
            "Secondary.TLabel",
            background=ModernStyle.BG_MEDIUM,
            foreground=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_MAIN
        )

        style.configure(
            "Muted.TLabel",
            background=ModernStyle.BG_MEDIUM,
            foreground=ModernStyle.TEXT_MUTED,
            font=ModernStyle.FONT_MAIN
        )
        
        # Entry stil
        style.configure(
            "Modern.TEntry",
            fieldbackground=ModernStyle.BG_LIGHT,
            foreground=ModernStyle.TEXT_PRIMARY,
            borderwidth=1,
            relief="flat"
        )
        
        # LabelFrame stil
        style.configure(
            "Card.TLabelframe",
            background=ModernStyle.BG_MEDIUM,
            foreground=ModernStyle.TEXT_PRIMARY,
            borderwidth=1,
            relief="solid"
        )
        
        style.configure(
            "Card.TLabelframe.Label",
            background=ModernStyle.BG_MEDIUM,
            foreground=ModernStyle.ACCENT_PRIMARY,
            font=ModernStyle.FONT_HEADER
        )

        style.configure(
            "Modern.TRadiobutton",
            background=ModernStyle.BG_MEDIUM,
            foreground=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_MAIN
        )

        style.configure(
            "Modern.TSpinbox",
            fieldbackground=ModernStyle.BG_LIGHT,
            foreground=ModernStyle.TEXT_PRIMARY,
            background=ModernStyle.BG_MEDIUM,
            borderwidth=1,
            relief="flat"
        )
    
    def _create_header(self):
        """Header bölməsi"""
        header = ttk.Frame(self.page_frame, style="Dark.TFrame", padding=18)
        header.pack(fill="x")
        
        # Title
        title_frame = ttk.Frame(header, style="Dark.TFrame")
        title_frame.pack(side="left", fill="x", expand=True)
        
        ttk.Label(
            title_frame,
            text="⚡ TRADE BOT",
            style="Header.TLabel"
        ).pack(side="left")

        ttk.Label(
            title_frame,
            text="Professional multi-timeframe scanner & plan builder",
            style="SubHeader.TLabel"
        ).pack(side="left", padx=(14, 0))
        
        # Status indicator
        self.status_indicator = StatusIndicator(header)
        self.status_indicator.pack(side="right", padx=(10, 0))
        
        self.status_var = tk.StringVar(value="Hazır")
        status_label = ttk.Label(
            header,
            textvariable=self.status_var,
            style="SubHeader.TLabel"
        )
        status_label.pack(side="right", padx=10)
        
        # Ping button
        ping_btn = ModernButton(
            header,
            text="📡 Binance",
            command=self.on_ping,
            width=120
        )
        ping_btn.pack(side="right", padx=5)
    
    def _create_parameters_section(self):
        """Parametrlər bölməsi"""
        params_card = ttk.LabelFrame(
            self.page_frame,
            text="⚙️  Parametrlər",
            style="Card.TLabelframe",
            padding=20
        )
        params_card.pack(fill="x", padx=15, pady=(10, 5))

        ttk.Label(
            params_card,
            text="Risk idarəetməsi və ölçüləmə üçün əsas parametrlər.",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(0, 10))
        
        # Variables
        self.budget_var = tk.DoubleVar(
            value=float(self.settings.get("budget", {}).get("default_usdt", 5.0))
        )
        self.risk_pct_var = tk.DoubleVar(
            value=float(self.settings.get("risk", {}).get("risk_pct", 0.10))
        )
        self.lev_var = tk.IntVar(
            value=int(self.settings.get("risk", {}).get("leverage", 3))
        )
        
        # Grid layout
        params_grid = ttk.Frame(params_card, style="Card.TFrame")
        params_grid.pack(fill="x")
        
        # Budget
        self._create_param_field(
            params_grid, 0,
            "💰 Budget (USDT)",
            self.budget_var,
            width=15
        )
        
        # Risk
        self._create_param_field(
            params_grid, 1,
            "⚠️  Risk % (0.10 = 10%)",
            self.risk_pct_var,
            width=15
        )
        
        # Leverage
        self._create_param_field(
            params_grid, 2,
            "📊 Leverage",
            self.lev_var,
            width=10
        )
        
        # Save button
        save_container = ttk.Frame(params_grid, style="Card.TFrame")
        save_container.grid(row=0, column=6, rowspan=2, padx=(20, 0), sticky="e")
        
        self.btn_save = ModernButton(
            save_container,
            text="💾 Saxla",
            command=self.on_save,
            width=100
        )
        self.btn_save.pack()

        params_grid.columnconfigure(0, weight=1)
        params_grid.columnconfigure(2, weight=1)
        params_grid.columnconfigure(4, weight=1)
    
    def _create_param_field(self, parent, col, label_text, variable, width=10):
        """Parameter field yaradır"""
        ttk.Label(
            parent,
            text=label_text,
            style="Normal.TLabel"
        ).grid(row=0, column=col*2, sticky="w", padx=(0, 8), pady=5)
        
        entry = ttk.Entry(
            parent,
            textvariable=variable,
            width=width,
            style="Modern.TEntry",
            font=ModernStyle.FONT_MAIN
        )
        entry.grid(row=1, column=col*2, sticky="w", padx=(0, 20))

    def _create_symbols_section(self):
        """Simvol seçimi bölməsi"""
        symbols_card = ttk.LabelFrame(
            self.page_frame,
            text="🧩 Simvol Seçimi",
            style="Card.TLabelframe",
            padding=20
        )
        symbols_card.pack(fill="x", padx=15, pady=5)

        ttk.Label(
            symbols_card,
            text="Skan üçün coin seçim mənbəyini və Top limit dəyərini buradan tənzimlə.",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(0, 10))

        symbols_settings = self.settings.get("symbols", {})
        if bool(symbols_settings.get("auto_top_usdtm", False)):
            mode = "top"
        elif bool(symbols_settings.get("auto_all_usdtm", False)):
            mode = "all"
        else:
            mode = "manual"

        self.symbol_mode_var = tk.StringVar(value=mode)
        self.top_limit_var = tk.IntVar(value=int(symbols_settings.get("top_limit", 200)))

        modes_frame = ttk.Frame(symbols_card, style="Card.TFrame")
        modes_frame.pack(fill="x")

        ttk.Radiobutton(
            modes_frame,
            text="Top USDT-M (həcmə görə)",
            variable=self.symbol_mode_var,
            value="top",
            style="Modern.TRadiobutton",
            command=self._update_symbols_controls
        ).grid(row=0, column=0, sticky="w", padx=(0, 18), pady=4)

        ttk.Radiobutton(
            modes_frame,
            text="Bütün USDT-M",
            variable=self.symbol_mode_var,
            value="all",
            style="Modern.TRadiobutton",
            command=self._update_symbols_controls
        ).grid(row=0, column=1, sticky="w", padx=(0, 18), pady=4)

        ttk.Radiobutton(
            modes_frame,
            text="Manual (settings.json list)",
            variable=self.symbol_mode_var,
            value="manual",
            style="Modern.TRadiobutton",
            command=self._update_symbols_controls
        ).grid(row=0, column=2, sticky="w", pady=4)

        limit_frame = ttk.Frame(symbols_card, style="Card.TFrame")
        limit_frame.pack(fill="x", pady=(8, 0))

        ttk.Label(
            limit_frame,
            text="Top limit:",
            style="Normal.TLabel"
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.top_limit_spinbox = ttk.Spinbox(
            limit_frame,
            from_=1,
            to=1000,
            textvariable=self.top_limit_var,
            width=8,
            style="Modern.TSpinbox",
            font=ModernStyle.FONT_MAIN
        )
        self.top_limit_spinbox.grid(row=0, column=1, sticky="w")

        ttk.Label(
            limit_frame,
            text="(1-1000 arası, məsələn 10, 20, 200)",
            style="Muted.TLabel"
        ).grid(row=0, column=2, sticky="w", padx=(10, 0))

        self._update_symbols_controls()

    def _update_symbols_controls(self):
        mode = self.symbol_mode_var.get()
        if mode == "top":
            self.top_limit_spinbox.configure(state="normal")
        else:
            self.top_limit_spinbox.configure(state="disabled")

    def _create_scan_section(self):
        """Scan bölməsi"""
        scan_card = ttk.LabelFrame(
            self.page_frame,
            text="🔍 Skan",
            style="Card.TLabelframe",
            padding=20
        )
        scan_card.pack(fill="x", padx=15, pady=5)

        ttk.Label(
            scan_card,
            text="Dəqiq mərhələ izləmə, progress animasiyası və real-time status.",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(0, 10))
        
        # Info row
        info_frame = ttk.Frame(scan_card, style="Card.TFrame")
        info_frame.pack(fill="x", pady=(0, 10))
        
        self.symbols_count_var = tk.StringVar(value=self._symbols_count_text())
        ttk.Label(
            info_frame,
            textvariable=self.symbols_count_var,
            style="Normal.TLabel"
        ).pack(side="left")
        
        self.spinner_var = tk.StringVar(value="")
        spinner_label = ttk.Label(
            info_frame,
            textvariable=self.spinner_var,
            style="Normal.TLabel",
            font=("Segoe UI", 14)
        )
        spinner_label.pack(side="left", padx=10)
        
        # Stage info
        self.stage_var = tk.StringVar(value="Gözləyir...")
        stage_label = ttk.Label(
            scan_card,
            textvariable=self.stage_var,
            style="Secondary.TLabel"
        )
        stage_label.pack(fill="x", pady=(0, 8))
        
        # Progress bar
        progress_frame = ttk.Frame(scan_card, style="Card.TFrame")
        progress_frame.pack(fill="x", pady=(0, 12))

        self.progress = AnimatedProgressBar(progress_frame)
        self.progress.pack(side="left", fill="x", expand=True)

        self.progress_pct_var = tk.StringVar(value="0%")
        ttk.Label(
            progress_frame,
            textvariable=self.progress_pct_var,
            style="Normal.TLabel"
        ).pack(side="right", padx=(12, 0))
        
        # Run button
        btn_container = ttk.Frame(scan_card, style="Card.TFrame")
        btn_container.pack()
        
        self.btn_run = ModernButton(
            btn_container,
            text="▶️  BAŞLAT",
            command=self.on_scan,
            width=180
        )
        self.btn_run.pack()
    
    def _create_output_section(self):
        """Output bölməsi"""
        output_card = ttk.LabelFrame(
            self.page_frame,
            text="📄 Nəticələr",
            style="Card.TLabelframe",
            padding=15
        )
        output_card.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        content_container = ttk.Frame(output_card, style="Card.TFrame")
        content_container.pack(fill="both", expand=True)

        content_frame = ttk.Frame(content_container, style="Card.TFrame")
        content_frame.pack(fill="both", expand=True)

        summary_frame = ttk.Frame(content_frame, style="Card.TFrame")
        summary_frame.pack(fill="x", pady=(0, 10))

        stats_frame = ttk.Frame(summary_frame, style="Card.TFrame")
        stats_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.ok_var = tk.StringVar(value="OK: 0")
        self.setup_var = tk.StringVar(value="SETUP: 0")
        self.no_var = tk.StringVar(value="NO_TRADE: 0")
        self.best_var = tk.StringVar(value="Best: -")

        badges = [
            (self.ok_var, ModernStyle.ACCENT_SUCCESS),
            (self.setup_var, ModernStyle.ACCENT_WARNING),
            (self.no_var, ModernStyle.ACCENT_ERROR),
            (self.best_var, ModernStyle.ACCENT_INFO),
        ]
        for idx, (var, color) in enumerate(badges):
            badge = self._create_stat_badge(stats_frame, var, color)
            badge.grid(row=0, column=idx, padx=6, sticky="ew")

        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(2, weight=1)
        stats_frame.columnconfigure(3, weight=1)

        visuals_frame = ttk.Frame(summary_frame, style="Card.TFrame")
        visuals_frame.pack(side="right", fill="both", expand=True)

        self.summary_chart = SummaryChart(visuals_frame)
        self.summary_chart.pack(fill="x", pady=(0, 8))

        self.workflow = WorkflowDiagram(visuals_frame)
        self.workflow.pack(fill="x")

        best_frame = ttk.LabelFrame(
            content_frame,
            text="⭐ Ən Yüksək Ehtimallı Əməliyyat",
            style="Card.TLabelframe",
            padding=12,
        )
        best_frame.pack(fill="x", pady=(0, 10))

        self.best_headline_var = tk.StringVar(value="-")
        self.best_subtitle_var = tk.StringVar(value="Skan nəticəsi gözlənilir.")
        self.best_fit_pct_var = tk.StringVar(value="-")
        self.best_score_short_var = tk.StringVar(value="-")
        self.best_rr_short_var = tk.StringVar(value="-")
        self.best_entry_short_var = tk.StringVar(value="-")

        hero_frame = ttk.Frame(best_frame, style="Card.TFrame")
        hero_frame.pack(fill="x", pady=(0, 10))

        hero_left = tk.Frame(
            hero_frame,
            bg=ModernStyle.BG_ELEVATED,
            highlightthickness=1,
            highlightbackground=ModernStyle.ACCENT_PRIMARY,
        )
        hero_left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(
            hero_left,
            text="TOP SIGNAL",
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.ACCENT_INFO,
            font=ModernStyle.FONT_SUBTITLE,
        ).pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(
            hero_left,
            textvariable=self.best_headline_var,
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HERO,
        ).pack(anchor="w", padx=10)
        tk.Label(
            hero_left,
            textvariable=self.best_subtitle_var,
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_HERO_SUB,
        ).pack(anchor="w", padx=10, pady=(0, 8))

        hero_center = tk.Frame(
            hero_frame,
            bg=ModernStyle.BG_ELEVATED,
            highlightthickness=1,
            highlightbackground=ModernStyle.ACCENT_SUCCESS,
        )
        hero_center.pack(side="left", fill="both", expand=True, padx=8)

        tk.Label(
            hero_center,
            text="Uğurluluq faizi",
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.TEXT_SECONDARY,
            font=ModernStyle.FONT_SUBTITLE,
        ).pack(anchor="w", padx=10, pady=(8, 4))
        self.best_confidence_meter = ConfidenceMeter(hero_center)
        self.best_confidence_meter.pack(fill="x", padx=10)
        tk.Label(
            hero_center,
            textvariable=self.best_fit_pct_var,
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADER,
        ).pack(anchor="w", padx=10, pady=(4, 8))

        hero_right = ttk.Frame(hero_frame, style="Card.TFrame")
        hero_right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        metric_grid = ttk.Frame(hero_right, style="Card.TFrame")
        metric_grid.pack(fill="both", expand=True)

        metric_items = [
            ("Score", self.best_score_short_var, ModernStyle.ACCENT_PRIMARY),
            ("RR2", self.best_rr_short_var, ModernStyle.ACCENT_PURPLE),
            ("Entry", self.best_entry_short_var, ModernStyle.ACCENT_INFO),
        ]
        for idx, (title, var, accent) in enumerate(metric_items):
            tile = self._create_metric_tile(metric_grid, title, var, accent)
            tile.grid(row=0, column=idx, padx=6, sticky="ew")
            metric_grid.columnconfigure(idx, weight=1)

        self.best_symbol_var = tk.StringVar(value="-")
        self.best_status_var = tk.StringVar(value="-")
        self.best_side_var = tk.StringVar(value="-")
        self.best_entry_var = tk.StringVar(value="-")
        self.best_sl_var = tk.StringVar(value="-")
        self.best_tp1_var = tk.StringVar(value="-")
        self.best_tp2_var = tk.StringVar(value="-")
        self.best_rr_var = tk.StringVar(value="-")
        self.best_fit_var = tk.StringVar(value="-")
        self.best_score_var = tk.StringVar(value="-")
        self.best_qty_var = tk.StringVar(value="-")
        self.best_leverage_var = tk.StringVar(value="-")
        self.best_risk_var = tk.StringVar(value="-")
        self.best_form_market_var = tk.StringVar(value="-")
        self.best_form_margin_var = tk.StringVar(value="Isolated")
        self.best_form_leverage_var = tk.StringVar(value="-")
        self.best_form_tab_var = tk.StringVar(value="Limit")
        self.best_form_price_var = tk.StringVar(value="-")
        self.best_form_qty_var = tk.StringVar(value="-")
        self.best_form_tp_var = tk.StringVar(value="-")
        self.best_form_sl_var = tk.StringVar(value="-")
        self.best_form_tif_var = tk.StringVar(value="GTC")
        self.best_form_action_var = tk.StringVar(value="-")
        self.best_form_reduce_only_var = tk.StringVar(value="OFF (entry açırsan)")
        self.pending_order_type_var = tk.StringVar(value="Limit (Pending)")
        self.pending_order_entry_trigger_var = tk.StringVar(value="Mark")
        self.pending_order_tpsl_trigger_var = tk.StringVar(value="Mark")
        self.pending_order_symbol_var = tk.StringVar(value="-")
        self.pending_order_side_var = tk.StringVar(value="-")
        self.pending_order_entry_var = tk.StringVar(value="-")
        self.pending_order_qty_var = tk.StringVar(value="-")
        self.pending_order_tp_var = tk.StringVar(value="-")
        self.pending_order_sl_var = tk.StringVar(value="-")
        self.pending_order_leverage_var = tk.StringVar(value="-")
        self.pending_order_margin_var = tk.StringVar(value="Isolated")
        expiry_days = int(self.settings.get("risk", {}).get("max_orders_expiry_days", 7))
        self.pending_order_expiry_var = tk.StringVar(value=f"{expiry_days} gün")
        self.pending_order_notional_var = tk.StringVar(value="-")
        self.pending_order_status_var = tk.StringVar(value="-")
        self.monitor_symbol_var = tk.StringVar(value="")
        self.monitor_interval_var = tk.IntVar(value=5)
        self.monitor_status_var = tk.StringVar(value="Monitor gözləmədədir.")
        self.monitor_signal_var = tk.StringVar(value="-")
        self.monitor_side_var = tk.StringVar(value="-")
        self.monitor_entry_var = tk.StringVar(value="-")
        self.monitor_mark_var = tk.StringVar(value="-")
        self.monitor_pnl_var = tk.StringVar(value="-")
        self.monitor_roi_var = tk.StringVar(value="-")
        self.monitor_leverage_var = tk.StringVar(value="-")
        self.monitor_liq_var = tk.StringVar(value="-")

        best_grid = ttk.Frame(best_frame, style="Card.TFrame")
        best_grid.pack(fill="x")

        rows = [
            ("Simvol", self.best_symbol_var),
            ("Status", self.best_status_var),
            ("Side", self.best_side_var),
            ("Entry", self.best_entry_var),
            ("Stop Loss", self.best_sl_var),
            ("TP1", self.best_tp1_var),
            ("TP2", self.best_tp2_var),
            ("RR1/RR2", self.best_rr_var),
            ("Fit %", self.best_fit_var),
            ("Score", self.best_score_var),
            ("Qty", self.best_qty_var),
            ("Leverage", self.best_leverage_var),
            ("Risk (Target/Actual)", self.best_risk_var),
        ]

        for idx, (label, var) in enumerate(rows):
            row = idx // 4
            col = (idx % 4) * 2
            ttk.Label(best_grid, text=f"{label}:", style="Secondary.TLabel").grid(
                row=row, column=col, sticky="w", padx=(0, 6), pady=3
            )
            ttk.Label(best_grid, textvariable=var, style="Normal.TLabel").grid(
                row=row, column=col + 1, sticky="w", padx=(0, 18), pady=3
            )

        for col in range(8):
            best_grid.columnconfigure(col, weight=1)

        details_frame = ttk.Frame(best_frame, style="Card.TFrame")
        details_frame.pack(fill="x", pady=(8, 0))
        
        # Action Buttons Frame
        action_frame = ttk.Frame(best_frame, style="Card.TFrame")
        action_frame.pack(fill="x", pady=(10, 0))
        
        self.btn_execute = ModernButton(
            action_frame,
            text="🚀 Execute Best Trade",
            command=self.on_execute_best,
            width=220,
            btn_bg=ModernStyle.ACCENT_PURPLE,
            btn_fg=ModernStyle.TEXT_PRIMARY
        )
        self.btn_execute.pack(side="left")
        
        ttk.Label(
            action_frame,
            text="⚠️ Real əməliyyat açır (Limit + TP/SL)",
            style="Muted.TLabel"
        ).pack(side="left", padx=10)

        ttk.Label(details_frame, text="Əlavə məlumat / səbəb:", style="Secondary.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        self.best_details = tk.Text(
            details_frame,
            height=4,
            wrap="word",
            bg=ModernStyle.BG_LIGHT,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.ACCENT_PRIMARY,
            font=ModernStyle.FONT_MAIN,
            relief="flat",
            padx=8,
            pady=6,
        )
        self.best_details.pack(fill="x")
        self.best_details.configure(state="disabled")

        manual_frame = ttk.LabelFrame(
            best_frame,
            text="✅ Manual Əməliyyat Addımları",
            style="Card.TLabelframe",
            padding=10,
        )
        manual_frame.pack(fill="x", pady=(10, 0))

        self.manual_steps_vars = [
            tk.StringVar(value="1) Simvolu və tərəfi təsdiqlə."),
            tk.StringVar(value="2) Entry qiymətinin zona ilə uyğunluğunu yoxla."),
            tk.StringVar(value="3) Qty və leverage dəyərlərini tətbiq et."),
            tk.StringVar(value="4) TP2 və SL limitlərini yerləşdir."),
            tk.StringVar(value="5) TIF = GTC, Reduce-only = OFF (entry açırsan)."),
            tk.StringVar(value="6) SETUP olduqda 5m təsdiqini gözlə."),
        ]
        for var in self.manual_steps_vars:
            tk.Label(
                manual_frame,
                textvariable=var,
                bg=ModernStyle.BG_MEDIUM,
                fg=ModernStyle.TEXT_SECONDARY,
                font=ModernStyle.FONT_MAIN,
                anchor="w",
            ).pack(fill="x", pady=2)

        form_frame = ttk.LabelFrame(
            best_frame,
            text="🧾 Binance Futures Form",
            style="Card.TLabelframe",
            padding=10,
        )
        form_frame.pack(fill="x", pady=(10, 0))

        form_grid = ttk.Frame(form_frame, style="Card.TFrame")
        form_grid.pack(fill="x")

        form_rows = [
            ("Market", self.best_form_market_var),
            ("Margin", self.best_form_margin_var),
            ("Leverage", self.best_form_leverage_var),
            ("Tab", self.best_form_tab_var),
            ("Price (Entry)", self.best_form_price_var),
            ("Size (Qty)", self.best_form_qty_var),
            ("Take Profit (TP2)", self.best_form_tp_var),
            ("Stop Loss", self.best_form_sl_var),
            ("TIF", self.best_form_tif_var),
            ("Action", self.best_form_action_var),
            ("Reduce-Only", self.best_form_reduce_only_var),
        ]

        for idx, (label, var) in enumerate(form_rows):
            row = idx // 3
            col = (idx % 3) * 2
            ttk.Label(form_grid, text=f"{label}:", style="Secondary.TLabel").grid(
                row=row, column=col, sticky="w", padx=(0, 6), pady=3
            )
            ttk.Label(form_grid, textvariable=var, style="Normal.TLabel").grid(
                row=row, column=col + 1, sticky="w", padx=(0, 18), pady=3
            )

        for col in range(6):
            form_grid.columnconfigure(col, weight=1)

        pending_frame = ttk.LabelFrame(
            best_frame,
            text="🟡 Pending Order Detalları",
            style="Card.TLabelframe",
            padding=10,
        )
        pending_frame.pack(fill="x", pady=(10, 0))

        pending_grid = ttk.Frame(pending_frame, style="Card.TFrame")
        pending_grid.pack(fill="x")

        pending_rows = [
            ("Market", self.pending_order_symbol_var),
            ("Side", self.pending_order_side_var),
            ("Order Type", self.pending_order_type_var),
            ("Entry", self.pending_order_entry_var),
            ("Size (Qty)", self.pending_order_qty_var),
            ("Notional", self.pending_order_notional_var),
            ("Take Profit (TP2)", self.pending_order_tp_var),
            ("Stop Loss", self.pending_order_sl_var),
            ("Leverage", self.pending_order_leverage_var),
            ("Margin", self.pending_order_margin_var),
            ("Entry Trigger", self.pending_order_entry_trigger_var),
            ("TP/SL Trigger", self.pending_order_tpsl_trigger_var),
            ("Action", self.best_form_action_var),
            ("TIF", self.best_form_tif_var),
            ("Reduce-Only", self.best_form_reduce_only_var),
            ("Expiry", self.pending_order_expiry_var),
            ("Pending Status", self.pending_order_status_var),
        ]

        for idx, (label, var) in enumerate(pending_rows):
            row = idx // 3
            col = (idx % 3) * 2
            ttk.Label(pending_grid, text=f"{label}:", style="Secondary.TLabel").grid(
                row=row, column=col, sticky="w", padx=(0, 6), pady=3
            )
            ttk.Label(pending_grid, textvariable=var, style="Normal.TLabel").grid(
                row=row, column=col + 1, sticky="w", padx=(0, 18), pady=3
            )

        for col in range(6):
            pending_grid.columnconfigure(col, weight=1)

        monitor_frame = ttk.LabelFrame(
            best_frame,
            text="🛰️ Trade Monitor (Order & Position nəzarəti)",
            style="Card.TLabelframe",
            padding=10,
        )
        monitor_frame.pack(fill="x", pady=(10, 0))

        monitor_controls = ttk.Frame(monitor_frame, style="Card.TFrame")
        monitor_controls.pack(fill="x", pady=(0, 6))

        ttk.Label(
            monitor_controls,
            text="Market/Symbol (opsional):",
            style="Secondary.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 6))

        ttk.Entry(
            monitor_controls,
            textvariable=self.monitor_symbol_var,
            width=12,
            style="Modern.TEntry",
            font=ModernStyle.FONT_MAIN,
        ).grid(row=0, column=1, sticky="w", padx=(0, 12))

        ttk.Label(
            monitor_controls,
            text="Refresh (s):",
            style="Secondary.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(0, 6))

        ttk.Spinbox(
            monitor_controls,
            from_=2,
            to=60,
            textvariable=self.monitor_interval_var,
            width=6,
            style="Modern.TSpinbox",
            font=ModernStyle.FONT_MAIN,
        ).grid(row=0, column=3, sticky="w", padx=(0, 12))

        ModernButton(
            monitor_controls,
            text="🧲 Best-ə tətbiq et",
            command=self._apply_best_to_monitor,
            width=160,
        ).grid(row=0, column=4, sticky="w", padx=(0, 8))

        self.btn_monitor_start = ModernButton(
            monitor_controls,
            text="▶️ Monitoru başlat",
            command=self.on_start_monitor,
            width=160,
        )
        self.btn_monitor_start.grid(row=0, column=5, sticky="w", padx=(0, 8))

        self.btn_monitor_stop = ModernButton(
            monitor_controls,
            text="⏹️ Dayandır",
            command=self.on_stop_monitor,
            width=120,
        )
        self.btn_monitor_stop.grid(row=0, column=6, sticky="w")
        self.btn_monitor_stop.set_disabled(True)

        monitor_controls.columnconfigure(7, weight=1)

        monitor_status_frame = ttk.Frame(monitor_frame, style="Card.TFrame")
        monitor_status_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(
            monitor_status_frame,
            textvariable=self.monitor_status_var,
            style="Normal.TLabel",
        ).pack(side="left")

        ttk.Label(
            monitor_status_frame,
            textvariable=self.monitor_signal_var,
            style="Secondary.TLabel",
        ).pack(side="right")

        monitor_grid = ttk.Frame(monitor_frame, style="Card.TFrame")
        monitor_grid.pack(fill="x", pady=(0, 8))

        monitor_rows = [
            ("Side", self.monitor_side_var),
            ("Entry", self.monitor_entry_var),
            ("Mark Price", self.monitor_mark_var),
            ("PnL (USDT)", self.monitor_pnl_var),
            ("ROI %", self.monitor_roi_var),
            ("Leverage", self.monitor_leverage_var),
            ("Liquidation", self.monitor_liq_var),
        ]

        for idx, (label, var) in enumerate(monitor_rows):
            row = idx // 3
            col = (idx % 3) * 2
            ttk.Label(monitor_grid, text=f"{label}:", style="Secondary.TLabel").grid(
                row=row, column=col, sticky="w", padx=(0, 6), pady=3
            )
            ttk.Label(monitor_grid, textvariable=var, style="Normal.TLabel").grid(
                row=row, column=col + 1, sticky="w", padx=(0, 18), pady=3
            )

        for col in range(6):
            monitor_grid.columnconfigure(col, weight=1)

        ttk.Label(
            monitor_frame,
            text="Active Positions (Deep Analysis)",
            style="Secondary.TLabel",
        ).pack(anchor="w", pady=(6, 4))

        # Compact Health View & Open Button
        health_frame = ttk.Frame(monitor_frame, style="Card.TFrame")
        health_frame.pack(fill="x", pady=5)
        
        # Open Dashboard Button
        ModernButton(
            health_frame,
            text="🖥️ Open Control Panel",
            command=self.on_open_dashboard,
            width=200
        ).pack(side="left")
        
        # Summary Label (Space filler / quick status)
        self.compact_status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(
            health_frame, 
            textvariable=self.compact_status_var,
            style="Secondary.TLabel"
        ).pack(side="left", padx=15)

        ttk.Separator(output_card, orient="horizontal").pack(fill="x", pady=(12, 8))

        ttk.Label(
            content_frame,
            text="📑 Ətraflı Hesabat",
            style="Normal.TLabel"
        ).pack(anchor="w", padx=6, pady=(0, 6))

        # Text widget with scrollbar
        txt_frame = ttk.Frame(content_frame, style="Card.TFrame")
        txt_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(txt_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.txt = tk.Text(
            txt_frame,
            wrap="word",
            bg=ModernStyle.BG_LIGHT,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.ACCENT_PRIMARY,
            selectbackground=ModernStyle.ACCENT_PRIMARY,
            selectforeground=ModernStyle.BG_DARK,
            font=ModernStyle.FONT_MONO,
            relief="flat",
            padx=10,
            pady=10,
            spacing1=4,
            spacing2=2,
            spacing3=4,
            yscrollcommand=scrollbar.set
        )
        self.txt.configure(height=16)
        self.txt.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.txt.yview)

    def _create_stat_badge(self, parent, var: tk.StringVar, accent: str) -> tk.Frame:
        frame = tk.Frame(
            parent,
            bg=ModernStyle.BG_ELEVATED,
            highlightthickness=1,
            highlightbackground=accent,
            highlightcolor=accent,
        )
        label = tk.Label(
            frame,
            textvariable=var,
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADER,
            padx=12,
            pady=8,
        )
        label.pack(fill="both", expand=True)
        return frame

    def _create_metric_tile(self, parent, title: str, var: tk.StringVar, accent: str) -> tk.Frame:
        frame = tk.Frame(
            parent,
            bg=ModernStyle.BG_ELEVATED,
            highlightthickness=1,
            highlightbackground=accent,
            highlightcolor=accent,
        )
        tk.Label(
            frame,
            text=title,
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.TEXT_MUTED,
            font=ModernStyle.FONT_SUBTITLE,
        ).pack(anchor="w", padx=10, pady=(6, 0))
        tk.Label(
            frame,
            textvariable=var,
            bg=ModernStyle.BG_ELEVATED,
            fg=ModernStyle.TEXT_PRIMARY,
            font=ModernStyle.FONT_HEADER,
        ).pack(anchor="w", padx=10, pady=(0, 8))
        return frame
    
    def _symbols_count_text(self) -> str:
        s = self.settings.get("symbols", {})
        if bool(s.get("auto_top_usdtm", False)):
            limit = int(s.get("top_limit", 200))
            return f"📊 Simvollar: Binance Top {limit} (USDT-M PERP)"
        if bool(s.get("auto_all_usdtm", False)):
            return "📊 Simvollar: AUTO (bütün USDT-M) — skan zamanı yüklənəcək"
        lst = s.get("list", []) or s.get("default", []) or []
        return f"📊 Simvollar: {len(lst)} ədəd (settings.json)"
    
    def on_ping(self):
        try:
            self.status_indicator.set_state("active")
            p = binance_data.ping()
            t = binance_data.server_time()
            self.status_var.set("Binance: ✓ Aktiv")
            self.status_indicator.set_state("success")
            self.txt.insert("end", f"✓ {p}\n✓ {t}\n\n")
            self.txt.see("end")
        except Exception as e:
            self.status_indicator.set_state("error")
            messagebox.showerror("Əlaqə Xətası", str(e))
    
    def on_save(self):
        self.settings.setdefault("budget", {})["default_usdt"] = float(self.budget_var.get())
        self.settings.setdefault("risk", {})["risk_pct"] = float(self.risk_pct_var.get())
        self.settings.setdefault("risk", {})["leverage"] = int(self.lev_var.get())
        symbols_settings = self.settings.setdefault("symbols", {})
        mode = self.symbol_mode_var.get()
        symbols_settings["auto_top_usdtm"] = mode == "top"
        symbols_settings["auto_all_usdtm"] = mode == "all"
        try:
            top_limit = int(self.top_limit_var.get())
        except (ValueError, tk.TclError):
            top_limit = int(symbols_settings.get("top_limit", 200))
        top_limit = max(1, top_limit)
        symbols_settings["top_limit"] = top_limit
        self.top_limit_var.set(top_limit)
        
        save_settings("settings.json", self.settings)
        self.symbols_count_var.set(self._symbols_count_text())
        self.status_var.set("✓ Saxlanıldı")
        self.status_indicator.set_state("success")
        
        # Reset status after 2 seconds
        self.root.after(2000, lambda: self.status_var.set("Hazır"))
        self.root.after(2000, lambda: self.status_indicator.set_state("idle"))
    
    def _set_busy(self, busy: bool):
        if busy:
            self.status_indicator.set_state("active")
        else:
            self.status_indicator.set_state("idle")
        self.btn_run.set_disabled(busy)
        self.btn_save.set_disabled(busy)
    
    def _start_spinner(self):
        self._spinning = True
        self._spinner_i = 0
        self._spin_tick()
    
    def _stop_spinner(self):
        self._spinning = False
        self.spinner_var.set("")
    
    def _spin_tick(self):
        if not self._spinning:
            return
        self.spinner_var.set(self._spinner_frames[self._spinner_i % len(self._spinner_frames)])
        self._spinner_i += 1
        self.root.after(120, self._spin_tick)
    
    def on_execute_best(self):
        """Execute the currently stored best plan."""
        plan = self.latest_best_plan
        if not plan:
            messagebox.showwarning("Xəbərdarlıq", "Hələ heç bir 'Best Plan' tapılmayıb.\nƏvvəlcə skan edin.")
            return
            
        symbol = plan.get("symbol")
        side = plan.get("side")
        qty = float(plan.get("qty", 0))
        entry = float(plan.get("entry", 0))
        sl = float(plan.get("sl", 0))
        tp = float(plan.get("tp2", 0))
        lev = int(plan.get("leverage", 1))
        
        msg = f"""Sifarişi təsdiqləyin:
        
Market: {symbol}
Side: {side}
Leverage: {lev}x
Qty: {qty}
Entry (Limit): {entry}
Stop Loss: {sl}
Take Profit: {tp}

Bu real əməliyyatdır! Davam edilsin?"""

        if not messagebox.askyesno("Təsdiq", msg, icon='warning'):
            return
            
        # Execute logic inside worker to stay responsive
        def trade_worker():
            self._q.put(("stage", "🚀 Sifariş göndərilir..."))
            try:
                res = binance_data.place_trade_setup(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    quantity=qty,
                    tp_price=tp,
                    sl_price=sl,
                    leverage=lev
                )
                
                status = res.get("status")
                logs = "\n".join(res.get("logs", []))
                
                if status == "success":
                    self._q.put(("info", f"✅ Uğurlu!\n\n{logs}"))
                else:
                    err = res.get("error", "Unknown error")
                    self._q.put(("error", f"❌ Xəta baş verdi:\n{err}\n\nLogs:\n{logs}"))
                    
            except Exception as e:
                self._q.put(("error", f"System Error: {e}"))
            
            self._q.put(("stage", "Hazır"))

        threading.Thread(target=trade_worker, daemon=True).start()

    def on_scan(self):
        self.on_save()
        
        self.txt.delete("1.0", "end")
        self.txt.insert("end", "🔄 Skan başlayır...\n\n")
        self._update_summary({"ok": 0, "setup": 0, "no": 0, "total": 0, "best": None, "best_fit": 0.0})
        self._set_best_plan(None)
        self.workflow.reset()
        
        self._set_busy(True)
        self._start_spinner()
        self.stage_var.set("⚙️  İnisializasiya...")
        self.progress.configure(value=0, maximum=100)
        self.progress_pct_var.set("0%")
        
        budget = float(self.budget_var.get())
        risk_pct = float(self.risk_pct_var.get())
        lev = int(self.lev_var.get())
        
        def worker():
            try:
                settings = load_settings("settings.json")
                
                sym_cfg = settings.get("symbols", {})
                auto_top_usdtm = bool(sym_cfg.get("auto_top_usdtm", False))
                auto_all_usdtm = bool(sym_cfg.get("auto_all_usdtm", False))
                if auto_top_usdtm:
                    limit = int(sym_cfg.get("top_limit", 200))
                    symbols = binance_data.list_usdtm_perp_symbols_by_volume(limit=limit)
                elif auto_all_usdtm:
                    symbols = binance_data.list_usdtm_perp_symbols()
                else:
                    symbols = sym_cfg.get("list", []) or sym_cfg.get("default", []) or []
                symbols = [str(x).upper().strip() for x in symbols if str(x).strip()]
                if (auto_top_usdtm or auto_all_usdtm) and hasattr(binance_data, "is_valid_usdtm_perp"):
                    symbols = [s for s in symbols if binance_data.is_valid_usdtm_perp(s)]
                
                if not symbols:
                    self._q.put(("error", "❌ settings.json-da symbols list boşdur"))
                    return
                
                def on_progress(i: int, total: int, symbol: str):
                    self._q.put(("progress", i, total, symbol))
                
                def on_stage(text: str):
                    self._q.put(("stage", text))
                
                results, best = run_scan_and_build_best_plan(
                    binance_data=binance_data,
                    analyzer=analyzer,
                    settings=settings,
                    symbols=symbols,
                    budget_usdt=budget,
                    risk_pct=risk_pct,
                    leverage=lev,
                    on_progress=on_progress,
                    on_stage=on_stage,
                )
                report = format_report(results, best, settings, snapshot_path=None)
                summary = {
                    "ok": len([r for r in results if r.status == "OK"]),
                    "setup": len([r for r in results if r.status == "SETUP"]),
                    "no": len([r for r in results if r.status == "NO_TRADE"]),
                    "total": len(results),
                    "best": best.symbol if best else None,
                    "best_fit": float(best.probability) if best else 0.0,
                }
                best_payload = self._build_best_payload(best)
                self._q.put(("done", report, summary, best_payload))
            except Exception as e:
                self._q.put(("error", str(e)))
        
        self._scan_thread = threading.Thread(target=worker, daemon=True)
        self._scan_thread.start()
    
    def _poll_queue(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                
                if kind == "progress":
                    _, i, total, symbol = msg
                    self.status_var.set(f"🔍 {symbol} ({i}/{total})")
                    self.stage_var.set(f"📊 Analiz: {symbol}")
                    self.workflow.set_stage("analiz")
                    if total > 0:
                        self.progress.configure(maximum=total, value=i)
                        pct = int((i / total) * 100)
                        self.progress_pct_var.set(f"{pct}%")
                
                elif kind == "stage":
                    _, text = msg
                    self.stage_var.set(f"⚙️  {text}")
                    self.workflow.set_stage(text)
                
                elif kind == "done":
                    _, report, summary, best_payload = msg
                    self.txt.delete("1.0", "end")
                    self.txt.insert("end", report)
                    self.status_var.set("✓ Tamamlandı")
                    self.status_indicator.set_state("success")
                    self._stop_spinner()
                    self._set_busy(False)
                    self.stage_var.set("✓ Hazır")
                    self.workflow.set_stage("report")
                    self._update_summary(summary)
                    self._set_best_plan(best_payload)
                    self.progress_pct_var.set("100%")
                    
                    # Sound Signal (5 seconds)
                    self._play_success_sound()
                
                elif kind == "error":
                    _, err = msg
                    self._stop_spinner()
                    self._set_busy(False)
                    self.status_indicator.set_state("error")
                    self.progress_pct_var.set("0%")
                    messagebox.showerror("Xəta", err)
                
                elif kind == "info":
                    _, text = msg
                    messagebox.showinfo("Məlumat", text)

                elif kind == "monitor_update":
                    _, payload = msg
                    self._update_monitor_display(payload)

                elif kind == "monitor_error":
                    _, err = msg
                    self.monitor_status_var.set(f"Monitor xətası: {err}")
                    self.monitor_signal_var.set("⚠️ API problem")

                elif kind == "position_analysis_update":
                    _, payload = msg
                    self._handle_position_update(payload)
                
                elif kind == "monitor_status":
                    _, status = msg
                    self.monitor_status_var.set(status)

                elif kind == "position_analysis_update":
                    _, payload = msg
                    self._handle_position_update(payload)
                
                elif kind == "monitor_status":
                    _, status = msg
                    self.monitor_status_var.set(status)

                elif kind == "position_analysis_update":
                    _, payload = msg
                    self._handle_position_update(payload)
                
                elif kind == "monitor_status":
                    _, status = msg
                    self.monitor_status_var.set(status)
        
        except queue.Empty:
            pass
        
        self.root.after(120, self._poll_queue)

    def _update_summary(self, summary: dict) -> None:
        ok = int(summary.get("ok", 0))
        setup = int(summary.get("setup", 0))
        no = int(summary.get("no", 0))
        total = int(summary.get("total", ok + setup + no))
        best = summary.get("best") or "-"
        best_fit = float(summary.get("best_fit", 0.0))
        self.ok_var.set(f"OK: {ok}")
        self.setup_var.set(f"SETUP: {setup}")
        self.no_var.set(f"NO_TRADE: {no}")
        self.best_var.set(f"Best: {best} ({best_fit:.1f}%) / {total}")
        self.summary_chart.set_counts(ok, setup, no)

    def _build_best_payload(self, best) -> Optional[dict]:
        if not best:
            return None
        return {
            "symbol": best.symbol,
            "status": best.status,
            "side": best.side,
            "entry": float(best.entry),
            "sl": float(best.sl),
            "tp1": float(best.tp1),
            "tp2": float(best.tp2),
            "rr1": float(best.rr1),
            "rr2": float(best.rr2),
            "fit": float(best.probability),
            "score": float(best.score),
            "qty": float(best.qty),
            "leverage": int(best.leverage),
            "risk_target": float(best.risk_target),
            "risk_actual": float(best.risk_actual),
            "reason": best.reason,
            "details": dict(best.details) if best.details else {},
        }

    def _set_best_plan(self, best: Optional[dict]) -> None:
        self.latest_best_plan = best
        if not best:
            self.best_headline_var.set("-")
            self.best_subtitle_var.set("Skan nəticəsi gözlənilir.")
            self.best_fit_pct_var.set("-")
            self.best_score_short_var.set("-")
            self.best_rr_short_var.set("-")
            self.best_entry_short_var.set("-")
            self.best_confidence_meter.set_value(0.0)
            placeholders = [
                (self.best_symbol_var, "-"),
                (self.best_status_var, "-"),
                (self.best_side_var, "-"),
                (self.best_entry_var, "-"),
                (self.best_sl_var, "-"),
                (self.best_tp1_var, "-"),
                (self.best_tp2_var, "-"),
                (self.best_rr_var, "-"),
                (self.best_fit_var, "-"),
                (self.best_score_var, "-"),
                (self.best_qty_var, "-"),
                (self.best_leverage_var, "-"),
                (self.best_risk_var, "-"),
            ]
            for var, value in placeholders:
                var.set(value)
            self._set_best_details("Skan nəticəsi gözlənilir.")
            self._set_best_form(None)
            self._set_pending_order_details(None)
            self._set_manual_steps(None)
            return

        symbol = best.get("symbol", "-")
        side = best.get("side", "-")
        status = best.get("status", "-")
        fit = float(best.get("fit", 0.0))
        score = float(best.get("score", 0.0))
        rr2 = float(best.get("rr2", 0.0))
        entry = float(best.get("entry", 0.0))

        self.best_headline_var.set(f"{symbol} • {side}")
        self.best_subtitle_var.set(f"{status} status • {fit:.1f}% ehtimal")
        self.best_fit_pct_var.set(f"Fit: {fit:.1f}%")
        self.best_score_short_var.set(f"{score:.2f}")
        self.best_rr_short_var.set(f"{rr2:.2f}")
        self.best_entry_short_var.set(f"{entry:.6f}")
        self.best_confidence_meter.set_value(fit)

        self.best_symbol_var.set(best.get("symbol", "-"))
        self.best_status_var.set(best.get("status", "-"))
        self.best_side_var.set(best.get("side", "-"))
        self.best_entry_var.set(f'{best.get("entry", 0.0):.6f}')
        self.best_sl_var.set(f'{best.get("sl", 0.0):.6f}')
        self.best_tp1_var.set(f'{best.get("tp1", 0.0):.6f}')
        self.best_tp2_var.set(f'{best.get("tp2", 0.0):.6f}')
        self.best_rr_var.set(f'{best.get("rr1", 0.0):.2f} / {best.get("rr2", 0.0):.2f}')
        self.best_fit_var.set(f'{best.get("fit", 0.0):.1f}%')
        self.best_score_var.set(f'{best.get("score", 0.0):.2f}')
        self.best_qty_var.set(f'{best.get("qty", 0.0):.6f}')
        self.best_leverage_var.set(f'{best.get("leverage", 0)}x')
        self.best_risk_var.set(
            f'{best.get("risk_target", 0.0):.4f} / {best.get("risk_actual", 0.0):.4f} USDT'
        )

        details_lines = [f"Səbəb: {best.get('reason', '-')}".strip()]
        for key, value in (best.get("details") or {}).items():
            details_lines.append(f"• {key}: {value}")
        self._set_best_details("\n".join(details_lines))
        self._set_best_form(best)
        self._set_pending_order_details(best)
        self._set_manual_steps(best)

    def _set_best_details(self, text: str) -> None:
        self.best_details.configure(state="normal")
        self.best_details.delete("1.0", "end")
        self.best_details.insert("end", text)
        self.best_details.configure(state="disabled")

    def _set_best_form(self, best: Optional[dict]) -> None:
        if not best:
            self.best_form_market_var.set("-")
            self.best_form_leverage_var.set("-")
            self.best_form_price_var.set("-")
            self.best_form_qty_var.set("-")
            self.best_form_tp_var.set("-")
            self.best_form_sl_var.set("-")
            self.best_form_action_var.set("-")
            self.best_form_tif_var.set("GTC")
            self.best_form_margin_var.set("Isolated")
            self.best_form_tab_var.set("Limit")
            self.best_form_reduce_only_var.set("OFF (entry açırsan)")
            return

        side = best.get("side", "-")
        self.best_form_market_var.set(best.get("symbol", "-"))
        self.best_form_leverage_var.set(f'{best.get("leverage", 0)}x')
        self.best_form_price_var.set(f'{best.get("entry", 0.0):.6f}')
        self.best_form_qty_var.set(f'{best.get("qty", 0.0):.6f}')
        self.best_form_tp_var.set(f'{best.get("tp2", 0.0):.6f}')
        self.best_form_sl_var.set(f'{best.get("sl", 0.0):.6f}')
        self.best_form_action_var.set("Buy/Long" if side == "LONG" else "Sell/Short")

    def _set_pending_order_details(self, best: Optional[dict]) -> None:
        if not best:
            self.pending_order_type_var.set("Limit (Pending)")
            self.pending_order_entry_trigger_var.set("Mark")
            self.pending_order_tpsl_trigger_var.set("Mark")
            self.pending_order_symbol_var.set("-")
            self.pending_order_side_var.set("-")
            self.pending_order_entry_var.set("-")
            self.pending_order_qty_var.set("-")
            self.pending_order_tp_var.set("-")
            self.pending_order_sl_var.set("-")
            self.pending_order_leverage_var.set("-")
            self.pending_order_margin_var.set("Isolated")
            self.pending_order_notional_var.set("-")
            self.pending_order_status_var.set("-")
            return

        symbol = best.get("symbol", "-")
        side = best.get("side", "-")
        entry = float(best.get("entry", 0.0))
        qty = float(best.get("qty", 0.0))
        tp2 = float(best.get("tp2", 0.0))
        sl = float(best.get("sl", 0.0))
        leverage = int(best.get("leverage", 0))
        notional = entry * qty
        status = best.get("status", "-")

        if status == "SETUP":
            pending_status = "Gözləmədə: 5m təsdiqi tələb olunur"
        elif status == "OK":
            pending_status = "Hazır: pending order yerləşdir"
        else:
            pending_status = "-"

        self.pending_order_type_var.set("Limit (Pending)")
        self.pending_order_entry_trigger_var.set("Mark")
        self.pending_order_tpsl_trigger_var.set("Mark")
        self.pending_order_symbol_var.set(symbol)
        self.pending_order_side_var.set(side)
        self.pending_order_entry_var.set(f"{entry:.6f}")
        self.pending_order_qty_var.set(f"{qty:.6f}")
        self.pending_order_tp_var.set(f"{tp2:.6f}")
        self.pending_order_sl_var.set(f"{sl:.6f}")
        self.pending_order_leverage_var.set(f"{leverage}x")
        self.pending_order_margin_var.set("Isolated")
        self.pending_order_notional_var.set(
            f"{notional:.4f} USDT" if notional > 0 else "-"
        )
        self.pending_order_status_var.set(pending_status)

    def _set_manual_steps(self, best: Optional[dict]) -> None:
        if not best:
            default_steps = [
                "1) Simvolu və tərəfi təsdiqlə.",
                "2) Entry qiymətinin zona ilə uyğunluğunu yoxla.",
                "3) Qty və leverage dəyərlərini tətbiq et.",
                "4) TP2 və SL limitlərini yerləşdir.",
                "5) TIF = GTC, Reduce-only = OFF (entry açırsan).",
                "6) SETUP olduqda 5m təsdiqini gözlə.",
            ]
            for var, text in zip(self.manual_steps_vars, default_steps):
                var.set(text)
            return

        side = best.get("side", "-")
        symbol = best.get("symbol", "-")
        entry = float(best.get("entry", 0.0))
        sl = float(best.get("sl", 0.0))
        tp2 = float(best.get("tp2", 0.0))
        leverage = int(best.get("leverage", 0))
        qty = float(best.get("qty", 0.0))
        status = best.get("status", "-")
        caution = "SETUP" if status == "SETUP" else "OK"

        steps = [
            f"1) {symbol} {side} əməliyyatı üçün giriş planını təsdiqlə ({caution}).",
            f"2) Entry zona: {entry:.6f} — qiymət uyğunlaşmasını yoxla.",
            f"3) Qty: {qty:.6f} | Leverage: {leverage}x (Isolated).",
            f"4) Take Profit (TP2): {tp2:.6f} | Stop Loss: {sl:.6f}.",
            "5) TIF = GTC, Reduce-only = OFF (entry açırsan).",
        ]
        if status == "SETUP":
            steps.append("6) SETUP üçün 5m təsdiqini gözlə (sweep + close).")
        for idx, var in enumerate(self.manual_steps_vars):
            if idx < len(steps):
                var.set(steps[idx])
            else:
                var.set("")

    def _apply_best_to_monitor(self) -> None:
        symbol = self.best_symbol_var.get().strip()
        if symbol and symbol != "-":
            self.monitor_symbol_var.set(symbol)

    def on_start_monitor(self) -> None:
        if self._monitoring:
            return
        self._monitor_event.clear()
        self._monitoring = True
        self.monitor_status_var.set("Monitor işə düşür...")
        self.btn_monitor_start.set_disabled(True)
        self.btn_monitor_stop.set_disabled(False)
        self.position_monitor.start()

    def on_stop_monitor(self) -> None:
        if not self._monitoring:
            return
        self.position_monitor.stop()
        self._monitoring = False
        self.monitor_status_var.set("Monitor dayandırıldı.")
        self.monitor_signal_var.set("-")
        self.btn_monitor_start.set_disabled(False)
        self.btn_monitor_stop.set_disabled(True)

    def _handle_position_update(self, payload: dict) -> None:
        """Handle deep analysis update from PositionMonitor."""
        health = payload.get("health", {})
        score = health.get("score", 50.0)
        reason = health.get("reason", "")
        
        # Update Status Bar
        symbol = payload.get("symbol", "-")
        self.monitor_status_var.set(f"Monitoring: {symbol}")
        self.monitor_signal_var.set(f"Health: {score:.1f}% ({reason})")
        
        # Update Dashboard
        analysis_result = payload.get("deep_analysis")
        if analysis_result:
            # Reconstruct details dict if needed or pass directly if compatible
            # The dashboard expects a dict with specific keys. 
            # We can use update_from_analysis_result which expects a dict representation.
            # Ideally deep_analyzer should have a to_dict method or we access attributes.
            # Assuming payload['deep_analysis'] is the object, we need to convert it.
            # But wait, deep_dashboard.py: update_from_analysis_result takes a Dict.
            # deep_analyzer.DeepAnalysisResult is a dataclass.
            
            # Let's create a helper to convert DeepAnalysisResult to dict for the dashboard
            res = analysis_result
            
            # Prepare dashboard data
            dash_data = {
                "confidence": res.confidence,
                "quality_score": res.quality_score,
                "signal": res.signal,
                "side": res.side,
                "indicator_score": res.indicator_score,
                "structure_score": res.structure_score,
                "volume_score": res.volume_score,
                "mtf_score": res.mtf_score,
                "market_data_score": res.market_data_score,
                "entry": res.entry,
                "sl": res.stop_loss,
                "tp1": res.take_profit_1,
                "tp2": res.take_profit_2,
                "reasons": res.reasons,
                "warnings": res.warnings,
                "indicators": res.indicators.indicators if res.indicators else {}
            }
            
            # Sync to separate window if open
            if self.dashboard_window and self.dashboard_window.winfo_exists():
                self.dashboard_window.update_from_result(dash_data)
                
                # Update metrics if available
                metrics = payload.get("metrics")
                if metrics:
                    self.dashboard_window.dashboard.update_metrics(metrics)
                
            # Update Compact Status
            self.compact_status_var.set(f"Signal: {res.signal} | Conf: {res.confidence:.1f}%")
            
            # We also have NEWS in the payload!
            # payload['news'] is a list of news items.
            # The dashboard doesn't have a news section yet? 
            # Wait, I added it to gui_tk.py separately as `self.news_text`, 
            # but I didn't integrate it into `DeepAnalysisDashboard`.
            # However, the user asked for visual effects.
            # I can update `self._update_news_display` with the specific coin news?
            # Yes, let's update the specific news section I created earlier with this context.
            if payload.get("news"):
                self._update_news_display(payload["news"])
                self.news_status_var.set(f"Showing news for {symbol}")

    # Old monitor methods removed (replaced by PositionMonitor)

    def _create_news_section(self):
        """Creates the News & Sentiment Analysis section in the UI."""
        news_frame = ttk.LabelFrame(
            self.page_frame,
            text="📰 Crypto News & Sentiment Analysis",
            style="Card.TLabelframe",
            padding=10
        )
        news_frame.pack(fill="x", padx=20, pady=10)
        
        controls_frame = ttk.Frame(news_frame, style="Card.TFrame")
        controls_frame.pack(fill="x", pady=(0, 10))
        
        ModernButton(
            controls_frame,
            text="🔄 Refresh News",
            command=self.on_refresh_news,
            width=160
        ).pack(side="left")
        
        self.news_status_var = tk.StringVar(value="Ready to fetch news.")
        ttk.Label(
            controls_frame,
            textvariable=self.news_status_var,
            style="Secondary.TLabel"
        ).pack(side="left", padx=10)
        
        # Scrollable Text area for news
        self.news_text = tk.Text(
            news_frame,
            height=12,
            wrap="word",
            bg=ModernStyle.BG_LIGHT,
            fg=ModernStyle.TEXT_PRIMARY,
            insertbackground=ModernStyle.ACCENT_PRIMARY,
            state="disabled",
            font=ModernStyle.FONT_MONO,
            relief="flat",
            padx=10,
            pady=10
        )
        self.news_text.pack(fill="x")
        
        # Configure tags for sentiment colors
        self.news_text.tag_config("POSITIVE", foreground="#00ff88")
        self.news_text.tag_config("NEGATIVE", foreground="#ff4444")
        self.news_text.tag_config("NEUTRAL", foreground="#8892b0")
        self.news_text.tag_config("TITLE", font=("Segoe UI Bold", 11))
        
    def on_refresh_news(self):
        """Fetches and displays latest news."""
        def worker():
            try:
                self.news_status_var.set("Fetching news from CoinGecko...")
                news_items = self.news_analyzer.get_latest_news(limit=10)
                
                self.root.after(0, lambda: self._update_news_display(news_items))
                self.root.after(0, lambda: self.news_status_var.set(f"Loaded {len(news_items)} updates."))
            except Exception as e:
                self.root.after(0, lambda: self.news_status_var.set(f"Error: {str(e)}"))
        
        threading.Thread(target=worker, daemon=True).start()
        
    def _update_news_display(self, news_items):
        """Updates the news text widget on the main thread."""
        self.news_text.configure(state="normal")
        self.news_text.delete("1.0", "end")
        
        if not news_items:
            self.news_text.insert("end", "No news found or connection failed.")
            self.news_text.configure(state="disabled")
            return
            
        for item in news_items:
            sentiment = item['sentiment']
            label = sentiment['label']
            
            # Insert Title
            self.news_text.insert("end", f"[{label}] {item['title']}\n", label)
            
            # Insert Description
            self.news_text.insert("end", f"{item['description']}\n")
            
            # Insert Metadata
            self.news_text.insert("end", f"Date: {item['published_at']} | Source: {item['author']}\n")
            self.news_text.insert("end", "-" * 50 + "\n\n")
            
        self.news_text.configure(state="disabled")


    def on_open_dashboard(self):
        """Open specialized Deep Analysis Dashboard window."""
        if self.dashboard_window is None or not self.dashboard_window.winfo_exists():
            self.dashboard_window = DeepAnalysisWindow(self.root)
            self.dashboard_window.protocol("WM_DELETE_WINDOW", self._on_dashboard_close)
        self.dashboard_window.lift()
        
    def _on_dashboard_close(self):
        if self.dashboard_window:
            self.dashboard_window.destroy()
            self.dashboard_window = None
