"""
Professional Deep Analysis Visualization Widgets

This module provides advanced visualization components for displaying
deep analysis results in a professional, animated interface.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional, Tuple
import threading


class DeepStyle:
    """Extended style configuration for deep analysis visualizations."""
    
    # Core Colors
    BG_DARK = "#121826"
    BG_MEDIUM = "#1f2736"
    BG_LIGHT = "#2d3850"
    BG_ELEVATED = "#253049"
    BG_PANEL = "#1a2234"
    BG_CARD = "#1e2a3d"
    
    # Accent Colors
    ACCENT_PRIMARY = "#00d4ff"
    ACCENT_SUCCESS = "#00ff88"
    ACCENT_WARNING = "#ffaa00"
    ACCENT_ERROR = "#ff4444"
    ACCENT_INFO = "#6aa7ff"
    ACCENT_PURPLE = "#b28dff"
    ACCENT_GOLD = "#ffd700"
    
    # Gradient Colors
    GRADIENT_BLUE = ["#00d4ff", "#0099cc", "#006699"]
    GRADIENT_GREEN = ["#00ff88", "#00cc6a", "#00994d"]
    GRADIENT_ORANGE = ["#ffaa00", "#cc8800", "#996600"]
    GRADIENT_RED = ["#ff4444", "#cc3333", "#992222"]
    
    # Text Colors
    TEXT_PRIMARY = "#f5f7ff"
    TEXT_SECONDARY = "#c5cbe6"
    TEXT_MUTED = "#8892b0"
    TEXT_BRIGHT = "#ffffff"
    
    # Border & Divider
    BORDER = "#3a475f"
    DIVIDER = "#2a3548"
    
    # Fonts
    FONT_MAIN = ("Segoe UI", 11)
    FONT_HEADER = ("Segoe UI Semibold", 12)
    FONT_TITLE = ("Segoe UI Bold", 14)
    FONT_HERO = ("Segoe UI Bold", 22)
    FONT_SUBTITLE = ("Segoe UI Semibold", 11)
    FONT_MONO = ("Consolas", 10)
    FONT_SMALL = ("Segoe UI", 9)
    FONT_LARGE = ("Segoe UI Bold", 16)


class RadialGauge(tk.Canvas):
    """
    Professional radial gauge for displaying confidence/scores.
    
    Features:
    - Animated needle movement
    - Gradient arc coloring
    - Smooth animations
    - Value display in center
    """
    
    def __init__(self, parent, size: int = 140, label: str = "Score", **kwargs):
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.size = size
        self.label = label
        self.value = 0.0
        self.target_value = 0.0
        self._animating = False
        
        self.bind("<Configure>", lambda e: self._draw())
        self.after(100, self._draw)
    
    def set_value(self, value: float, animate: bool = True) -> None:
        """Set gauge value with optional animation."""
        self.target_value = max(0.0, min(100.0, value))
        if animate and not self._animating:
            self._animating = True
            self._animate()
        elif not animate:
            self.value = self.target_value
            self._draw()
    
    def _animate(self) -> None:
        diff = self.target_value - self.value
        if abs(diff) < 0.5:
            self.value = self.target_value
            self._animating = False
            self._draw()
            return
        
        self.value += diff * 0.15
        self._draw()
        self.after(16, self._animate)
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            w = self.size
        if h <= 1:
            h = self.size
        
        center_x = w // 2
        center_y = h // 2 + 10
        radius = min(w, h) // 2 - 15
        
        # Background circle
        self.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            fill=DeepStyle.BG_LIGHT, outline=DeepStyle.BORDER, width=2
        )
        
        # Draw arc segments (color coded)
        arc_width = 8
        segments = [
            (0, 33, DeepStyle.ACCENT_ERROR),
            (33, 66, DeepStyle.ACCENT_WARNING),
            (66, 100, DeepStyle.ACCENT_SUCCESS),
        ]
        
        for start_pct, end_pct, color in segments:
            start_angle = 180 - (start_pct * 1.8)
            extent = -((end_pct - start_pct) * 1.8)
            
            self.create_arc(
                center_x - radius + arc_width,
                center_y - radius + arc_width,
                center_x + radius - arc_width,
                center_y + radius - arc_width,
                start=start_angle, extent=extent,
                outline=color, width=arc_width, style="arc"
            )
        
        # Draw needle
        needle_angle = math.radians(180 - (self.value * 1.8))
        needle_length = radius - 25
        needle_x = center_x + needle_length * math.cos(needle_angle)
        needle_y = center_y - needle_length * math.sin(needle_angle)
        
        # Needle color based on value
        if self.value >= 66:
            needle_color = DeepStyle.ACCENT_SUCCESS
        elif self.value >= 33:
            needle_color = DeepStyle.ACCENT_WARNING
        else:
            needle_color = DeepStyle.ACCENT_ERROR
        
        self.create_line(
            center_x, center_y, needle_x, needle_y,
            fill=needle_color, width=3, arrow=tk.LAST
        )
        
        # Center circle
        self.create_oval(
            center_x - 8, center_y - 8,
            center_x + 8, center_y + 8,
            fill=DeepStyle.BG_ELEVATED, outline=needle_color, width=2
        )
        
        # Value text
        self.create_text(
            center_x, center_y - 30,
            text=f"{self.value:.1f}%",
            fill=DeepStyle.TEXT_PRIMARY, font=DeepStyle.FONT_LARGE
        )
        
        # Label
        self.create_text(
            center_x, h - 10,
            text=self.label,
            fill=DeepStyle.TEXT_MUTED, font=DeepStyle.FONT_SMALL
        )


class ComponentScoreBar(tk.Canvas):
    """
    Horizontal bar chart for component score visualization.
    
    Features:
    - Multiple component bars
    - Animated value changes
    - Color-coded based on score
    - Labels and values
    """
    
    def __init__(self, parent, height: int = 200, **kwargs):
        super().__init__(
            parent,
            height=height,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.components: List[Dict[str, Any]] = []
        self.target_values: List[float] = []
        self._animating = False
        
        self.bind("<Configure>", lambda e: self._draw())
    
    def set_components(self, components: List[Tuple[str, float, str]]) -> None:
        """
        Set component data.
        
        Args:
            components: List of (name, value, icon) tuples
        """
        self.components = [
            {"name": name, "value": 0.0, "icon": icon}
            for name, _, icon in components
        ]
        self.target_values = [value for _, value, _ in components]
        
        if not self._animating:
            self._animating = True
            self._animate()
    
    def _animate(self) -> None:
        done = True
        for i, comp in enumerate(self.components):
            target = self.target_values[i]
            diff = target - comp["value"]
            if abs(diff) > 0.5:
                done = False
                comp["value"] += diff * 0.12
        
        self._draw()
        
        if not done:
            self.after(16, self._animate)
        else:
            self._animating = False
            for i, comp in enumerate(self.components):
                comp["value"] = self.target_values[i]
            self._draw()
    
    def _get_bar_color(self, value: float) -> str:
        if value >= 70:
            return DeepStyle.ACCENT_SUCCESS
        elif value >= 50:
            return DeepStyle.ACCENT_WARNING
        elif value >= 30:
            return DeepStyle.ACCENT_INFO
        else:
            return DeepStyle.ACCENT_ERROR
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or not self.components:
            return
        
        padding = 15
        bar_height = 24
        bar_gap = 12
        label_width = 100
        value_width = 50
        
        # Title
        self.create_text(
            padding, 12,
            text="Component Scores",
            fill=DeepStyle.TEXT_SECONDARY, font=DeepStyle.FONT_SUBTITLE,
            anchor="w"
        )
        
        start_y = 35
        bar_area_width = w - padding * 2 - label_width - value_width
        
        for i, comp in enumerate(self.components):
            y = start_y + i * (bar_height + bar_gap)
            
            # Icon and label
            self.create_text(
                padding, y + bar_height // 2,
                text=f"{comp['icon']} {comp['name']}",
                fill=DeepStyle.TEXT_PRIMARY, font=DeepStyle.FONT_MAIN,
                anchor="w"
            )
            
            # Bar background
            bar_x = padding + label_width
            self.create_rectangle(
                bar_x, y + 2,
                bar_x + bar_area_width, y + bar_height - 2,
                fill=DeepStyle.BG_LIGHT, outline=""
            )
            
            # Bar fill
            fill_width = (comp["value"] / 100.0) * bar_area_width
            bar_color = self._get_bar_color(comp["value"])
            
            if fill_width > 0:
                self.create_rectangle(
                    bar_x, y + 2,
                    bar_x + fill_width, y + bar_height - 2,
                    fill=bar_color, outline=""
                )
            
            # Value text
            self.create_text(
                w - padding, y + bar_height // 2,
                text=f"{comp['value']:.0f}%",
                fill=bar_color, font=DeepStyle.FONT_SUBTITLE,
                anchor="e"
            )


class SignalPanel(tk.Canvas):
    """
    Professional signal display panel with animated indicators.
    
    Features:
    - Buy/Sell signal with strength
    - Animated pulse effect
    - Color-coded presentation
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=100,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.signal = "NEUTRAL"
        self.strength = 0.0
        self.side = "NONE"
        self._pulse_phase = 0
        self._pulsing = False
        
        self.bind("<Configure>", lambda e: self._draw())
    
    def set_signal(self, signal: str, strength: float, side: str) -> None:
        """Set signal data."""
        self.signal = signal
        self.strength = strength
        self.side = side
        
        if signal in ("STRONG_BUY", "STRONG_SELL"):
            self._start_pulse()
        else:
            self._pulsing = False
        
        self._draw()
    
    def _start_pulse(self):
        if not self._pulsing:
            self._pulsing = True
            self._pulse()
    
    def _pulse(self):
        if not self._pulsing:
            return
        self._pulse_phase = (self._pulse_phase + 1) % 20
        self._draw()
        self.after(50, self._pulse)
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            return
        
        # Determine colors based on signal
        if self.side == "LONG":
            main_color = DeepStyle.ACCENT_SUCCESS
            bg_color = "#1a3328"
        elif self.side == "SHORT":
            main_color = DeepStyle.ACCENT_ERROR
            bg_color = "#331a1a"
        else:
            main_color = DeepStyle.TEXT_MUTED
            bg_color = DeepStyle.BG_LIGHT
        
        # Background panel
        self.create_rectangle(
            10, 10, w - 10, h - 10,
            fill=bg_color, outline=main_color, width=2
        )
        
        # Pulse effect for strong signals
        if self._pulsing:
            pulse_alpha = abs(math.sin(self._pulse_phase * 0.3)) * 0.3
            # Simulate glow with rectangle
            glow_expand = int(pulse_alpha * 5)
            self.create_rectangle(
                10 - glow_expand, 10 - glow_expand,
                w - 10 + glow_expand, h - 10 + glow_expand,
                outline=main_color, width=3, dash=(4, 4)
            )
        
        # Signal arrow
        arrow_size = 30
        center_x = 60
        center_y = h // 2
        
        if self.side == "LONG":
            # Up arrow
            points = [
                center_x, center_y - arrow_size // 2,
                center_x - arrow_size // 2, center_y + arrow_size // 2,
                center_x + arrow_size // 2, center_y + arrow_size // 2,
            ]
            self.create_polygon(points, fill=main_color, outline="")
        elif self.side == "SHORT":
            # Down arrow
            points = [
                center_x, center_y + arrow_size // 2,
                center_x - arrow_size // 2, center_y - arrow_size // 2,
                center_x + arrow_size // 2, center_y - arrow_size // 2,
            ]
            self.create_polygon(points, fill=main_color, outline="")
        else:
            # Neutral circle
            self.create_oval(
                center_x - 15, center_y - 15,
                center_x + 15, center_y + 15,
                fill=main_color, outline=""
            )
        
        # Signal text
        self.create_text(
            130, center_y - 12,
            text=self.signal.replace("_", " "),
            fill=main_color, font=DeepStyle.FONT_LARGE,
            anchor="w"
        )
        
        # Side and strength
        self.create_text(
            130, center_y + 15,
            text=f"{self.side} • Strength: {self.strength:.0f}%",
            fill=DeepStyle.TEXT_SECONDARY, font=DeepStyle.FONT_MAIN,
            anchor="w"
        )


class IndicatorMatrix(tk.Canvas):
    """
    Matrix display of all technical indicators.
    
    Features:
    - Grid of indicator cards
    - Color-coded by signal
    - Animated appearance
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=180,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.indicators: List[Dict[str, Any]] = []
        self._animation_progress = 0.0
        self._animating = False
        
        self.bind("<Configure>", lambda e: self._draw())
    
    def set_indicators(self, indicators: List[Dict[str, Any]]) -> None:
        """
        Set indicator data.
        
        Args:
            indicators: List of {"name": str, "signal": str, "strength": float}
        """
        self.indicators = indicators
        self._animation_progress = 0.0
        if not self._animating:
            self._animating = True
            self._animate()
    
    def _animate(self):
        if self._animation_progress >= 1.0:
            self._animation_progress = 1.0
            self._animating = False
            self._draw()
            return
        
        self._animation_progress += 0.08
        self._draw()
        self.after(16, self._animate)
    
    def _get_signal_color(self, signal: str) -> str:
        signal = signal.upper()
        if signal == "BULLISH":
            return DeepStyle.ACCENT_SUCCESS
        elif signal == "BEARISH":
            return DeepStyle.ACCENT_ERROR
        else:
            return DeepStyle.TEXT_MUTED
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or not self.indicators:
            return
        
        # Title
        self.create_text(
            15, 15,
            text="📊 Technical Indicators",
            fill=DeepStyle.TEXT_SECONDARY, font=DeepStyle.FONT_SUBTITLE,
            anchor="w"
        )
        
        # Calculate grid
        padding = 15
        card_width = 110
        card_height = 48
        gap = 8
        start_y = 38
        
        cols = max(1, (w - padding * 2 + gap) // (card_width + gap))
        
        visible_count = int(len(self.indicators) * self._animation_progress)
        
        for i, indicator in enumerate(self.indicators[:visible_count]):
            row = i // cols
            col = i % cols
            
            x = padding + col * (card_width + gap)
            y = start_y + row * (card_height + gap)
            
            signal_color = self._get_signal_color(indicator.get("signal", "NEUTRAL"))
            
            # Card background
            self.create_rectangle(
                x, y, x + card_width, y + card_height,
                fill=DeepStyle.BG_LIGHT, outline=signal_color, width=1
            )
            
            # Indicator name
            self.create_text(
                x + 8, y + 14,
                text=indicator.get("name", "").upper()[:10],
                fill=DeepStyle.TEXT_PRIMARY, font=DeepStyle.FONT_SMALL,
                anchor="w"
            )
            
            # Signal dot
            self.create_oval(
                x + card_width - 18, y + 8,
                x + card_width - 8, y + 18,
                fill=signal_color, outline=""
            )
            
            # Strength bar
            bar_width = card_width - 16
            strength = indicator.get("strength", 0.5) * 100
            fill_width = (strength / 100.0) * bar_width
            
            self.create_rectangle(
                x + 8, y + 30,
                x + 8 + bar_width, y + 38,
                fill=DeepStyle.BG_DARK, outline=""
            )
            
            if fill_width > 0:
                self.create_rectangle(
                    x + 8, y + 30,
                    x + 8 + fill_width, y + 38,
                    fill=signal_color, outline=""
                )


class TradeLevelsPanel(tk.Canvas):
    """
    Professional trade levels visualization.
    
    Features:
    - Entry, SL, TP display
    - Visual price ladder
    - Risk/Reward visualization
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=180,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.entry = 0.0
        self.sl = 0.0
        self.tp1 = 0.0
        self.tp2 = 0.0
        self.side = "LONG"
        self.current_price = 0.0
        
        self.bind("<Configure>", lambda e: self._draw())
    
    def set_levels(
        self,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        side: str,
        current_price: float = 0.0
    ) -> None:
        """Set trade levels."""
        self.entry = entry
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.side = side
        self.current_price = current_price or entry
        self._draw()
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1 or self.entry <= 0:
            return
        
        # Title
        self.create_text(
            15, 15,
            text="🎯 Trade Setup",
            fill=DeepStyle.TEXT_SECONDARY, font=DeepStyle.FONT_SUBTITLE,
            anchor="w"
        )
        
        # Calculate price range
        all_prices = [p for p in [self.entry, self.sl, self.tp1, self.tp2] if p > 0]
        if not all_prices:
            return
        
        min_price = min(all_prices) * 0.998
        max_price = max(all_prices) * 1.002
        price_range = max_price - min_price
        
        # Ladder area
        ladder_left = 120
        ladder_right = w - 100
        ladder_top = 45
        ladder_bottom = h - 20
        ladder_height = ladder_bottom - ladder_top
        
        # Background
        self.create_rectangle(
            ladder_left, ladder_top,
            ladder_right, ladder_bottom,
            fill=DeepStyle.BG_LIGHT, outline=DeepStyle.BORDER
        )
        
        # Draw price levels
        levels = [
            ("TP2", self.tp2, DeepStyle.ACCENT_SUCCESS, "◆"),
            ("TP1", self.tp1, DeepStyle.ACCENT_SUCCESS, "◇"),
            ("Entry", self.entry, DeepStyle.ACCENT_PRIMARY, "►"),
            ("SL", self.sl, DeepStyle.ACCENT_ERROR, "■"),
        ]
        
        for name, price, color, symbol in levels:
            if price <= 0:
                continue
            
            y = ladder_top + (1 - (price - min_price) / price_range) * ladder_height
            y = max(ladder_top + 5, min(ladder_bottom - 5, y))
            
            # Level line
            self.create_line(
                ladder_left, y, ladder_right, y,
                fill=color, width=2, dash=(4, 2)
            )
            
            # Label on left
            self.create_text(
                ladder_left - 10, y,
                text=f"{symbol} {name}",
                fill=color, font=DeepStyle.FONT_SMALL,
                anchor="e"
            )
            
            # Price on right
            self.create_text(
                ladder_right + 10, y,
                text=f"${price:,.4f}",
                fill=color, font=DeepStyle.FONT_MONO,
                anchor="w"
            )
        
        # Current price marker
        if self.current_price > 0:
            y = ladder_top + (1 - (self.current_price - min_price) / price_range) * ladder_height
            y = max(ladder_top + 5, min(ladder_bottom - 5, y))
            
            self.create_polygon(
                ladder_left - 5, y,
                ladder_left + 5, y - 5,
                ladder_left + 5, y + 5,
                fill=DeepStyle.TEXT_PRIMARY
            )


class WorkflowProgress(tk.Canvas):
    """
    Professional horizontal workflow progress with stages.
    
    Features:
    - Stage indicators
    - Progress animation
    - Active stage highlighting
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=80,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.stages = [
            {"name": "Data", "icon": "📊"},
            {"name": "Indicators", "icon": "📈"},
            {"name": "Structure", "icon": "🏗️"},
            {"name": "Volume", "icon": "📊"},
            {"name": "MTF", "icon": "🔄"},
            {"name": "Score", "icon": "⭐"},
        ]
        self.active_stage = -1
        self.completed_stages = set()
        self._pulse_phase = 0
        self._pulsing = False
        
        self.bind("<Configure>", lambda e: self._draw())
    
    def set_stage(self, stage_index: int) -> None:
        """Set active stage and mark previous as complete."""
        for i in range(stage_index):
            self.completed_stages.add(i)
        self.active_stage = stage_index
        
        if not self._pulsing:
            self._pulsing = True
            self._pulse()
        
        self._draw()
    
    def reset(self) -> None:
        """Reset workflow."""
        self.active_stage = -1
        self.completed_stages.clear()
        self._pulsing = False
        self._draw()
    
    def complete_all(self) -> None:
        """Mark all stages complete."""
        self.completed_stages = set(range(len(self.stages)))
        self.active_stage = -1
        self._pulsing = False
        self._draw()
    
    def _pulse(self):
        if not self._pulsing or self.active_stage < 0:
            return
        self._pulse_phase = (self._pulse_phase + 1) % 10
        self._draw()
        self.after(100, self._pulse)
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            return
        
        padding = 30
        stage_count = len(self.stages)
        available_width = w - padding * 2
        stage_spacing = available_width / (stage_count - 1) if stage_count > 1 else 0
        
        node_radius = 18
        center_y = h // 2
        
        # Draw connecting lines first
        for i in range(stage_count - 1):
            x1 = padding + i * stage_spacing + node_radius
            x2 = padding + (i + 1) * stage_spacing - node_radius
            
            line_color = DeepStyle.ACCENT_SUCCESS if i in self.completed_stages else DeepStyle.BG_LIGHT
            self.create_line(
                x1, center_y, x2, center_y,
                fill=line_color, width=3
            )
        
        # Draw stage nodes
        for i, stage in enumerate(self.stages):
            x = padding + i * stage_spacing
            
            is_active = i == self.active_stage
            is_complete = i in self.completed_stages
            
            # Determine colors
            if is_complete:
                fill_color = DeepStyle.ACCENT_SUCCESS
                border_color = DeepStyle.ACCENT_SUCCESS
            elif is_active:
                fill_color = DeepStyle.ACCENT_PRIMARY
                border_color = DeepStyle.ACCENT_PRIMARY
            else:
                fill_color = DeepStyle.BG_LIGHT
                border_color = DeepStyle.BORDER
            
            # Pulse effect for active
            if is_active:
                glow = abs(math.sin(self._pulse_phase * 0.6)) * 4
                self.create_oval(
                    x - node_radius - glow, center_y - node_radius - glow,
                    x + node_radius + glow, center_y + node_radius + glow,
                    outline=DeepStyle.ACCENT_PRIMARY, width=2, dash=(2, 2)
                )
            
            # Node circle
            self.create_oval(
                x - node_radius, center_y - node_radius,
                x + node_radius, center_y + node_radius,
                fill=fill_color, outline=border_color, width=2
            )
            
            # Icon or check
            if is_complete:
                self.create_text(
                    x, center_y,
                    text="✓", fill=DeepStyle.BG_DARK, font=DeepStyle.FONT_HEADER
                )
            else:
                self.create_text(
                    x, center_y,
                    text=stage["icon"], font=DeepStyle.FONT_MAIN
                )
            
            # Label below
            self.create_text(
                x, center_y + node_radius + 14,
                text=stage["name"],
                fill=DeepStyle.TEXT_SECONDARY if not is_active else DeepStyle.TEXT_PRIMARY,
                font=DeepStyle.FONT_SMALL
            )


class ThreadingMetrics(tk.Canvas):
    """
    Real-time threading performance metrics display.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=60,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time_ms": 0.0,
            "active_threads": 0,
            "peak_threads": 0,
        }
        
        self.bind("<Configure>", lambda e: self._draw())
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update displayed metrics."""
        self.metrics.update(metrics)
        self._draw()
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            return
        
        # Background
        self.create_rectangle(
            10, 5, w - 10, h - 5,
            fill=DeepStyle.BG_LIGHT, outline=DeepStyle.BORDER
        )
        
        # Title
        self.create_text(
            25, h // 2,
            text="⚡",
            font=DeepStyle.FONT_HEADER
        )
        
        # Metrics display
        metrics_text = [
            f"Tasks: {self.metrics['completed_tasks']}/{self.metrics['total_tasks']}",
            f"Failed: {self.metrics['failed_tasks']}",
            f"Avg: {self.metrics['avg_execution_time_ms']:.0f}ms",
            f"Threads: {self.metrics['active_threads']}/{self.metrics['peak_threads']}",
        ]
        
        x = 55
        for text in metrics_text:
            self.create_text(
                x, h // 2,
                text=text,
                fill=DeepStyle.TEXT_SECONDARY, font=DeepStyle.FONT_SMALL,
                anchor="w"
            )
            x += 130


class ReasonsList(tk.Canvas):
    """
    Scrollable list of analysis reasons and warnings.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=150,
            bg=DeepStyle.BG_MEDIUM,
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.reasons: List[str] = []
        self.warnings: List[str] = []
        
        self.bind("<Configure>", lambda e: self._draw())
    
    def set_data(self, reasons: List[str], warnings: List[str]) -> None:
        """Set reasons and warnings."""
        self.reasons = reasons[:6]  # Limit to 6
        self.warnings = warnings[:4]  # Limit to 4
        self._draw()
    
    def _draw(self):
        self.delete("all")
        
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            return
        
        y = 15
        line_height = 18
        
        # Reasons header
        if self.reasons:
            self.create_text(
                15, y,
                text="✅ Analysis Factors",
                fill=DeepStyle.ACCENT_SUCCESS, font=DeepStyle.FONT_SUBTITLE,
                anchor="w"
            )
            y += line_height + 5
            
            for reason in self.reasons:
                # Truncate long text
                display_text = reason[:60] + "..." if len(reason) > 60 else reason
                self.create_text(
                    25, y,
                    text=f"• {display_text}",
                    fill=DeepStyle.TEXT_SECONDARY, font=DeepStyle.FONT_SMALL,
                    anchor="w"
                )
                y += line_height
        
        # Warnings header
        if self.warnings:
            y += 8
            self.create_text(
                15, y,
                text="⚠️ Warnings",
                fill=DeepStyle.ACCENT_WARNING, font=DeepStyle.FONT_SUBTITLE,
                anchor="w"
            )
            y += line_height + 5
            
            for warning in self.warnings:
                display_text = warning[:60] + "..." if len(warning) > 60 else warning
                self.create_text(
                    25, y,
                    text=f"• {display_text}",
                    fill=DeepStyle.ACCENT_WARNING, font=DeepStyle.FONT_SMALL,
                    anchor="w"
                )
                y += line_height
