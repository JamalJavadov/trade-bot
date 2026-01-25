"""
Professional Deep Analysis Dashboard

This module provides a complete dashboard UI that integrates all
deep analysis visualization widgets into a cohesive, professional interface.
"""
from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Callable

from .deep_visualizations import (
    DeepStyle,
    RadialGauge,
    ComponentScoreBar,
    SignalPanel,
    IndicatorMatrix,
    TradeLevelsPanel,
    WorkflowProgress,
    ThreadingMetrics,
    ReasonsList,
)


class DeepAnalysisDashboard(ttk.Frame):
    """
    Complete deep analysis dashboard with all visualizations.
    
    This dashboard provides:
    - Real-time workflow progress
    - Confidence gauges
    - Component score breakdown
    - Signal display
    - Indicator matrix
    - Trade levels visualization
    - Reasons and warnings list
    - Threading metrics
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.configure(style="Dark.TFrame")
        self._setup_styles()
        self._create_widgets()
    
    def _setup_styles(self):
        """Configure TTK styles for the dashboard."""
        style = ttk.Style()
        
        style.configure(
            "Deep.TFrame",
            background=DeepStyle.BG_DARK
        )
        
        style.configure(
            "DeepCard.TFrame",
            background=DeepStyle.BG_MEDIUM
        )
        
        style.configure(
            "DeepLabel.TLabel",
            background=DeepStyle.BG_DARK,
            foreground=DeepStyle.TEXT_PRIMARY,
            font=DeepStyle.FONT_MAIN
        )
        
        style.configure(
            "DeepHeader.TLabel",
            background=DeepStyle.BG_DARK,
            foreground=DeepStyle.ACCENT_PRIMARY,
            font=DeepStyle.FONT_HERO
        )
        
        style.configure(
            "DeepCard.TLabelframe",
            background=DeepStyle.BG_MEDIUM,
            foreground=DeepStyle.TEXT_PRIMARY,
            borderwidth=1
        )
        
        style.configure(
            "DeepCard.TLabelframe.Label",
            background=DeepStyle.BG_MEDIUM,
            foreground=DeepStyle.ACCENT_PRIMARY,
            font=DeepStyle.FONT_HEADER
        )
    
    def _create_widgets(self):
        """Create all dashboard widgets."""
        # Main container with scroll
        self._create_scroll_container()
        
        # Header
        self._create_header()
        
        # Workflow progress
        self._create_workflow_section()
        
        # Main content in two columns
        self._create_main_content()
        
        # Threading metrics footer
        self._create_metrics_footer()
    
    def _create_scroll_container(self):
        """Create scrollable container."""
        self.canvas = tk.Canvas(
            self,
            bg=DeepStyle.BG_DARK,
            highlightthickness=0,
            bd=0
        )
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.content_frame = ttk.Frame(self.canvas, style="Deep.TFrame")
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        
        def on_frame_configure(e):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        def on_canvas_configure(e):
            self.canvas.itemconfigure(self.canvas_window, width=e.width)
        
        self.content_frame.bind("<Configure>", on_frame_configure)
        self.canvas.bind("<Configure>", on_canvas_configure)
        
        # Mouse wheel binding
        def on_mousewheel(event):
            if hasattr(event, "delta") and event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.canvas.bind_all("<MouseWheel>", on_mousewheel)
    
    def _create_header(self):
        """Create dashboard header."""
        header = ttk.Frame(self.content_frame, style="Deep.TFrame")
        header.pack(fill="x", padx=20, pady=(15, 10))
        
        ttk.Label(
            header,
            text="🔬 Deep Analysis Dashboard",
            style="DeepHeader.TLabel"
        ).pack(side="left")
        
        # Status indicator
        self.status_label = ttk.Label(
            header,
            text="Ready",
            style="DeepLabel.TLabel"
        )
        self.status_label.pack(side="right", padx=10)
    
    def _create_workflow_section(self):
        """Create workflow progress section."""
        workflow_card = ttk.LabelFrame(
            self.content_frame,
            text="📊 Analysis Workflow",
            style="DeepCard.TLabelframe",
            padding=10
        )
        workflow_card.pack(fill="x", padx=20, pady=5)
        
        self.workflow_progress = WorkflowProgress(workflow_card)
        self.workflow_progress.pack(fill="x", expand=True)
    
    def _create_main_content(self):
        """Create main content area with two columns."""
        main_container = ttk.Frame(self.content_frame, style="Deep.TFrame")
        main_container.pack(fill="both", expand=True, padx=20, pady=5)
        
        # Configure grid
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        
        # Left column
        left_col = ttk.Frame(main_container, style="Deep.TFrame")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self._create_gauges_section(left_col)
        self._create_signal_section(left_col)
        self._create_trade_levels_section(left_col)
        
        # Right column
        right_col = ttk.Frame(main_container, style="Deep.TFrame")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        self._create_components_section(right_col)
        self._create_indicators_section(right_col)
        self._create_reasons_section(right_col)
    
    def _create_gauges_section(self, parent):
        """Create confidence gauges section."""
        gauges_card = ttk.LabelFrame(
            parent,
            text="📈 Confidence Scores",
            style="DeepCard.TLabelframe",
            padding=10
        )
        gauges_card.pack(fill="x", pady=5)
        
        gauges_container = ttk.Frame(gauges_card, style="DeepCard.TFrame")
        gauges_container.pack(fill="x")
        
        # Create gauges
        self.confidence_gauge = RadialGauge(
            gauges_container, size=140, label="Confidence"
        )
        self.confidence_gauge.pack(side="left", padx=10, pady=10)
        
        self.quality_gauge = RadialGauge(
            gauges_container, size=140, label="Quality"
        )
        self.quality_gauge.pack(side="left", padx=10, pady=10)
    
    def _create_signal_section(self, parent):
        """Create signal display section."""
        signal_card = ttk.LabelFrame(
            parent,
            text="🎯 Trade Signal",
            style="DeepCard.TLabelframe",
            padding=10
        )
        signal_card.pack(fill="x", pady=5)
        
        self.signal_panel = SignalPanel(signal_card)
        self.signal_panel.pack(fill="x", expand=True)
    
    def _create_trade_levels_section(self, parent):
        """Create trade levels visualization."""
        levels_card = ttk.LabelFrame(
            parent,
            text="📍 Trade Levels",
            style="DeepCard.TLabelframe",
            padding=10
        )
        levels_card.pack(fill="x", pady=5)
        
        self.trade_levels = TradeLevelsPanel(levels_card)
        self.trade_levels.pack(fill="x", expand=True)
    
    def _create_components_section(self, parent):
        """Create component scores section."""
        components_card = ttk.LabelFrame(
            parent,
            text="🧩 Component Analysis",
            style="DeepCard.TLabelframe",
            padding=10
        )
        components_card.pack(fill="x", pady=5)
        
        self.component_scores = ComponentScoreBar(components_card, height=200)
        self.component_scores.pack(fill="x", expand=True)
    
    def _create_indicators_section(self, parent):
        """Create indicators matrix section."""
        indicators_card = ttk.LabelFrame(
            parent,
            text="📊 Technical Indicators",
            style="DeepCard.TLabelframe",
            padding=10
        )
        indicators_card.pack(fill="x", pady=5)
        
        self.indicator_matrix = IndicatorMatrix(indicators_card)
        self.indicator_matrix.pack(fill="x", expand=True)
    
    def _create_reasons_section(self, parent):
        """Create reasons and warnings section."""
        reasons_card = ttk.LabelFrame(
            parent,
            text="📝 Analysis Reasoning",
            style="DeepCard.TLabelframe",
            padding=10
        )
        reasons_card.pack(fill="x", pady=5)
        
        self.reasons_list = ReasonsList(reasons_card)
        self.reasons_list.pack(fill="x", expand=True)
    
    def _create_metrics_footer(self):
        """Create threading metrics footer."""
        metrics_card = ttk.LabelFrame(
            self.content_frame,
            text="⚡ Performance Metrics",
            style="DeepCard.TLabelframe",
            padding=5
        )
        metrics_card.pack(fill="x", padx=20, pady=(5, 15))
        
        self.threading_metrics = ThreadingMetrics(metrics_card)
        self.threading_metrics.pack(fill="x", expand=True)
    
    # ==========================================================================
    # Public API for updating dashboard
    # ==========================================================================
    
    def set_status(self, status: str) -> None:
        """Update status label."""
        self.status_label.configure(text=status)
    
    def set_workflow_stage(self, stage_index: int) -> None:
        """Set active workflow stage."""
        self.workflow_progress.set_stage(stage_index)
    
    def reset_workflow(self) -> None:
        """Reset workflow progress."""
        self.workflow_progress.reset()
    
    def complete_workflow(self) -> None:
        """Mark workflow as complete."""
        self.workflow_progress.complete_all()
    
    def update_confidence(self, confidence: float, quality: float) -> None:
        """Update confidence and quality gauges."""
        self.confidence_gauge.set_value(confidence)
        self.quality_gauge.set_value(quality)
    
    def update_signal(self, signal: str, strength: float, side: str) -> None:
        """Update signal display."""
        self.signal_panel.set_signal(signal, strength, side)
    
    def update_components(self, components: List[tuple]) -> None:
        """
        Update component scores.
        
        Args:
            components: List of (name, value, icon) tuples
        """
        self.component_scores.set_components(components)
    
    def update_indicators(self, indicators: List[Dict[str, Any]]) -> None:
        """Update indicator matrix."""
        self.indicator_matrix.set_indicators(indicators)
    
    def update_trade_levels(
        self,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        side: str,
        current_price: float = 0.0
    ) -> None:
        """Update trade levels visualization."""
        self.trade_levels.set_levels(entry, sl, tp1, tp2, side, current_price)
    
    def update_reasons(self, reasons: List[str], warnings: List[str]) -> None:
        """Update reasons and warnings list."""
        self.reasons_list.set_data(reasons, warnings)
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update threading metrics display."""
        self.threading_metrics.update_metrics(metrics)
    
    def update_from_analysis_result(self, result: Dict[str, Any]) -> None:
        """
        Update entire dashboard from analysis result.
        
        Args:
            result: Deep analysis result dictionary
        """
        # Update confidence
        self.update_confidence(
            result.get("confidence", 0.0),
            result.get("quality_score", 0.0)
        )
        
        # Update signal
        self.update_signal(
            result.get("signal", "NEUTRAL"),
            result.get("confidence", 0.0),
            result.get("side", "NONE")
        )
        
        # Update components
        components = [
            ("Indicators", result.get("indicator_score", 0.0) * 100, "📈"),
            ("Structure", result.get("structure_score", 0.0) * 100, "🏗️"),
            ("Volume", result.get("volume_score", 0.0) * 100, "📊"),
            ("MTF", result.get("mtf_score", 0.0) * 100, "🔄"),
            ("Market", result.get("market_data_score", 0.0) * 100, "📡"),
        ]
        self.update_components(components)
        
        # Update trade levels
        self.update_trade_levels(
            result.get("entry", 0.0),
            result.get("sl", 0.0),
            result.get("tp1", 0.0),
            result.get("tp2", 0.0),
            result.get("side", "LONG"),
            result.get("entry", 0.0)
        )
        
        # Update indicators (if available)
        if "indicators" in result and result["indicators"]:
            indicator_data = []
            for name, ind in result["indicators"].items():
                if hasattr(ind, "signal") and hasattr(ind, "strength"):
                    indicator_data.append({
                        "name": name,
                        "signal": ind.signal,
                        "strength": ind.strength
                    })
            if indicator_data:
                self.update_indicators(indicator_data)
        
        # Update reasons
        self.update_reasons(
            result.get("reasons", []),
            result.get("warnings", [])
        )
        
        # Complete workflow
        self.complete_workflow()
        self.set_status("Analysis Complete")


class DeepAnalysisWindow(tk.Toplevel):
    """
    Standalone window for deep analysis dashboard.
    
    Can be opened from the main GUI to show detailed analysis results.
    """
    
    def __init__(self, parent, title: str = "Deep Analysis Dashboard"):
        super().__init__(parent)
        
        self.title(title)
        self.geometry("900x800")
        self.minsize(800, 600)
        self.configure(bg=DeepStyle.BG_DARK)
        
        # Create dashboard
        self.dashboard = DeepAnalysisDashboard(self)
        self.dashboard.pack(fill="both", expand=True)
    
    def update_from_result(self, result: Dict[str, Any]) -> None:
        """Update dashboard with analysis result."""
        self.dashboard.update_from_analysis_result(result)
