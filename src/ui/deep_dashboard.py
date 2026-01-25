"""
Professional Deep Analysis Dashboard

This module provides a complete dashboard UI that integrates all
deep analysis visualization widgets into a cohesive, professional interface.
"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional

from ..core import binance_data
from ..core.planner import load_settings

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
        
        self._create_binance_section(right_col)
        self._create_components_section(right_col)
        self._create_indicators_section(right_col)
        self._create_reasons_section(right_col)

    def _create_binance_section(self, parent):
        """Create live Binance feed section."""
        binance_card = ttk.LabelFrame(
            parent,
            text="📡 Binance Live Feed (1s)",
            style="DeepCard.TLabelframe",
            padding=10
        )
        binance_card.pack(fill="x", pady=5)

        self.binance_status_var = tk.StringVar(value="Waiting for updates...")
        self.binance_symbol_var = tk.StringVar(value="-")
        self.binance_updates_var = tk.StringVar(value="0")
        self.binance_last_update_var = tk.StringVar(value="-")
        self.binance_server_time_var = tk.StringVar(value="-")
        self.binance_latency_var = tk.StringVar(value="-")
        self.binance_mark_price_var = tk.StringVar(value="-")
        self.binance_funding_var = tk.StringVar(value="-")
        self.binance_oi_var = tk.StringVar(value="-")
        self.binance_order_pressure_var = tk.StringVar(value="-")
        self.binance_taker_pressure_var = tk.StringVar(value="-")
        self.binance_ls_ratio_var = tk.StringVar(value="-")
        self.binance_sentiment_var = tk.StringVar(value="-")

        rows = [
            ("Status", self.binance_status_var),
            ("Symbol", self.binance_symbol_var),
            ("Updates", self.binance_updates_var),
            ("Last Update", self.binance_last_update_var),
            ("Server Time", self.binance_server_time_var),
            ("Latency", self.binance_latency_var),
            ("Mark Price", self.binance_mark_price_var),
            ("Funding Rate", self.binance_funding_var),
            ("Open Interest", self.binance_oi_var),
            ("Order Book", self.binance_order_pressure_var),
            ("Taker Flow", self.binance_taker_pressure_var),
            ("Long/Short", self.binance_ls_ratio_var),
            ("Sentiment", self.binance_sentiment_var),
        ]

        grid = ttk.Frame(binance_card, style="DeepCard.TFrame")
        grid.pack(fill="x")

        for idx, (label, var) in enumerate(rows):
            row = idx // 2
            col = (idx % 2) * 2
            ttk.Label(grid, text=f"{label}:", style="DeepLabel.TLabel").grid(
                row=row, column=col, sticky="w", padx=(0, 6), pady=2
            )
            ttk.Label(grid, textvariable=var, style="DeepLabel.TLabel").grid(
                row=row, column=col + 1, sticky="w", padx=(0, 12), pady=2
            )

        grid.columnconfigure(0, weight=0)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=0)
        grid.columnconfigure(3, weight=1)

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
            result.get("stop_loss", result.get("sl", 0.0)),
            result.get("take_profit_1", result.get("tp1", 0.0)),
            result.get("take_profit_2", result.get("tp2", 0.0)),
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

    def update_binance_live_data(self, payload: Dict[str, Any]) -> None:
        """Update Binance live feed section."""
        if payload.get("ok"):
            self.binance_status_var.set("Connected ✓")
            self.binance_symbol_var.set(payload.get("symbol", "-"))
            self.binance_updates_var.set(str(payload.get("updates", 0)))
            self.binance_last_update_var.set(payload.get("last_update", "-"))
            self.binance_server_time_var.set(payload.get("server_time", "-"))
            self.binance_latency_var.set(payload.get("latency", "-"))
            self.binance_mark_price_var.set(payload.get("mark_price", "-"))
            self.binance_funding_var.set(payload.get("funding_rate", "-"))
            self.binance_oi_var.set(payload.get("open_interest", "-"))
            self.binance_order_pressure_var.set(payload.get("order_book", "-"))
            self.binance_taker_pressure_var.set(payload.get("taker_ratio", "-"))
            self.binance_ls_ratio_var.set(payload.get("long_short_ratio", "-"))
            self.binance_sentiment_var.set(payload.get("sentiment", "-"))
            self.set_status("Binance Live ✓")
        else:
            error = payload.get("error", "Unknown error")
            self.binance_status_var.set(f"Error: {error}")
            self.set_status("Binance Error")


class DeepAnalysisWindow(tk.Toplevel):
    """
    Standalone window for deep analysis dashboard.
    
    Can be opened from the main GUI to show detailed analysis results.
    """
    
    def __init__(
        self,
        parent,
        title: str = "Deep Analysis Dashboard",
        symbol: Optional[str] = None,
        refresh_interval_s: float = 1.0,
    ):
        super().__init__(parent)
        
        self.title(title)
        self.geometry("900x800")
        self.minsize(800, 600)
        self.configure(bg=DeepStyle.BG_DARK)

        self._live_symbol = self._resolve_symbol(symbol)
        self._refresh_interval_s = max(1.0, refresh_interval_s)
        self._live_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._live_stop = threading.Event()
        self._live_updates = 0
        
        # Create dashboard
        self.dashboard = DeepAnalysisDashboard(self)
        self.dashboard.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_live_updates()

    def update_from_result(self, result: Dict[str, Any]) -> None:
        """Update dashboard with analysis result."""
        self.dashboard.update_from_analysis_result(result)

    def _resolve_symbol(self, symbol: Optional[str]) -> str:
        if symbol:
            return symbol.upper()
        settings = load_settings("settings.json")
        sym_cfg = settings.get("symbols", {})
        symbols = sym_cfg.get("list") or sym_cfg.get("default") or []
        if symbols:
            return str(symbols[0]).upper()
        return "BTCUSDT"

    def _start_live_updates(self) -> None:
        self.dashboard.update_binance_live_data(
            {"ok": True, "symbol": self._live_symbol, "updates": 0}
        )
        self._live_thread = threading.Thread(
            target=self._live_worker,
            name="binance-live-feed",
            daemon=True,
        )
        self._live_thread.start()
        self.after(200, self._drain_live_queue)

    def _live_worker(self) -> None:
        while not self._live_stop.is_set():
            start = time.perf_counter()
            try:
                market_data = binance_data.get_comprehensive_market_data(self._live_symbol)
                server_time = binance_data.server_time()
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._live_updates += 1
                payload = self._format_live_payload(
                    market_data,
                    server_time,
                    latency_ms,
                    self._live_updates,
                )
            except Exception as exc:
                payload = {"ok": False, "error": str(exc)}
            self._live_queue.put(payload)
            self._live_stop.wait(self._refresh_interval_s)

    def _format_live_payload(
        self,
        market_data: Dict[str, Any],
        server_time: Dict[str, Any],
        latency_ms: float,
        updates: int,
    ) -> Dict[str, Any]:
        mark_price = market_data.get("markPrice", {})
        funding = market_data.get("funding", {})
        open_interest = market_data.get("openInterest", {})
        order_book = market_data.get("orderBook", {})
        taker_ratio = market_data.get("takerRatio", {})
        long_short = market_data.get("longShortRatio", {})
        server_ts = server_time.get("serverTime")
        if server_ts:
            server_dt = datetime.fromtimestamp(server_ts / 1000, tz=timezone.utc)
            server_time_str = server_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            server_time_str = "-"
        return {
            "ok": True,
            "symbol": market_data.get("symbol", self._live_symbol),
            "updates": updates,
            "last_update": datetime.now().strftime("%H:%M:%S"),
            "server_time": server_time_str,
            "latency": f"{latency_ms:.0f} ms",
            "mark_price": f"{mark_price.get('markPrice', 0.0):.4f}",
            "funding_rate": f"{funding.get('fundingRatePct', 0.0):.4f}%",
            "open_interest": f"{open_interest.get('openInterest', 0.0):,.2f}",
            "order_book": order_book.get("pressure", "UNKNOWN"),
            "taker_ratio": taker_ratio.get("pressure", "UNKNOWN"),
            "long_short_ratio": f"{long_short.get('longShortRatio', 0.0):.2f}",
            "sentiment": market_data.get("overallSentiment", "UNKNOWN"),
        }

    def _drain_live_queue(self) -> None:
        try:
            while True:
                payload = self._live_queue.get_nowait()
                if payload.get("ok"):
                    payload.setdefault("symbol", self._live_symbol)
                self.dashboard.update_binance_live_data(payload)
        except queue.Empty:
            pass
        if not self._live_stop.is_set():
            self.after(200, self._drain_live_queue)

    def _on_close(self) -> None:
        self._live_stop.set()
        self.destroy()
