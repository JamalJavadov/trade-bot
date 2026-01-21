from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from ..core import binance_data
from ..core import analyzer
from ..core.planner import (
    load_settings,
    save_settings,
    run_scan_and_build_best_plan,
    format_report,
)


class ModernStyle:
    """Modern rəng və stil konfiqurasiyası"""
    
    # Rəng Paleti
    BG_DARK = "#1e1e2e"
    BG_MEDIUM = "#2a2a3e"
    BG_LIGHT = "#363654"
    ACCENT_PRIMARY = "#00d4ff"
    ACCENT_SUCCESS = "#00ff88"
    ACCENT_WARNING = "#ffaa00"
    ACCENT_ERROR = "#ff4444"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#b4b4c8"
    BORDER = "#404058"
    
    # Font Konfiqurasiyası
    FONT_MAIN = ("Segoe UI", 10)
    FONT_HEADER = ("Segoe UI Semibold", 11)
    FONT_TITLE = ("Segoe UI Bold", 12)
    FONT_MONO = ("Consolas", 9)


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


class ModernButton(tk.Canvas):
    """Hover effektli custom button"""
    
    def __init__(self, parent, text, command, **kwargs):
        self.text = text
        self.command = command
        self.hovered = False
        
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
        bg_color = ModernStyle.ACCENT_PRIMARY if self.hovered else ModernStyle.BG_LIGHT
        self.create_rectangle(
            2, 2, width-2, height-2,
            fill=bg_color,
            outline=ModernStyle.ACCENT_PRIMARY,
            width=1
        )
        
        # Text
        text_color = ModernStyle.BG_DARK if self.hovered else ModernStyle.TEXT_PRIMARY
        self.create_text(
            width // 2, height // 2,
            text=self.text,
            fill=text_color,
            font=ModernStyle.FONT_HEADER
        )
    
    def _on_enter(self, event):
        self.hovered = True
        self._draw()
    
    def _on_leave(self, event):
        self.hovered = False
        self._draw()
    
    def _on_click(self, event):
        if self.command:
            self.command()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Trade Bot - Professional Edition")
        self.root.configure(bg=ModernStyle.BG_DARK)
        
        # Window konfiqurasiyası
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Queue və thread
        self._q: queue.Queue = queue.Queue()
        self._scan_thread: Optional[threading.Thread] = None
        self._spinning = False
        self._spinner_i = 0
        self._spinner_frames = ["◐", "◓", "◑", "◒"]
        
        # Settings
        self.settings = load_settings("settings.json")
        
        # Style konfiqurasiyası
        self._setup_styles()
        
        # UI Elementləri
        self._create_header()
        self._create_parameters_section()
        self._create_scan_section()
        self._create_output_section()
        
        # Queue polling
        self.root.after(120, self._poll_queue)
    
    def _setup_styles(self):
        """TTK stil konfiqurasiyası"""
        style = ttk.Style()
        
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
    
    def _create_header(self):
        """Header bölməsi"""
        header = ttk.Frame(self.root, style="Dark.TFrame", padding=15)
        header.pack(fill="x")
        
        # Title
        title_frame = ttk.Frame(header, style="Dark.TFrame")
        title_frame.pack(side="left", fill="x", expand=True)
        
        ttk.Label(
            title_frame,
            text="⚡ TRADE BOT",
            style="Header.TLabel"
        ).pack(side="left")
        
        # Status indicator
        self.status_indicator = StatusIndicator(header)
        self.status_indicator.pack(side="right", padx=(10, 0))
        
        self.status_var = tk.StringVar(value="Hazır")
        status_label = ttk.Label(
            header,
            textvariable=self.status_var,
            style="Header.TLabel"
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
            self.root,
            text="⚙️  Parametrlər",
            style="Card.TLabelframe",
            padding=20
        )
        params_card.pack(fill="x", padx=15, pady=(10, 5))
        
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
    
    def _create_scan_section(self):
        """Scan bölməsi"""
        scan_card = ttk.LabelFrame(
            self.root,
            text="🔍 Skan",
            style="Card.TLabelframe",
            padding=20
        )
        scan_card.pack(fill="x", padx=15, pady=5)
        
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
        self.progress = AnimatedProgressBar(scan_card)
        self.progress.pack(fill="x", pady=(0, 15))
        
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
            self.root,
            text="📄 Nəticələr",
            style="Card.TLabelframe",
            padding=15
        )
        output_card.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        
        # Text widget with scrollbar
        txt_frame = ttk.Frame(output_card, style="Card.TFrame")
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
            yscrollcommand=scrollbar.set
        )
        self.txt.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.txt.yview)
    
    def _symbols_count_text(self) -> str:
        s = self.settings.get("symbols", {})
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
        
        save_settings("settings.json", self.settings)
        self.symbols_count_var.set(self._symbols_count_text())
        self.status_var.set("✓ Saxlanıldı")
        self.status_indicator.set_state("success")
        
        # Reset status after 2 seconds
        self.root.after(2000, lambda: self.status_var.set("Hazır"))
        self.root.after(2000, lambda: self.status_indicator.set_state("idle"))
    
    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        # Note: ModernButton doesn't have configure, we'll handle it differently
        # For now, we can just change the visual state
        if busy:
            self.status_indicator.set_state("active")
        else:
            self.status_indicator.set_state("idle")
    
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
    
    def on_scan(self):
        self.on_save()
        
        self.txt.delete("1.0", "end")
        self.txt.insert("end", "🔄 Skan başlayır...\n\n")
        
        self._set_busy(True)
        self._start_spinner()
        self.stage_var.set("⚙️  İnisializasiya...")
        self.progress.configure(value=0, maximum=100)
        
        budget = float(self.budget_var.get())
        risk_pct = float(self.risk_pct_var.get())
        lev = int(self.lev_var.get())
        
        def worker():
            try:
                settings = load_settings("settings.json")
                
                sym_cfg = settings.get("symbols", {})
                if bool(sym_cfg.get("auto_all_usdtm", False)):
                    symbols = binance_data.list_usdtm_perp_symbols()
                else:
                    symbols = sym_cfg.get("list", []) or sym_cfg.get("default", []) or []
                symbols = [str(x).upper().strip() for x in symbols if str(x).strip()]
                
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
                report = format_report(results, best, settings)
                self._q.put(("done", report))
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
                    if total > 0:
                        self.progress.configure(maximum=total, value=i)
                
                elif kind == "stage":
                    _, text = msg
                    self.stage_var.set(f"⚙️  {text}")
                
                elif kind == "done":
                    _, report = msg
                    self.txt.delete("1.0", "end")
                    self.txt.insert("end", report)
                    self.status_var.set("✓ Tamamlandı")
                    self.status_indicator.set_state("success")
                    self._stop_spinner()
                    self._set_busy(False)
                    self.stage_var.set("✓ Hazır")
                
                elif kind == "error":
                    _, err = msg
                    self._stop_spinner()
                    self._set_busy(False)
                    self.status_indicator.set_state("error")
                    messagebox.showerror("Xəta", err)
        
        except queue.Empty:
            pass
        
        self.root.after(120, self._poll_queue)
