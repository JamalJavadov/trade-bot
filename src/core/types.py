from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SymbolFilters:
    tick_size: float = 0.0
    step_size: float = 0.0
    min_qty: float = 0.0
    min_notional: float = 0.0


@dataclass
class Plan:
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr1: float
    rr2: float
    score: float
    reason: str
    leverage: int
    margin_type: str = "Isolated"
    qty: float = 0.0
    risk_target: float = 0.0
    risk_actual: float = 0.0
    notes: Optional[List[str]] = None

    def to_txt(self) -> str:
        notes = ""
        if self.notes:
            notes = "\n[QEYD] " + " | ".join(self.notes)

        action = "Buy/Long" if self.side == "LONG" else "Sell/Short"

        return (
            f"=== BEST PLAN ===\n"
            f"{self.symbol} | {self.side}\n"
            f"Entry={self.entry:.6f} SL={self.sl:.6f} TP1={self.tp1:.6f} TP2={self.tp2:.6f}\n"
            f"RR1={self.rr1:.2f} RR2={self.rr2:.2f}\n"
            f"Səbəb: {self.reason}\n\n"
            f"=== BINANCE FUTURES FORM (manual doldurma) ===\n"
            f"Market: {self.symbol}\n"
            f"Margin: {self.margin_type}\n"
            f"Leverage: {self.leverage}x\n\n"
            f"Tab: Limit\n"
            f"Price (Entry): {self.entry:.6f}\n"
            f"Size (Qty): {self.qty:.6f}\n"
            f"TP/SL: ON\n"
            f"  Take Profit: {self.tp2:.6f}   Trigger: Mark\n"
            f"  Stop Loss:   {self.sl:.6f}   Trigger: Mark\n"
            f"Reduce-Only: OFF (entry açırsan)\n"
            f"TIF: GTC\n"
            f"Action: {action}\n\n"
            f"Expiry: {7} gün (trigger olmazsa cancel)\n"
            f"Risk target: {self.risk_target:.4f} USDT | Risk actual: {self.risk_actual:.4f} USDT\n"
            f"{notes}\n"
        )


@dataclass
class ScanResult:
    symbol: str
    ok: bool
    side: str = "NO_TRADE"
    rr2: float = 0.0
    score: float = 0.0
    reason: str = ""

    def summary_line(self) -> str:
        if not self.ok:
            return f"{self.symbol}: NO TRADE"
        return f"{self.symbol}: OK | {self.side} | RR2={self.rr2:.2f} | score={self.score:.2f}"
