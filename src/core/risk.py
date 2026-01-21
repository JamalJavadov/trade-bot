from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .types import SymbolFilters


@dataclass
class RiskResult:
    ok: bool
    reason: str = ""
    qty: float = 0.0
    risk_target: float = 0.0
    risk_actual: float = 0.0
    notional: float = 0.0
    margin_used: float = 0.0
    notes: Optional[List[str]] = None


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return (x // step) * step


def calc_position_size(
    entry: float,
    stop: float,
    budget_usdt: float,
    risk_pct: float,
    leverage: int,
    filters: Optional[SymbolFilters] = None,
) -> RiskResult:
    if budget_usdt <= 0:
        return RiskResult(ok=False, reason="Budget <= 0")

    if leverage <= 0:
        return RiskResult(ok=False, reason="Leverage <= 0")

    risk_target = budget_usdt * risk_pct
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return RiskResult(ok=False, reason="Entry/SL eynidir (risk 0)")

    qty = risk_target / per_unit_risk
    notes: List[str] = []

    # Notional cap: margin budget * leverage
    notional_cap = budget_usdt * leverage
    notional = qty * entry
    if notional > notional_cap:
        qty = notional_cap / entry
        notes.append("Notional cap tətbiq olundu")
        notional = qty * entry

    # Apply filters
    if filters:
        if filters.step_size > 0:
            qty = _round_step(qty, filters.step_size)
        if filters.min_qty > 0 and qty < filters.min_qty:
            return RiskResult(ok=False, reason=f"Qty minQty-dən aşağıdır ({qty} < {filters.min_qty})")
        if filters.min_notional > 0 and notional < filters.min_notional:
            notes.append(f"Min notional aşağıdır ({notional:.4f} < {filters.min_notional:.4f})")

    risk_actual = qty * per_unit_risk
    margin_used = notional / leverage

    return RiskResult(
        ok=True,
        qty=float(qty),
        risk_target=float(risk_target),
        risk_actual=float(risk_actual),
        notional=float(notional),
        margin_used=float(margin_used),
        notes=notes or None,
    )
