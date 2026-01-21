from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple

import numpy as np
import pandas as pd


@dataclass
class Analysis:
    status: str              # "OK" | "SETUP" | "NO_TRADE"
    side: str                # "LONG" | "SHORT" | "-"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    rr1: float = 0.0
    rr2: float = 0.0
    score: float = 0.0
    reason: str = ""         # niyə OK/SETUP/NO_TRADE


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    if len(tr) < period:
        return float(np.nan)
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def _trend_4h(df4: pd.DataFrame) -> str:
    if len(df4) < 250:
        return "RANGE"
    ema50 = _ema(df4["close"], 50).iloc[-1]
    ema200 = _ema(df4["close"], 200).iloc[-1]
    if ema50 > ema200:
        return "LONG"
    if ema50 < ema200:
        return "SHORT"
    return "RANGE"


def _impulse_leg_1h(df1: pd.DataFrame, side: str, lookback: int = 240) -> Optional[Tuple[float, float]]:
    d = df1.tail(lookback).copy()
    if len(d) < 80:
        return None

    if side == "LONG":
        # impulse: recent low -> later high
        i_low = int(d["low"].idxmin())
        d2 = df1.loc[i_low:].copy()
        if len(d2) < 30:
            return None
        hi = float(d2["high"].max())
        lo = float(df1.loc[i_low, "low"])
        if hi <= lo:
            return None
        return lo, hi

    if side == "SHORT":
        # impulse: recent high -> later low
        i_high = int(d["high"].idxmax())
        d2 = df1.loc[i_high:].copy()
        if len(d2) < 30:
            return None
        lo = float(d2["low"].min())
        hi = float(df1.loc[i_high, "high"])
        if hi <= lo:
            return None
        return hi, lo

    return None


def _fib_zone_for_long(lo: float, hi: float) -> Tuple[float, float]:
    rng = hi - lo
    z50 = hi - rng * 0.50
    z618 = hi - rng * 0.618
    return float(max(z50, z618)), float(min(z50, z618))  # (zone_high, zone_low)


def _fib_zone_for_short(hi: float, lo: float) -> Tuple[float, float]:
    rng = hi - lo
    z50 = lo + rng * 0.50
    z618 = lo + rng * 0.618
    return float(min(z50, z618)), float(max(z50, z618))  # (zone_low, zone_high)


def _liquidity_sweep_confirm(df5: pd.DataFrame, side: str, z_low: float, z_high: float, lookback: int = 60) -> bool:
    d = df5.tail(lookback)
    if len(d) < 20:
        return False

    if side == "LONG":
        # sweep: low pierces below zone_low then close back inside zone (>= zone_low)
        pierced = (d["low"] < z_low).to_numpy()
        close_back = (d["close"] >= z_low).to_numpy()
        return bool(np.any(pierced & close_back))

    if side == "SHORT":
        # sweep: high pierces above zone_high then close back inside zone (<= zone_high)
        pierced = (d["high"] > z_high).to_numpy()
        close_back = (d["close"] <= z_high).to_numpy()
        return bool(np.any(pierced & close_back))

    return False


def analyze_symbol(
    symbol: str,
    fetch_ohlcv: Callable[[str, str, int], pd.DataFrame],
    settings: Dict[str, Any],
    on_stage: Optional[Callable[[str], None]] = None,
) -> Analysis:
    scan_cfg = settings.get("scan", {})
    limit4 = int(scan_cfg.get("limit_4h", 500))
    limit1 = int(scan_cfg.get("limit_1h", 500))
    limit5 = int(scan_cfg.get("limit_5m", 300))

    risk = settings.get("risk", {})
    min_rr2 = float(risk.get("min_rr2", 2.0))
    max_entry_atr = float(risk.get("max_entry_distance_atr", 2.0))
    sl_atr_mult = float(risk.get("sl_atr_mult", 0.8))

    plan_cfg = settings.get("plan", {})
    require_sweep = bool(plan_cfg.get("require_liquidity_sweep", True))
    allow_setup = bool(plan_cfg.get("allow_setup_if_no_confirm", True))  # SETUP mode

    if on_stage:
        on_stage("fetch 4H")
    df4 = fetch_ohlcv(symbol, "4h", limit4)

    bias = _trend_4h(df4)
    if bias == "RANGE":
        return Analysis(status="NO_TRADE", side="-", reason="4H bias RANGE (EMA50≈EMA200)")

    if on_stage:
        on_stage("fetch 1H")
    df1 = fetch_ohlcv(symbol, "1h", limit1)

    leg = _impulse_leg_1h(df1, bias)
    if not leg:
        return Analysis(status="NO_TRADE", side="-", reason="1H impulse leg tapılmadı (struktur zəif)")

    last_close = float(df1["close"].iloc[-1])
    atr1 = _atr(df1, 14)
    if not np.isfinite(atr1) or atr1 <= 0:
        return Analysis(status="NO_TRADE", side="-", reason="ATR hesablanmadı")

    if bias == "LONG":
        lo, hi = leg
        z_high, z_low = _fib_zone_for_long(lo, hi)
        # price must be above zone_low (zone not broken)
        if last_close < z_low:
            return Analysis(status="NO_TRADE", side="-", reason="Price 0.618 altına düşüb (zone broken)")
        entry = z_high if last_close >= z_high else z_low
        # SL below zone_low with ATR buffer
        sl = z_low - atr1 * sl_atr_mult
        tp1 = hi  # prior swing high
        tp2 = hi + (hi - lo) * 0.272  # extension-ish
    else:
        hi, lo = leg
        z_low, z_high = _fib_zone_for_short(hi, lo)
        if last_close > z_high:
            return Analysis(status="NO_TRADE", side="-", reason="Price 0.618 üstünə çıxıb (zone broken)")
        entry = z_low if last_close <= z_low else z_high
        sl = z_high + atr1 * sl_atr_mult
        tp1 = lo
        tp2 = lo - (hi - lo) * 0.272

    # distance filter (pending setups too far are not "professional")
    dist_atr = abs(last_close - entry) / atr1
    if dist_atr > max_entry_atr:
        return Analysis(status="NO_TRADE", side="-", reason=f"Entry çox uzaqdır (dist={dist_atr:.2f} ATR)")

    # RR
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return Analysis(status="NO_TRADE", side="-", reason="SL risk invalid")

    rr1 = abs(tp1 - entry) / risk_per_unit
    rr2 = abs(tp2 - entry) / risk_per_unit

    # confirmation (liquidity sweep + close back inside zone) when required
    confirmed = True
    if require_sweep:
        if on_stage:
            on_stage("fetch 5M (confirm sweep)")
        df5 = fetch_ohlcv(symbol, "5m", limit5)
        if bias == "LONG":
            confirmed = _liquidity_sweep_confirm(df5, "LONG", z_low=z_low, z_high=z_high)
        else:
            confirmed = _liquidity_sweep_confirm(df5, "SHORT", z_low=z_low, z_high=z_high)

    # scoring: reward (RR2) - penalty (distance)
    score = rr2 * 10.0 - dist_atr * 5.0

    if rr2 < min_rr2:
        return Analysis(status="NO_TRADE", side="-", reason=f"RR2 aşağıdır ({rr2:.2f} < {min_rr2})")

    if confirmed:
        return Analysis(
            status="OK",
            side=bias,
            entry=float(entry),
            sl=float(sl),
            tp1=float(tp1),
            tp2=float(tp2),
            rr1=float(rr1),
            rr2=float(rr2),
            score=float(score),
            reason="HTF bias + Fib zone + sweep confirmation",
        )

    # not confirmed yet => SETUP (watch)
    if allow_setup:
        return Analysis(
            status="SETUP",
            side=bias,
            entry=float(entry),
            sl=float(sl),
            tp1=float(tp1),
            tp2=float(tp2),
            rr1=float(rr1),
            rr2=float(rr2),
            score=float(score),
            reason="Setup var, amma sweep confirmation hələ gəlməyib (WATCH)",
        )

    return Analysis(status="NO_TRADE", side="-", reason="Confirmation yoxdur")
