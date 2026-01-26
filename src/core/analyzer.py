from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple, List

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
    details: Optional[Dict[str, Any]] = None


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return float("nan")
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    if len(tr) < period:
        return float("nan")
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period).mean()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _fractals_5(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    up: List[int] = []
    down: List[int] = []
    for i in range(2, len(df) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            up.append(i)
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            down.append(i)
    return up, down


def _trend_bias_ema(df4: pd.DataFrame) -> str:
    if len(df4) < 250:
        return "RANGE"
    ema50 = _ema(df4["close"], 50).iloc[-1]
    ema200 = _ema(df4["close"], 200).iloc[-1]
    if ema50 > ema200:
        return "LONG"
    if ema50 < ema200:
        return "SHORT"
    return "RANGE"


def _premium_discount_bias(df4: pd.DataFrame, lookback: int) -> Optional[str]:
    if len(df4) < 60:
        return None
    d = df4.tail(lookback)
    high = float(d["high"].max())
    low = float(d["low"].min())
    if high <= low:
        return None
    mid = (high + low) / 2.0
    last = float(d["close"].iloc[-1])
    if last < mid:
        return "LONG"
    if last > mid:
        return "SHORT"
    return None


def _aggregate_bias(biases: Dict[str, Optional[str]], min_alignment: float) -> Tuple[Optional[str], Dict[str, int], float]:
    counts = {"LONG": 0, "SHORT": 0}
    for bias in biases.values():
        if bias in counts:
            counts[bias] += 1
    total = counts["LONG"] + counts["SHORT"]
    if total == 0:
        return None, counts, 0.0
    if counts["LONG"] == counts["SHORT"]:
        return None, counts, 0.0
    best = "LONG" if counts["LONG"] > counts["SHORT"] else "SHORT"
    ratio = counts[best] / total
    if ratio < min_alignment:
        return None, counts, ratio
    return best, counts, ratio


def _select_impulse_leg(df: pd.DataFrame, side: str, lookback: int) -> Optional[Tuple[int, int, float, float]]:
    d = df.tail(lookback)
    if len(d) < 50:
        return None

    up, down = _fractals_5(d)
    if side == "LONG":
        if up:
            last_up = up[-1]
            downs_before = [i for i in down if i < last_up]
            if downs_before:
                start = downs_before[-1]
                end = last_up
                lo = float(d.iloc[start]["low"])
                hi = float(d.iloc[end]["high"])
                if hi > lo:
                    return d.index[start], d.index[end], lo, hi
        idx = int(d["low"].idxmin())
        d2 = df.loc[idx:]
        if len(d2) >= 10:
            lo = float(df.loc[idx, "low"])
            hi = float(d2["high"].max())
            if hi > lo:
                return idx, int(d2["high"].idxmax()), lo, hi

    if side == "SHORT":
        if down:
            last_down = down[-1]
            ups_before = [i for i in up if i < last_down]
            if ups_before:
                start = ups_before[-1]
                end = last_down
                hi = float(d.iloc[start]["high"])
                lo = float(d.iloc[end]["low"])
                if hi > lo:
                    return d.index[start], d.index[end], hi, lo
        idx = int(d["high"].idxmax())
        d2 = df.loc[idx:]
        if len(d2) >= 10:
            hi = float(df.loc[idx, "high"])
            lo = float(d2["low"].min())
            if hi > lo:
                return idx, int(d2["low"].idxmin()), hi, lo

    return None


def _fib_retracement(lo: float, hi: float, level: float) -> float:
    rng = hi - lo
    return hi - rng * level


def _fib_retracement_short(hi: float, lo: float, level: float) -> float:
    rng = hi - lo
    return lo + rng * level


def _fib_extension_long(lo: float, hi: float, level: float) -> float:
    rng = hi - lo
    return hi - rng * level


def _fib_extension_short(hi: float, lo: float, level: float) -> float:
    rng = hi - lo
    return lo + (-level) * rng


def _is_strong_trend(
    df: pd.DataFrame,
    side: str,
    atr: float,
    slope_lookback: int,
    min_slope_atr: float,
    min_ema_separation_atr: float,
) -> bool:
    if len(df) < 220 or not np.isfinite(atr) or atr <= 0:
        return False
    ema50 = _ema(df["close"], 50)
    ema200 = _ema(df["close"], 200)
    slope = ema50.iloc[-1] - ema50.iloc[-slope_lookback]
    if side == "LONG" and slope <= 0:
        return False
    if side == "SHORT" and slope >= 0:
        return False
    slope_atr = abs(float(slope)) / atr
    ema_separation_atr = abs(float(ema50.iloc[-1] - ema200.iloc[-1])) / atr
    return slope_atr >= min_slope_atr and ema_separation_atr >= min_ema_separation_atr


def _trend_strength_score(
    df: pd.DataFrame,
    side: str,
    atr: float,
    slope_lookback: int,
    min_slope_atr: float,
    min_ema_separation_atr: float,
) -> float:
    if len(df) < 220 or not np.isfinite(atr) or atr <= 0:
        return 0.0
    ema50 = _ema(df["close"], 50)
    ema200 = _ema(df["close"], 200)
    slope = ema50.iloc[-1] - ema50.iloc[-slope_lookback]
    if side == "LONG" and slope <= 0:
        return 0.0
    if side == "SHORT" and slope >= 0:
        return 0.0
    slope_atr = abs(float(slope)) / atr
    ema_separation_atr = abs(float(ema50.iloc[-1] - ema200.iloc[-1])) / atr
    slope_score = _clamp(slope_atr / max(min_slope_atr, 1e-6))
    separation_score = _clamp(ema_separation_atr / max(min_ema_separation_atr, 1e-6))
    return (slope_score + separation_score) / 2.0


def _zone_bounds(a: float, b: float) -> Tuple[float, float]:
    return (min(a, b), max(a, b))


def _zone_tolerance(price: float, atr: float, atr_mult: float, pct: float) -> float:
    atr_part = atr * atr_mult if np.isfinite(atr) else 0.0
    pct_part = price * pct
    return max(atr_part, pct_part)


def _percentile_rank(series: pd.Series, value: float) -> float:
    if series.empty or not np.isfinite(value):
        return 0.0
    return float((series <= value).mean())


def _in_zone(price: float, low: float, high: float, tol: float) -> bool:
    return (price >= low - tol) and (price <= high + tol)


def _ema50_confluence(df: pd.DataFrame, side: str, zone_low: float, zone_high: float, tol: float) -> bool:
    if len(df) < 55:
        return False
    ema50 = _ema(df["close"], 50)
    slope = ema50.iloc[-1] - ema50.iloc[-5]
    last_ema = float(ema50.iloc[-1])
    if side == "LONG" and slope <= 0:
        return False
    if side == "SHORT" and slope >= 0:
        return False
    return _in_zone(last_ema, zone_low, zone_high, tol)


def _anchored_vwap(df: pd.DataFrame, start_idx: int) -> Optional[float]:
    if start_idx not in df.index:
        return None
    d = df.loc[start_idx:]
    if d.empty:
        return None
    typical = (d["high"] + d["low"] + d["close"]) / 3.0
    vol = d["volume"].astype(float)
    denom = float(vol.sum())
    if denom == 0:
        return None
    return float((typical * vol).sum() / denom)


def _swap_zone_confluence(df: pd.DataFrame, side: str, zone_low: float, zone_high: float, tol: float) -> bool:
    up, down = _fractals_5(df)
    if side == "LONG" and up:
        last_res_idx = up[-1]
        res = float(df.iloc[last_res_idx]["high"])
        return _in_zone(res, zone_low, zone_high, tol)
    if side == "SHORT" and down:
        last_sup_idx = down[-1]
        sup = float(df.iloc[last_sup_idx]["low"])
        return _in_zone(sup, zone_low, zone_high, tol)
    return False


def _bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]
    return (c2["close"] > c2["open"] and c1["close"] < c1["open"]
            and c2["open"] <= c1["close"] and c2["close"] >= c1["open"])


def _bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    c1 = df.iloc[-2]
    c2 = df.iloc[-1]
    return (c2["close"] < c2["open"] and c1["close"] > c1["open"]
            and c2["open"] >= c1["close"] and c2["close"] <= c1["open"])


def _morning_star(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]
    body1 = abs(c1["close"] - c1["open"])
    body2 = abs(c2["close"] - c2["open"])
    body3 = abs(c3["close"] - c3["open"])
    if body1 == 0 or body3 == 0:
        return False
    midpoint = c1["open"] - body1 / 2 if c1["close"] < c1["open"] else c1["open"] + body1 / 2
    return (
        c1["close"] < c1["open"]
        and body2 <= body1 * 0.5
        and c3["close"] > c3["open"]
        and c3["close"] >= midpoint
    )


def _evening_star(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]
    body1 = abs(c1["close"] - c1["open"])
    body2 = abs(c2["close"] - c2["open"])
    body3 = abs(c3["close"] - c3["open"])
    if body1 == 0 or body3 == 0:
        return False
    midpoint = c1["open"] + body1 / 2 if c1["close"] > c1["open"] else c1["open"] - body1 / 2
    return (
        c1["close"] > c1["open"]
        and body2 <= body1 * 0.5
        and c3["close"] < c3["open"]
        and c3["close"] <= midpoint
    )


def _rejection_wick(df: pd.DataFrame, side: str, zone_low: float, zone_high: float) -> bool:
    if len(df) < 1:
        return False
    c = df.iloc[-1]
    if side == "LONG":
        wick_through = c["low"] < zone_low
        close_back = c["close"] >= zone_low
        return bool(wick_through and close_back and c["close"] > c["open"])
    wick_through = c["high"] > zone_high
    close_back = c["close"] <= zone_high
    return bool(wick_through and close_back and c["close"] < c["open"])


def _golden_respect(
    df: pd.DataFrame,
    side: str,
    zone_low: float,
    zone_high: float,
    tol: float,
    atr: float,
    body_atr_ratio: float,
) -> bool:
    recent = df.tail(3)
    if recent.empty:
        return False
    if side == "LONG":
        closes_ok = (recent["close"] >= zone_low - tol).all()
        wicks_ok = (recent["low"] <= zone_high + tol).any()
    else:
        closes_ok = (recent["close"] <= zone_high + tol).all()
        wicks_ok = (recent["high"] >= zone_low - tol).any()
    body_ok = True
    if np.isfinite(atr) and atr > 0:
        avg_body = (recent["close"] - recent["open"]).abs().mean()
        body_ok = avg_body <= atr * body_atr_ratio
    return bool(closes_ok and wicks_ok and body_ok)


def _liquidity_sweep(df: pd.DataFrame, side: str) -> Optional[int]:
    up, down = _fractals_5(df)
    if side == "SHORT" and up:
        prev_high_idx = up[-1]
        prev_high = float(df.iloc[prev_high_idx]["high"])
        recent = df.tail(5)
        sweep = recent[(recent["high"] > prev_high) & (recent["close"] < prev_high)]
        if not sweep.empty:
            return int(sweep.index[-1])
    if side == "LONG" and down:
        prev_low_idx = down[-1]
        prev_low = float(df.iloc[prev_low_idx]["low"])
        recent = df.tail(5)
        sweep = recent[(recent["low"] < prev_low) & (recent["close"] > prev_low)]
        if not sweep.empty:
            return int(sweep.index[-1])
    return None


def _break_of_structure(df: pd.DataFrame, side: str, lookback: int = 30) -> Optional[int]:
    d = df.tail(lookback)
    up, down = _fractals_5(d)
    if side == "SHORT" and down:
        last_low_idx = down[-1]
        lvl = float(d.iloc[last_low_idx]["low"])
        bos = d[d["close"] < lvl]
        if not bos.empty:
            return int(bos.index[-1])
    if side == "LONG" and up:
        last_high_idx = up[-1]
        lvl = float(d.iloc[last_high_idx]["high"])
        bos = d[d["close"] > lvl]
        if not bos.empty:
            return int(bos.index[-1])
    return None


def _find_fvg(df: pd.DataFrame, side: str, lookback: int = 40) -> Optional[Tuple[float, float]]:
    d = df.tail(lookback)
    if len(d) < 3:
        return None
    for i in range(len(d) - 1, 1, -1):
        c0 = d.iloc[i - 2]
        c2 = d.iloc[i]
        if side == "SHORT":
            if c0["low"] > c2["high"]:
                return float(c2["high"]), float(c0["low"])
        else:
            if c0["high"] < c2["low"]:
                return float(c0["high"]), float(c2["low"])
    return None


def _build_golden_zone_setup(
    df1: pd.DataFrame,
    df_confirm: pd.DataFrame,
    side: str,
    atr: float,
    cfg: Dict[str, Any],
) -> Optional[Analysis]:
    lookback = int(cfg.get("impulse_lookback", 240))
    leg = _select_impulse_leg(df1, side, lookback=lookback)
    if not leg:
        return None

    start_idx, end_idx, a, b = leg
    zone_mode = str(cfg.get("fib_zone_mode", "golden")).lower()
    zone_label = "Golden zone"
    strong_trend = _is_strong_trend(
        df1,
        side,
        atr,
        int(cfg.get("strong_trend_slope_lookback", 8)),
        float(cfg.get("strong_trend_slope_atr", 0.12)),
        float(cfg.get("strong_trend_ema_separation_atr", 0.3)),
    )
    low_lvl, high_lvl = 0.618, 0.5
    if zone_mode not in ("golden", "golden_zone"):
        zone_mode = "golden"

    if side == "LONG":
        lo, hi = a, b
        z_low, z_high = _zone_bounds(
            _fib_retracement(lo, hi, low_lvl),
            _fib_retracement(lo, hi, high_lvl),
        )
    elif side == "SHORT":
        hi, lo = a, b
        z_low, z_high = _zone_bounds(
            _fib_retracement_short(hi, lo, high_lvl),
            _fib_retracement_short(hi, lo, low_lvl),
        )
    else:
        return None

    last_close = float(df1["close"].iloc[-1])
    tol = _zone_tolerance(last_close, atr, float(cfg.get("zone_tolerance_atr", 0.25)), float(cfg.get("zone_tolerance_pct", 0.002)))
    in_zone = _in_zone(last_close, z_low, z_high, tol)

    confluences = []
    if _ema50_confluence(df1, side, z_low, z_high, tol):
        confluences.append("EMA50")
    avwap = _anchored_vwap(df1, start_idx)
    if avwap is not None and _in_zone(avwap, z_low, z_high, tol):
        confluences.append("AVWAP")
    if _swap_zone_confluence(df1, side, z_low, z_high, tol):
        confluences.append("SWAP")

    min_conf = int(cfg.get("min_confluence", 1))
    require_confluence = bool(cfg.get("require_confluence", True))
    allow_weak_confluence = bool(cfg.get("allow_weak_confluence", False))
    weak_confluence = False
    if require_confluence and len(confluences) < min_conf:
        if not allow_weak_confluence:
            return None
        weak_confluence = True

    confirmation = False
    confirmation_pattern = None
    if in_zone:
        if side == "LONG":
            if _bullish_engulfing(df_confirm):
                confirmation = True
                confirmation_pattern = "bullish_engulfing"
        else:
            if _bearish_engulfing(df_confirm):
                confirmation = True
                confirmation_pattern = "bearish_engulfing"
        if confirmation:
            confirmation = _golden_respect(
                df_confirm,
                side,
                z_low,
                z_high,
                tol,
                atr,
                float(cfg.get("respect_body_atr_ratio", 0.6)),
            )
            if not confirmation:
                confirmation_pattern = None

    entry = last_close if confirmation and in_zone else (z_low + z_high) / 2.0
    sl_buffer = float(cfg.get("sl_atr_mult", 1.2))

    if side == "LONG":
        sl = lo - atr * sl_buffer
        tp1 = hi
        tp2 = _fib_extension_long(lo, hi, float(cfg.get("tp2_extension_level", -0.618)))
    else:
        sl = hi + atr * sl_buffer
        tp1 = lo
        tp2 = _fib_extension_short(hi, lo, float(cfg.get("tp2_extension_level", -0.618)))

    risk = abs(entry - sl)
    if risk <= 0:
        return None
    rr1 = abs(tp1 - entry) / risk
    rr2 = abs(tp2 - entry) / risk

    allow_setup = bool(cfg.get("allow_setup_if_no_confirm", True))
    allow_pre_zone = bool(cfg.get("allow_pre_zone", False))
    max_pre_zone_atr = float(cfg.get("max_pre_zone_atr", 1.5))
    dist_to_zone = min(abs(last_close - z_low), abs(last_close - z_high))
    near_zone = dist_to_zone <= atr * max_pre_zone_atr if np.isfinite(atr) else False
    if confirmation and in_zone:
        status = "OK"
        reason = f"{zone_label} + confluence + confirmation"
    elif allow_setup and in_zone:
        status = "SETUP"
        reason = f"{zone_label} confluence var, confirmation gözlənilir"
    elif allow_setup and allow_pre_zone and near_zone:
        status = "SETUP"
        reason = f"{zone_label} yaxınlığında (zone hələ touch etməyib)"
    else:
        return None

    score = rr2
    if weak_confluence:
        score *= 0.6
        reason = f"{reason} (Confluence zəif)"
    if not in_zone and near_zone:
        penalty = max(0.4, 1.0 - (dist_to_zone / (atr * max_pre_zone_atr)))
        score *= penalty

    return Analysis(
        status=status,
        side=side,
        entry=float(entry),
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        rr1=float(rr1),
        rr2=float(rr2),
        score=float(score),
        reason=f"{reason} ({', '.join(confluences)})",
        details={
            "zone_low": float(z_low),
            "zone_high": float(z_high),
            "in_zone": bool(in_zone),
            "near_zone": bool(near_zone),
            "dist_to_zone_atr": float(dist_to_zone / atr) if atr else None,
            "confirmation": bool(confirmation),
            "confirmation_pattern": confirmation_pattern,
            "confluences": confluences,
            "confluence_count": int(len(confluences)),
            "weak_confluence": bool(weak_confluence),
            "atr": float(atr),
            "fib_zone_mode": zone_mode,
            "fib_zone_label": zone_label,
            "strong_trend": bool(strong_trend),
        },
    )


def _build_measurement_setup(
    df15: pd.DataFrame,
    side: str,
    atr: float,
    cfg: Dict[str, Any],
) -> Optional[Analysis]:
    sweep_idx = _liquidity_sweep(df15, side)
    if sweep_idx is None:
        return None

    bos_idx = _break_of_structure(df15, side)
    if bos_idx is None or bos_idx <= sweep_idx:
        return None

    if side == "SHORT":
        impulse_high = float(df15.loc[sweep_idx:, "high"].max())
        impulse_low = float(df15.loc[bos_idx:, "low"].min())
        if impulse_high <= impulse_low:
            return None
        entry = _fib_retracement_short(impulse_high, impulse_low, 0.71)
        zone_low, zone_high = _zone_bounds(
            _fib_retracement_short(impulse_high, impulse_low, 0.71),
            _fib_retracement_short(impulse_high, impulse_low, 0.75),
        )
        sl = impulse_high
        tp = impulse_low
    elif side == "LONG":
        impulse_low = float(df15.loc[sweep_idx:, "low"].min())
        impulse_high = float(df15.loc[bos_idx:, "high"].max())
        if impulse_high <= impulse_low:
            return None
        entry = _fib_retracement(impulse_low, impulse_high, 0.71)
        zone_low, zone_high = _zone_bounds(
            _fib_retracement(impulse_low, impulse_high, 0.75),
            _fib_retracement(impulse_low, impulse_high, 0.71),
        )
        sl = impulse_low
        tp = impulse_high
    else:
        return None

    tol = _zone_tolerance(entry, atr, float(cfg.get("zone_tolerance_atr", 0.25)), float(cfg.get("zone_tolerance_pct", 0.002)))
    fvg = _find_fvg(df15, side)
    if fvg:
        fvg_low, fvg_high = _zone_bounds(fvg[0], fvg[1])
        fvg_overlap = _in_zone(entry, fvg_low, fvg_high, tol)
    else:
        fvg_overlap = False

    risk = abs(entry - sl)
    if risk <= 0:
        return None
    rr = abs(tp - entry) / risk

    status = "SETUP"
    reason = "71% measurement setup"
    if fvg_overlap:
        reason += " + FVG overlap"

    return Analysis(
        status=status,
        side=side,
        entry=float(entry),
        sl=float(sl),
        tp1=float(tp),
        tp2=float(tp),
        rr1=float(rr),
        rr2=float(rr),
        score=float(rr + (0.5 if fvg_overlap else 0.0)),
        reason=reason,
        details={
            "entry_zone_low": float(zone_low),
            "entry_zone_high": float(zone_high),
            "in_zone": True,
            "dist_to_zone_atr": 0.0,
            "fvg_overlap": bool(fvg_overlap),
            "confluence_count": int(1 if fvg_overlap else 0),
            "atr": float(atr),
            "sweep_idx": int(sweep_idx),
            "bos_idx": int(bos_idx),
        },
    )


def _fallback_bias(ema_bias: Optional[str], pd_bias: Optional[str], df_impulse: pd.DataFrame) -> Optional[str]:
    if ema_bias in ("LONG", "SHORT"):
        return ema_bias
    if pd_bias in ("LONG", "SHORT"):
        return pd_bias
    if len(df_impulse) >= 55:
        ema50 = _ema(df_impulse["close"], 50).iloc[-1]
        last = float(df_impulse["close"].iloc[-1])
        return "LONG" if last >= ema50 else "SHORT"
    return None


def _build_fallback_setup(
    df_impulse: pd.DataFrame,
    side: str,
    atr: float,
    min_rr2: float,
    cfg: Dict[str, Any],
) -> Optional[Analysis]:
    if atr <= 0 or not np.isfinite(atr):
        return None
    last_close = float(df_impulse["close"].iloc[-1])
    sl_buffer = float(cfg.get("sl_atr_mult", 1.2))
    risk = atr * sl_buffer
    if risk <= 0:
        return None
    rr2 = max(min_rr2, 2.0)
    if side == "LONG":
        entry = last_close
        sl = entry - risk
        tp1 = entry + risk
        tp2 = entry + risk * rr2
    else:
        entry = last_close
        sl = entry + risk
        tp1 = entry - risk
        tp2 = entry - risk * rr2
    return Analysis(
        status="SETUP",
        side=side,
        entry=float(entry),
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        rr1=1.0,
        rr2=float(rr2),
        score=float(rr2 * 0.4),
        reason="Fallback bias plan (setup tapılmadı, trend yönü ilə ehtiyatlı plan)",
        details={
            "fallback": True,
            "atr": float(atr),
            "in_zone": False,
            "dist_to_zone_atr": None,
        },
    )


def _micro_confirmation(
    df_micro: pd.DataFrame,
    side: str,
    zone_low: float,
    zone_high: float,
    tol: float,
) -> Tuple[bool, Optional[str]]:
    if df_micro.empty:
        return False, None
    last_close = float(df_micro["close"].iloc[-1])
    if not _in_zone(last_close, zone_low, zone_high, tol):
        return False, None
    if side == "LONG":
        if _bullish_engulfing(df_micro):
            return True, "bullish_engulfing"
        if _morning_star(df_micro):
            return True, "morning_star"
        if _rejection_wick(df_micro, "LONG", zone_low, zone_high):
            return True, "rejection_wick"
    else:
        if _bearish_engulfing(df_micro):
            return True, "bearish_engulfing"
        if _evening_star(df_micro):
            return True, "evening_star"
        if _rejection_wick(df_micro, "SHORT", zone_low, zone_high):
            return True, "rejection_wick"
    return False, None


def analyze_symbol(
    symbol: str,
    fetch_ohlcv: Callable[[str, str, int], pd.DataFrame],
    settings: Dict[str, Any],
    on_stage: Optional[Callable[[str], None]] = None,
) -> Analysis:
    scan_cfg = settings.get("scan", {})
    timeframes_cfg = settings.get("timeframes", {})

    def _limit_for(tf: str, fallback: int = 500) -> int:
        return int(timeframes_cfg.get(tf, {}).get("limit", scan_cfg.get(f"limit_{tf}", fallback)))

    risk_cfg = settings.get("risk", {})
    min_rr2 = float(risk_cfg.get("min_rr2", 3.0))
    max_entry_atr = float(risk_cfg.get("max_entry_distance_atr", 2.0))

    strategy_cfg = settings.get("strategy", {})
    htf_lookback = int(strategy_cfg.get("htf_range_lookback", 200))
    htf_timeframes = strategy_cfg.get("htf_bias_timeframes", ["1d", "4h"])
    if isinstance(htf_timeframes, str):
        htf_timeframes = [htf_timeframes]
    if not htf_timeframes:
        htf_timeframes = ["1d", "4h"]
    htf_min_alignment = float(strategy_cfg.get("htf_min_alignment", 0.6))
    impulse_tf = strategy_cfg.get("impulse_timeframe", "1h")
    confirm_tf = strategy_cfg.get("confirm_timeframe", "15m")
    measurement_tf = strategy_cfg.get("measurement_timeframe", confirm_tf)
    micro_tf = strategy_cfg.get("micro_confirm_timeframe")

    tf_cache: Dict[str, pd.DataFrame] = {}

    def _fetch_tf(tf: str) -> pd.DataFrame:
        if tf not in tf_cache:
            if on_stage:
                on_stage(f"fetch {tf.upper()}")
            tf_cache[tf] = fetch_ohlcv(symbol, tf, _limit_for(tf))
        return tf_cache[tf]

    ema_biases: Dict[str, Optional[str]] = {}
    pd_biases: Dict[str, Optional[str]] = {}
    for tf in htf_timeframes:
        df_htf = _fetch_tf(tf)
        ema_biases[tf] = _trend_bias_ema(df_htf)
        pd_biases[tf] = _premium_discount_bias(df_htf, htf_lookback)

    ema_bias, ema_counts, ema_ratio = _aggregate_bias(ema_biases, htf_min_alignment)
    pd_bias, pd_counts, pd_ratio = _aggregate_bias(pd_biases, htf_min_alignment)

    df_impulse = _fetch_tf(impulse_tf)
    df_confirm = _fetch_tf(confirm_tf)
    df_measure = _fetch_tf(measurement_tf)

    atr1 = _atr(df_impulse, 14)
    if not np.isfinite(atr1) or atr1 <= 0:
        return Analysis(status="NO_TRADE", side="-", reason="ATR hesablanmadı")

    candidates: List[Analysis] = []

    golden_biases = [ema_bias] if ema_bias in ("LONG", "SHORT") else ["LONG", "SHORT"]
    measurement_biases = [pd_bias] if pd_bias in ("LONG", "SHORT") else ["LONG", "SHORT"]

    for bias in golden_biases:
        golden = _build_golden_zone_setup(df_impulse, df_confirm, bias, atr1, strategy_cfg)
        if golden:
            candidates.append(golden)

    if bool(strategy_cfg.get("allow_measurement_setup", True)):
        for bias in measurement_biases:
            measurement = _build_measurement_setup(df_measure, bias, atr1, strategy_cfg)
            if measurement:
                candidates.append(measurement)

    eligible = [c for c in candidates if c.rr2 >= min_rr2]
    best: Optional[Analysis] = None
    if eligible:
        best = max(eligible, key=lambda x: x.score)
    else:
        fallback_side = _fallback_bias(ema_bias, pd_bias, df_impulse)
        if fallback_side:
            fallback = _build_fallback_setup(df_impulse, fallback_side, atr1, min_rr2, strategy_cfg)
            if fallback and fallback.rr2 >= min_rr2:
                if fallback.details is not None:
                    fallback.details = {
                        **fallback.details,
                        "ema_bias": ema_bias,
                        "premium_discount_bias": pd_bias,
                        "impulse_timeframe": impulse_tf,
                        "confirm_timeframe": confirm_tf,
                        "measurement_timeframe": measurement_tf,
                    }
                best = fallback
        if best is None:
            return Analysis(
                status="NO_TRADE",
                side="-",
                reason=f"RR hədəfi {min_rr2:.2f} qarşılanmadı (3x1 qaydası)",
            )

    last_close = float(df_impulse["close"].iloc[-1])

    # Enforce strict 3:1 RR if enabled
    enforce_strict = bool(risk_cfg.get("enforce_strict_rr", False))
    rr_buffer = float(risk_cfg.get("rr_buffer_pct", 1.0))
    if enforce_strict and best.side in ("LONG", "SHORT"):
        risk_dist = abs(best.entry - best.sl)
        if risk_dist > 0:
            direction = 1 if best.side == "LONG" else -1
            # Recalculate TP2 based on 3x risk * buffer
            target_rr = 3.0 * rr_buffer
            best.tp2 = float(best.entry + direction * (risk_dist * target_rr))
            # Recalculate RR2
            best.rr2 = float(abs(best.tp2 - best.entry) / risk_dist)
            # Also adjust TP1 to 1:1 if it was further
            if abs(best.tp1 - best.entry) / risk_dist > 1.0:
                best.tp1 = float(best.entry + direction * risk_dist)
                best.rr1 = 1.0
            best.reason = f"{best.reason} | Strict 3:1 RR applied (buf={rr_buffer})"

    dist_atr = abs(last_close - best.entry) / atr1
    penalty = 1.0
    if dist_atr > max_entry_atr:
        penalty *= 0.6
        best.reason = f"{best.reason} | Entry uzaqdır ({dist_atr:.2f} ATR)"
        best.status = "SETUP"

    if on_stage:
        on_stage("score")

    micro_confirmation = False
    micro_pattern = None
    zone_low = None
    zone_high = None
    if best.details is not None:
        zone_low = best.details.get("zone_low", best.details.get("entry_zone_low"))
        zone_high = best.details.get("zone_high", best.details.get("entry_zone_high"))
    if micro_tf and zone_low is not None and zone_high is not None:
        df_micro = _fetch_tf(str(micro_tf))
        tol = _zone_tolerance(
            float(df_impulse["close"].iloc[-1]),
            atr1,
            float(strategy_cfg.get("zone_tolerance_atr", 0.25)),
            float(strategy_cfg.get("zone_tolerance_pct", 0.002)),
        )
        micro_confirmation, micro_pattern = _micro_confirmation(
            df_micro,
            best.side,
            float(zone_low),
            float(zone_high),
            tol,
        )

    scoring_cfg = settings.get("scoring", {})
    w_rr2 = float(scoring_cfg.get("w_rr2", 10.0))
    w_trend = float(scoring_cfg.get("w_trend", 5.0))
    w_trend_strength = float(scoring_cfg.get("w_trend_strength", 3.0))
    w_conf = float(scoring_cfg.get("w_confluence", 3.0))
    w_conf_count = float(scoring_cfg.get("w_confluence_count", 1.0))
    w_confirmation = float(scoring_cfg.get("w_confirmation", 2.0))
    w_micro_confirmation = float(scoring_cfg.get("w_micro_confirmation", 1.5))
    w_alignment = float(scoring_cfg.get("w_alignment", 2.0))
    w_entry_distance = float(scoring_cfg.get("w_entry_distance", 1.0))
    w_zone = float(scoring_cfg.get("w_zone", 2.0))
    w_zone_balance = float(scoring_cfg.get("w_zone_balance", 1.0))
    w_status = float(scoring_cfg.get("w_status", 2.0))
    w_momentum = float(scoring_cfg.get("w_momentum", 2.0))
    w_volume = float(scoring_cfg.get("w_volume", 1.5))
    w_volatility = float(scoring_cfg.get("w_volatility", 1.5))
    w_liquidity = float(scoring_cfg.get("w_liquidity", 1.0))
    rr2_target = float(scoring_cfg.get("rr2_target", 6.0))
    confluence_target = int(scoring_cfg.get("confluence_target", 3))
    min_setup_quality_pct = float(scoring_cfg.get("min_setup_quality_pct", 55.0))
    rsi_period = int(scoring_cfg.get("rsi_period", 14))
    volume_lookback = int(scoring_cfg.get("volume_lookback", 30))
    volume_ratio_min = float(scoring_cfg.get("volume_ratio_min", 0.6))
    volume_ratio_max = float(scoring_cfg.get("volume_ratio_max", 2.0))
    volatility_lookback = int(scoring_cfg.get("volatility_lookback", 120))
    volatility_target_pctile = float(scoring_cfg.get("volatility_target_pctile", 0.6))

    trend_score = 0.0
    if best.side == ema_bias:
        trend_score = 1.0
    elif ema_bias in ("RANGE", None):
        trend_score = 0.5

    confluence_count = 0
    confirmation_score = 0.0
    micro_confirmation_score = 1.0 if micro_confirmation else 0.0
    if best.details:
        confluence_count = int(best.details.get("confluence_count", 0))
        if best.details.get("confirmation"):
            confirmation_score = 1.0
        elif best.details.get("fvg_overlap"):
            confirmation_score = 0.5

    alignment_ratio = (float(ema_ratio) + float(pd_ratio)) / 2.0
    alignment_score = 0.0
    if htf_min_alignment < 1.0:
        alignment_score = max(0.0, min(1.0, (alignment_ratio - htf_min_alignment) / (1.0 - htf_min_alignment)))
    entry_distance_score = 1.0 - min(1.0, dist_atr / max_entry_atr) if max_entry_atr > 0 else 0.0
    rr2_score = _clamp(best.rr2 / max(rr2_target, 1e-6))
    min_conf = int(strategy_cfg.get("min_confluence", 1))
    confluence_presence = 1.0 if confluence_count >= min_conf else 0.5 if confluence_count > 0 else 0.0
    confluence_score = _clamp(confluence_count / max(1, confluence_target))
    status_score = 1.0 if best.status == "OK" else 0.6

    zone_score = 0.0
    zone_balance_score = 0.0
    dist_to_zone_atr = None
    in_zone = False
    zone_low = None
    zone_high = None
    if best.details:
        in_zone = bool(best.details.get("in_zone", False))
        dist_to_zone_atr = best.details.get("dist_to_zone_atr")
        zone_low = best.details.get("zone_low", best.details.get("entry_zone_low"))
        zone_high = best.details.get("zone_high", best.details.get("entry_zone_high"))
    if in_zone:
        zone_score = 1.0
    elif dist_to_zone_atr is not None:
        max_pre_zone_atr = float(strategy_cfg.get("max_pre_zone_atr", 1.5))
        if max_pre_zone_atr > 0:
            zone_score = _clamp(1.0 - (float(dist_to_zone_atr) / max_pre_zone_atr))
    if in_zone and zone_low is not None and zone_high is not None:
        mid = (float(zone_low) + float(zone_high)) / 2.0
        half_range = max(abs(float(zone_high) - float(zone_low)) / 2.0, 1e-9)
        zone_balance_score = _clamp(1.0 - abs(last_close - mid) / half_range)

    trend_strength_score = _trend_strength_score(
        df_impulse,
        best.side,
        atr1,
        int(strategy_cfg.get("strong_trend_slope_lookback", 8)),
        float(strategy_cfg.get("strong_trend_slope_atr", 0.12)),
        float(strategy_cfg.get("strong_trend_ema_separation_atr", 0.3)),
    )

    rsi_value = _rsi(df_impulse["close"], rsi_period)
    if best.side == "LONG":
        momentum_score = _clamp((rsi_value - 50.0) / 20.0) if np.isfinite(rsi_value) else 0.0
    else:
        momentum_score = _clamp((50.0 - rsi_value) / 20.0) if np.isfinite(rsi_value) else 0.0

    volume_score = 0.0
    volume_ratio = float("nan")
    if len(df_impulse) >= max(2, volume_lookback):
        vol_series = df_impulse["volume"].tail(volume_lookback)
        vol_avg = float(vol_series.mean()) if not vol_series.empty else 0.0
        last_volume = float(df_impulse["volume"].iloc[-1])
        if vol_avg > 0:
            volume_ratio = last_volume / vol_avg
            volume_score = _clamp((volume_ratio - volume_ratio_min) / max(volume_ratio_max - volume_ratio_min, 1e-6))

    volatility_score = 0.0
    volatility_pctile = 0.0
    atr_series = _atr_series(df_impulse, 14)
    if not atr_series.empty:
        atr_pct = atr_series / df_impulse["close"].astype(float)
        window = atr_pct.dropna().tail(volatility_lookback)
        if not window.empty:
            last_atr_pct = float(window.iloc[-1])
            volatility_pctile = _percentile_rank(window, last_atr_pct)
            scale = max(volatility_target_pctile, 1.0 - volatility_target_pctile, 1e-6)
            volatility_score = _clamp(1.0 - abs(volatility_pctile - volatility_target_pctile) / scale)

    liquidity_score = 0.0
    liquidity_pctile = 0.0
    quote_volume_series = (df_impulse["volume"].astype(float) * df_impulse["close"].astype(float)).dropna()
    if not quote_volume_series.empty:
        window = quote_volume_series.tail(volatility_lookback)
        last_qv = float(window.iloc[-1])
        liquidity_pctile = _percentile_rank(window, last_qv)
        liquidity_score = _clamp(liquidity_pctile)

    total_weight = (
        w_rr2
        + w_trend
        + w_trend_strength
        + w_conf
        + w_conf_count
        + w_confirmation
        + w_micro_confirmation
        + w_alignment
        + w_entry_distance
        + w_zone
        + w_zone_balance
        + w_status
        + w_momentum
        + w_volume
        + w_volatility
        + w_liquidity
    )
    weighted_sum = (
        rr2_score * w_rr2
        + trend_score * w_trend
        + trend_strength_score * w_trend_strength
        + confluence_presence * w_conf
        + confluence_score * w_conf_count
        + confirmation_score * w_confirmation
        + micro_confirmation_score * w_micro_confirmation
        + alignment_score * w_alignment
        + entry_distance_score * w_entry_distance
        + zone_score * w_zone
        + zone_balance_score * w_zone_balance
        + status_score * w_status
        + momentum_score * w_momentum
        + volume_score * w_volume
        + volatility_score * w_volatility
        + liquidity_score * w_liquidity
    )
    quality_score = 0.0
    if total_weight > 0:
        quality_score = (weighted_sum / total_weight) * 100.0
    quality_score *= penalty

    best.score = float(quality_score)
    if best.details is not None:
        best.details = {
            **best.details,
            "ema_bias": ema_bias,
            "premium_discount_bias": pd_bias,
            "entry_distance_atr": float(dist_atr),
            "filters": {
                "min_rr2": min_rr2,
                "max_entry_distance_atr": max_entry_atr,
                "penalty_multiplier": penalty,
            },
            "score_meta": {
                "rr2_target": rr2_target,
                "confluence_target": confluence_target,
                "min_setup_quality_pct": min_setup_quality_pct,
                "rsi_period": rsi_period,
                "volume_lookback": volume_lookback,
                "volume_ratio_min": volume_ratio_min,
                "volume_ratio_max": volume_ratio_max,
                "volatility_lookback": volatility_lookback,
                "volatility_target_pctile": volatility_target_pctile,
                "weight_total": total_weight,
            },
            "htf_bias_timeframes": htf_timeframes,
            "htf_bias_alignment_min": htf_min_alignment,
            "htf_bias_ema_votes": ema_biases,
            "htf_bias_pd_votes": pd_biases,
            "htf_bias_ema_counts": ema_counts,
            "htf_bias_pd_counts": pd_counts,
            "htf_bias_ema_ratio": float(ema_ratio),
            "htf_bias_pd_ratio": float(pd_ratio),
            "impulse_timeframe": impulse_tf,
            "confirm_timeframe": confirm_tf,
            "measurement_timeframe": measurement_tf,
            "micro_confirm_timeframe": micro_tf,
            "micro_confirmation": bool(micro_confirmation),
            "micro_confirmation_pattern": micro_pattern,
            "score_components": {
                "rr2": {
                    "raw": float(best.rr2),
                    "normalized": rr2_score,
                    "weighted": rr2_score * w_rr2,
                },
                "trend": {
                    "raw": best.side,
                    "normalized": trend_score,
                    "weighted": trend_score * w_trend,
                },
                "trend_strength": {
                    "normalized": trend_strength_score,
                    "weighted": trend_strength_score * w_trend_strength,
                },
                "momentum": {
                    "raw": float(rsi_value) if np.isfinite(rsi_value) else None,
                    "normalized": momentum_score,
                    "weighted": momentum_score * w_momentum,
                },
                "volume_activity": {
                    "raw": float(volume_ratio) if np.isfinite(volume_ratio) else None,
                    "normalized": volume_score,
                    "weighted": volume_score * w_volume,
                },
                "volatility_regime": {
                    "raw": float(volatility_pctile),
                    "normalized": volatility_score,
                    "weighted": volatility_score * w_volatility,
                },
                "liquidity": {
                    "raw": float(liquidity_pctile),
                    "normalized": liquidity_score,
                    "weighted": liquidity_score * w_liquidity,
                },
                "confluence_presence": {
                    "raw": confluence_count,
                    "normalized": confluence_presence,
                    "weighted": confluence_presence * w_conf,
                },
                "confluence_count": {
                    "raw": confluence_count,
                    "normalized": confluence_score,
                    "weighted": confluence_score * w_conf_count,
                },
                "confirmation": {
                    "normalized": confirmation_score,
                    "weighted": confirmation_score * w_confirmation,
                },
                "micro_confirmation": {
                    "normalized": micro_confirmation_score,
                    "weighted": micro_confirmation_score * w_micro_confirmation,
                },
                "alignment": {
                    "raw": alignment_ratio,
                    "normalized": alignment_score,
                    "weighted": alignment_score * w_alignment,
                },
                "entry_distance": {
                    "raw": dist_atr,
                    "normalized": entry_distance_score,
                    "weighted": entry_distance_score * w_entry_distance,
                },
                "zone_proximity": {
                    "normalized": zone_score,
                    "weighted": zone_score * w_zone,
                },
                "zone_balance": {
                    "normalized": zone_balance_score,
                    "weighted": zone_balance_score * w_zone_balance,
                },
                "status": {
                    "raw": best.status,
                    "normalized": status_score,
                    "weighted": status_score * w_status,
                },
                "quality_pct": float(quality_score),
            },
        }

    force_best_plan = bool(strategy_cfg.get("force_best_plan", True))
    if best.status == "SETUP" and quality_score < min_setup_quality_pct:
        if force_best_plan:
            best.reason = (
                f"{best.reason} | Setup keyfiyyəti {quality_score:.1f}% < tələb olunan "
                f"{min_setup_quality_pct:.1f}% (force_best_plan=ON)"
            )
            if best.details is not None:
                best.details = {
                    **best.details,
                    "quality_gate": {
                        "min_setup_quality_pct": float(min_setup_quality_pct),
                        "quality_pct": float(quality_score),
                        "force_best_plan": True,
                    },
                }
        else:
            best.status = "NO_TRADE"
            best.side = "-"
            best.reason = f"Setup keyfiyyəti {quality_score:.1f}% < tələb olunan {min_setup_quality_pct:.1f}%"

    return best
