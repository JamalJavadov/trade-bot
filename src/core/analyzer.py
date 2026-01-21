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


def _zone_bounds(a: float, b: float) -> Tuple[float, float]:
    return (min(a, b), max(a, b))


def _zone_tolerance(price: float, atr: float, atr_mult: float, pct: float) -> float:
    atr_part = atr * atr_mult if np.isfinite(atr) else 0.0
    pct_part = price * pct
    return max(atr_part, pct_part)


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


def _golden_respect(df: pd.DataFrame, side: str, zone_low: float, zone_high: float, tol: float) -> bool:
    recent = df.tail(3)
    if recent.empty:
        return False
    if side == "LONG":
        return bool((recent["close"] >= zone_low - tol).all())
    return bool((recent["close"] <= zone_high + tol).all())


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
    if side == "LONG":
        lo, hi = a, b
        z_low, z_high = _zone_bounds(
            _fib_retracement(lo, hi, 0.618),
            _fib_retracement(lo, hi, 0.5),
        )
    elif side == "SHORT":
        hi, lo = a, b
        z_low, z_high = _zone_bounds(
            _fib_retracement_short(hi, lo, 0.5),
            _fib_retracement_short(hi, lo, 0.618),
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
            elif _morning_star(df_confirm):
                confirmation = True
                confirmation_pattern = "morning_star"
            elif _rejection_wick(df_confirm, "LONG", z_low, z_high):
                confirmation = True
                confirmation_pattern = "rejection_wick"
        else:
            if _bearish_engulfing(df_confirm):
                confirmation = True
                confirmation_pattern = "bearish_engulfing"
            elif _evening_star(df_confirm):
                confirmation = True
                confirmation_pattern = "evening_star"
            elif _rejection_wick(df_confirm, "SHORT", z_low, z_high):
                confirmation = True
                confirmation_pattern = "rejection_wick"
        if confirmation:
            confirmation = _golden_respect(df_confirm, side, z_low, z_high, tol)
            if not confirmation:
                confirmation_pattern = None

    entry = last_close if confirmation and in_zone else (z_low + z_high) / 2.0
    sl_buffer = float(cfg.get("sl_atr_mult", 1.2))

    if side == "LONG":
        sl = lo - atr * sl_buffer
        tp1 = hi
        tp2 = _fib_extension_long(lo, hi, -0.272)
    else:
        sl = hi + atr * sl_buffer
        tp1 = lo
        tp2 = _fib_extension_short(hi, lo, -0.272)

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
        reason = "Golden zone + confluence + confirmation"
    elif allow_setup and in_zone:
        status = "SETUP"
        reason = "Golden zone confluence var, confirmation gözlənilir"
    elif allow_setup and allow_pre_zone and near_zone:
        status = "SETUP"
        reason = "Golden zone yaxınlığında (zone hələ touch etməyib)"
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
    min_rr2 = float(risk_cfg.get("min_rr2", 2.0))
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

    for bias in measurement_biases:
        measurement = _build_measurement_setup(df_measure, bias, atr1, strategy_cfg)
        if measurement:
            candidates.append(measurement)

    if not candidates:
        fallback_side = _fallback_bias(ema_bias, pd_bias, df_impulse)
        if fallback_side:
            fallback = _build_fallback_setup(df_impulse, fallback_side, atr1, min_rr2, strategy_cfg)
            if fallback:
                if fallback.details is not None:
                    fallback.details = {
                        **fallback.details,
                        "ema_bias": ema_bias,
                        "premium_discount_bias": pd_bias,
                        "impulse_timeframe": impulse_tf,
                        "confirm_timeframe": confirm_tf,
                        "measurement_timeframe": measurement_tf,
                    }
                return fallback
        return Analysis(status="NO_TRADE", side="-", reason="Uyğun setup tapılmadı (Codex filtrləri)")

    best = max(candidates, key=lambda x: x.score)

    dist_atr = abs(float(df_impulse["close"].iloc[-1]) - best.entry) / atr1
    penalty = 1.0
    if dist_atr > max_entry_atr:
        penalty *= 0.6
        best.reason = f"{best.reason} | Entry uzaqdır ({dist_atr:.2f} ATR)"
        best.status = "SETUP"

    if best.rr2 < min_rr2:
        penalty *= 0.7
        best.reason = f"{best.reason} | RR aşağıdır ({best.rr2:.2f} < {min_rr2})"
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
    w_conf = float(scoring_cfg.get("w_confluence", 3.0))
    w_conf_count = float(scoring_cfg.get("w_confluence_count", 1.0))
    w_confirmation = float(scoring_cfg.get("w_confirmation", 2.0))
    w_micro_confirmation = float(scoring_cfg.get("w_micro_confirmation", 1.5))

    trend_bonus = 1.0 if best.side == ema_bias else 0.5
    conf_bonus = 1.0 if "(" in best.reason else 0.0
    if best.details and best.details.get("fallback"):
        conf_bonus = 0.0
    confluence_count = 0
    confirmation_bonus = 0.0
    micro_confirmation_bonus = 1.0 if micro_confirmation else 0.0
    if best.details:
        confluence_count = int(best.details.get("confluence_count", 0))
        if best.details.get("confirmation"):
            confirmation_bonus = 1.0
        elif best.details.get("fvg_overlap"):
            confirmation_bonus = 0.5

    best.score = (
        best.rr2 * w_rr2
        + trend_bonus * w_trend
        + conf_bonus * w_conf
        + confluence_count * w_conf_count
        + confirmation_bonus * w_confirmation
        + micro_confirmation_bonus * w_micro_confirmation
    ) * penalty
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
                "rr2": best.rr2 * w_rr2,
                "trend": trend_bonus * w_trend,
                "confluence": conf_bonus * w_conf,
                "confluence_count": confluence_count * w_conf_count,
                "confirmation": confirmation_bonus * w_confirmation,
                "micro_confirmation": micro_confirmation_bonus * w_micro_confirmation,
            },
        }

    return best
