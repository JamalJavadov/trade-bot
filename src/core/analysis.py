from __future__ import annotations
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr, tr2, tr3], axis=1).max(axis=1)
    v = true_range.rolling(period).mean().iloc[-1]
    return float(v) if pd.notna(v) else 0.0


def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
    highs = df["high"].values
    lows = df["low"].values
    sh, sl = [], []
    n = len(df)
    for i in range(left, n - right):
        if highs[i] == max(highs[i - left:i + right + 1]):
            sh.append(i)
        if lows[i] == min(lows[i - left:i + right + 1]):
            sl.append(i)
    return sh, sl


def bias_from_4h(df4h: pd.DataFrame) -> str:
    c = df4h["close"]
    e50 = ema(c, 50).iloc[-1]
    e200 = ema(c, 200).iloc[-1] if len(c) >= 200 else ema(c, 100).iloc[-1]
    if e50 > e200:
        return "LONG"
    if e50 < e200:
        return "SHORT"
    return "NO_TRADE"


def fib_zone(high: float, low: float) -> tuple[float, float]:
    # 0.5–0.618 golden zone
    diff = high - low
    z50 = high - diff * 0.5
    z618 = high - diff * 0.618
    return (min(z50, z618), max(z50, z618))
