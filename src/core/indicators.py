"""
Professional Technical Indicators Module

This module provides institutional-grade technical analysis indicators
for deep market analysis. Each indicator includes signal interpretation
and normalized scoring capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class IndicatorResult:
    """Standardized result from indicator calculation."""
    value: float
    signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0.0 to 1.0 normalized strength
    details: Optional[Dict[str, Any]] = None


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(period).mean()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp value between low and high."""
    return max(low, min(high, value))


# =============================================================================
# MACD (Moving Average Convergence Divergence)
# =============================================================================

@dataclass
class MACDResult:
    """MACD calculation result."""
    macd_line: float
    signal_line: float
    histogram: float
    signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0.0 to 1.0
    crossover: Optional[str]  # "BULLISH_CROSS", "BEARISH_CROSS", None
    divergence: Optional[str]  # "BULLISH_DIV", "BEARISH_DIV", None


def calculate_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    price_col: str = "close",
) -> Optional[MACDResult]:
    """
    Calculate MACD with signal interpretation.
    
    Professional-grade MACD includes:
    - MACD line (fast EMA - slow EMA)
    - Signal line (EMA of MACD line)
    - Histogram (MACD - Signal)
    - Crossover detection
    - Divergence detection
    """
    if len(df) < slow_period + signal_period:
        return None
    
    prices = df[price_col].astype(float)
    
    # Calculate MACD components
    ema_fast = _ema(prices, fast_period)
    ema_slow = _ema(prices, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    # Get current and previous values
    curr_macd = float(macd_line.iloc[-1])
    curr_signal = float(signal_line.iloc[-1])
    curr_hist = float(histogram.iloc[-1])
    prev_hist = float(histogram.iloc[-2]) if len(histogram) >= 2 else curr_hist
    prev_macd = float(macd_line.iloc[-2]) if len(macd_line) >= 2 else curr_macd
    prev_signal = float(signal_line.iloc[-2]) if len(signal_line) >= 2 else curr_signal
    
    # Determine signal
    if curr_hist > 0:
        signal = "BULLISH"
    elif curr_hist < 0:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"
    
    # Calculate strength based on histogram momentum
    hist_series = histogram.tail(20).dropna()
    if not hist_series.empty:
        hist_max = float(hist_series.abs().max())
        if hist_max > 0:
            strength = _clamp(abs(curr_hist) / hist_max)
        else:
            strength = 0.0
    else:
        strength = 0.0
    
    # Detect crossover
    crossover = None
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        crossover = "BULLISH_CROSS"
    elif prev_macd >= prev_signal and curr_macd < curr_signal:
        crossover = "BEARISH_CROSS"
    
    # Detect divergence (simplified)
    divergence = None
    lookback = 14
    if len(df) >= lookback:
        price_change = float(prices.iloc[-1]) - float(prices.iloc[-lookback])
        macd_change = curr_macd - float(macd_line.iloc[-lookback])
        
        if price_change > 0 and macd_change < 0:
            divergence = "BEARISH_DIV"
        elif price_change < 0 and macd_change > 0:
            divergence = "BULLISH_DIV"
    
    return MACDResult(
        macd_line=curr_macd,
        signal_line=curr_signal,
        histogram=curr_hist,
        signal=signal,
        strength=strength,
        crossover=crossover,
        divergence=divergence,
    )


# =============================================================================
# Bollinger Bands
# =============================================================================

@dataclass
class BollingerResult:
    """Bollinger Bands calculation result."""
    upper: float
    middle: float
    lower: float
    bandwidth: float  # (upper - lower) / middle
    percent_b: float  # (price - lower) / (upper - lower)
    signal: str  # "OVERBOUGHT", "OVERSOLD", "SQUEEZE", "EXPANSION", "NEUTRAL"
    strength: float
    squeeze_active: bool


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    price_col: str = "close",
) -> Optional[BollingerResult]:
    """
    Calculate Bollinger Bands with professional analysis.
    
    Includes:
    - Upper/Middle/Lower bands
    - Bandwidth (volatility measure)
    - %B (position within bands)
    - Squeeze detection (low volatility breakout setup)
    """
    if len(df) < period:
        return None
    
    prices = df[price_col].astype(float)
    
    # Calculate bands
    middle = _sma(prices, period)
    std = prices.rolling(period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # Current values
    curr_price = float(prices.iloc[-1])
    curr_upper = float(upper.iloc[-1])
    curr_middle = float(middle.iloc[-1])
    curr_lower = float(lower.iloc[-1])
    
    # Calculate bandwidth and %B
    if curr_middle > 0:
        bandwidth = (curr_upper - curr_lower) / curr_middle
    else:
        bandwidth = 0.0
    
    band_range = curr_upper - curr_lower
    if band_range > 0:
        percent_b = (curr_price - curr_lower) / band_range
    else:
        percent_b = 0.5
    
    # Detect squeeze (bandwidth is low relative to recent history)
    bandwidth_series = ((upper - lower) / middle).tail(50).dropna()
    squeeze_active = False
    if not bandwidth_series.empty:
        bandwidth_pctile = float((bandwidth_series <= bandwidth).mean())
        squeeze_active = bandwidth_pctile <= 0.2  # Bottom 20% = squeeze
    
    # Determine signal
    if percent_b >= 1.0:
        signal = "OVERBOUGHT"
        strength = _clamp((percent_b - 1.0) * 2)
    elif percent_b <= 0.0:
        signal = "OVERSOLD"
        strength = _clamp(abs(percent_b) * 2)
    elif squeeze_active:
        signal = "SQUEEZE"
        strength = 0.8
    elif not bandwidth_series.empty and bandwidth > float(bandwidth_series.quantile(0.8)):
        signal = "EXPANSION"
        strength = _clamp((bandwidth - float(bandwidth_series.median())) / float(bandwidth_series.std() + 1e-9))
    else:
        signal = "NEUTRAL"
        strength = _clamp(1.0 - abs(percent_b - 0.5) * 2)
    
    return BollingerResult(
        upper=curr_upper,
        middle=curr_middle,
        lower=curr_lower,
        bandwidth=bandwidth,
        percent_b=percent_b,
        signal=signal,
        strength=strength,
        squeeze_active=squeeze_active,
    )


# =============================================================================
# Stochastic RSI
# =============================================================================

@dataclass
class StochRSIResult:
    """Stochastic RSI calculation result."""
    k: float  # Fast line (0-100)
    d: float  # Slow line (0-100)
    signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    strength: float
    crossover: Optional[str]  # "BULLISH_CROSS", "BEARISH_CROSS", None


def _calculate_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI as a series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stoch_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
    price_col: str = "close",
) -> Optional[StochRSIResult]:
    """
    Calculate Stochastic RSI for precision overbought/oversold detection.
    
    StochRSI = (RSI - RSI_Low) / (RSI_High - RSI_Low)
    
    More sensitive than regular RSI for catching turns.
    """
    min_len = rsi_period + stoch_period + max(smooth_k, smooth_d)
    if len(df) < min_len:
        return None
    
    prices = df[price_col].astype(float)
    
    # Calculate RSI
    rsi = _calculate_rsi_series(prices, rsi_period)
    
    # Calculate Stochastic of RSI
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min
    
    stoch_rsi = ((rsi - rsi_min) / rsi_range.replace(0, np.nan)) * 100
    
    # Smooth K and D lines
    k_line = stoch_rsi.rolling(smooth_k).mean()
    d_line = k_line.rolling(smooth_d).mean()
    
    # Get current values
    curr_k = float(k_line.iloc[-1])
    curr_d = float(d_line.iloc[-1])
    prev_k = float(k_line.iloc[-2]) if len(k_line) >= 2 else curr_k
    prev_d = float(d_line.iloc[-2]) if len(d_line) >= 2 else curr_d
    
    if not np.isfinite(curr_k) or not np.isfinite(curr_d):
        return None
    
    # Determine signal
    if curr_k >= 80:
        signal = "OVERBOUGHT"
        strength = _clamp((curr_k - 80) / 20)
    elif curr_k <= 20:
        signal = "OVERSOLD"
        strength = _clamp((20 - curr_k) / 20)
    else:
        signal = "NEUTRAL"
        strength = _clamp(1.0 - abs(curr_k - 50) / 50)
    
    # Detect crossover
    crossover = None
    if prev_k <= prev_d and curr_k > curr_d:
        crossover = "BULLISH_CROSS"
    elif prev_k >= prev_d and curr_k < curr_d:
        crossover = "BEARISH_CROSS"
    
    return StochRSIResult(
        k=curr_k,
        d=curr_d,
        signal=signal,
        strength=strength,
        crossover=crossover,
    )


# =============================================================================
# ADX (Average Directional Index)
# =============================================================================

@dataclass
class ADXResult:
    """ADX calculation result."""
    adx: float  # Trend strength (0-100)
    plus_di: float  # +DI
    minus_di: float  # -DI
    signal: str  # "STRONG_TREND", "WEAK_TREND", "NO_TREND"
    trend_direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> Optional[ADXResult]:
    """
    Calculate ADX (Average Directional Index) for trend strength measurement.
    
    ADX:
    - > 25: Strong trend
    - 20-25: Possible trend starting
    - < 20: Weak/no trend
    
    +DI > -DI: Bullish trend
    -DI > +DI: Bearish trend
    """
    if len(df) < period * 2:
        return None
    
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth with Wilder's smoothing
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    
    # Get current values
    curr_adx = float(adx.iloc[-1])
    curr_plus_di = float(plus_di.iloc[-1])
    curr_minus_di = float(minus_di.iloc[-1])
    
    if not np.isfinite(curr_adx):
        return None
    
    # Determine trend strength signal
    if curr_adx >= 40:
        signal = "STRONG_TREND"
        strength = _clamp((curr_adx - 25) / 50)
    elif curr_adx >= 25:
        signal = "WEAK_TREND"
        strength = _clamp((curr_adx - 15) / 25)
    else:
        signal = "NO_TREND"
        strength = _clamp(curr_adx / 25)
    
    # Determine trend direction
    if curr_plus_di > curr_minus_di:
        trend_direction = "BULLISH"
    elif curr_minus_di > curr_plus_di:
        trend_direction = "BEARISH"
    else:
        trend_direction = "NEUTRAL"
    
    return ADXResult(
        adx=curr_adx,
        plus_di=curr_plus_di,
        minus_di=curr_minus_di,
        signal=signal,
        trend_direction=trend_direction,
        strength=strength,
    )


# =============================================================================
# OBV (On-Balance Volume)
# =============================================================================

@dataclass
class OBVResult:
    """OBV calculation result."""
    obv: float
    obv_ema: float
    trend: str  # "ACCUMULATION", "DISTRIBUTION", "NEUTRAL"
    divergence: Optional[str]  # "BULLISH_DIV", "BEARISH_DIV", None
    strength: float


def calculate_obv(
    df: pd.DataFrame,
    ema_period: int = 21,
) -> Optional[OBVResult]:
    """
    Calculate On-Balance Volume for volume-price confirmation.
    
    OBV accumulates volume based on price direction:
    - Price up: Add volume
    - Price down: Subtract volume
    
    Divergence between OBV and price indicates potential reversals.
    """
    if len(df) < ema_period + 5:
        return None
    
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    
    # Calculate OBV
    price_change = close.diff()
    obv = volume.where(price_change > 0, -volume).where(price_change != 0, 0).cumsum()
    
    # Calculate OBV EMA for trend
    obv_ema = _ema(obv, ema_period)
    
    # Get current values
    curr_obv = float(obv.iloc[-1])
    curr_obv_ema = float(obv_ema.iloc[-1])
    
    # Determine trend
    if curr_obv > curr_obv_ema * 1.02:
        trend = "ACCUMULATION"
    elif curr_obv < curr_obv_ema * 0.98:
        trend = "DISTRIBUTION"
    else:
        trend = "NEUTRAL"
    
    # Detect divergence
    divergence = None
    lookback = 20
    if len(df) >= lookback:
        price_change_pct = (float(close.iloc[-1]) - float(close.iloc[-lookback])) / float(close.iloc[-lookback])
        obv_change_pct = (curr_obv - float(obv.iloc[-lookback])) / (abs(float(obv.iloc[-lookback])) + 1e-9)
        
        if price_change_pct > 0.02 and obv_change_pct < -0.02:
            divergence = "BEARISH_DIV"
        elif price_change_pct < -0.02 and obv_change_pct > 0.02:
            divergence = "BULLISH_DIV"
    
    # Calculate strength
    obv_range = obv.tail(50)
    if not obv_range.empty:
        obv_pctile = float((obv_range <= curr_obv).mean())
        if trend == "ACCUMULATION":
            strength = _clamp(obv_pctile)
        elif trend == "DISTRIBUTION":
            strength = _clamp(1.0 - obv_pctile)
        else:
            strength = 0.5
    else:
        strength = 0.5
    
    return OBVResult(
        obv=curr_obv,
        obv_ema=curr_obv_ema,
        trend=trend,
        divergence=divergence,
        strength=strength,
    )


# =============================================================================
# Ichimoku Cloud
# =============================================================================

@dataclass
class IchimokuResult:
    """Ichimoku Cloud calculation result."""
    tenkan_sen: float  # Conversion line
    kijun_sen: float  # Base line
    senkou_span_a: float  # Leading span A
    senkou_span_b: float  # Leading span B
    chikou_span: float  # Lagging span
    cloud_color: str  # "BULLISH", "BEARISH"
    price_location: str  # "ABOVE_CLOUD", "IN_CLOUD", "BELOW_CLOUD"
    tk_cross: Optional[str]  # "BULLISH_CROSS", "BEARISH_CROSS", None
    signal: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    strength: float


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> Optional[IchimokuResult]:
    """
    Calculate Ichimoku Cloud for comprehensive trend analysis.
    
    Components:
    - Tenkan-sen (Conversion): (9-period high + 9-period low) / 2
    - Kijun-sen (Base): (26-period high + 26-period low) / 2
    - Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
    - Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
    - Chikou Span: Close plotted 26 periods back
    """
    if len(df) < senkou_b_period + kijun_period:
        return None
    
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(tenkan_period).max()
    tenkan_low = low.rolling(tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = high.rolling(kijun_period).max()
    kijun_low = low.rolling(kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B)
    senkou_b_high = high.rolling(senkou_b_period).max()
    senkou_b_low = low.rolling(senkou_b_period).min()
    senkou_span_b = (senkou_b_high + senkou_b_low) / 2
    
    # Current values
    curr_price = float(close.iloc[-1])
    curr_tenkan = float(tenkan_sen.iloc[-1])
    curr_kijun = float(kijun_sen.iloc[-1])
    curr_senkou_a = float(senkou_span_a.iloc[-1])
    curr_senkou_b = float(senkou_span_b.iloc[-1])
    curr_chikou = curr_price  # Current close is chikou at -26
    
    prev_tenkan = float(tenkan_sen.iloc[-2]) if len(tenkan_sen) >= 2 else curr_tenkan
    prev_kijun = float(kijun_sen.iloc[-2]) if len(kijun_sen) >= 2 else curr_kijun
    
    # Cloud color
    cloud_color = "BULLISH" if curr_senkou_a > curr_senkou_b else "BEARISH"
    
    # Cloud boundaries
    cloud_top = max(curr_senkou_a, curr_senkou_b)
    cloud_bottom = min(curr_senkou_a, curr_senkou_b)
    
    # Price location relative to cloud
    if curr_price > cloud_top:
        price_location = "ABOVE_CLOUD"
    elif curr_price < cloud_bottom:
        price_location = "BELOW_CLOUD"
    else:
        price_location = "IN_CLOUD"
    
    # TK Cross detection
    tk_cross = None
    if prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun:
        tk_cross = "BULLISH_CROSS"
    elif prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun:
        tk_cross = "BEARISH_CROSS"
    
    # Determine overall signal
    bull_points = 0
    bear_points = 0
    
    # Price vs cloud
    if price_location == "ABOVE_CLOUD":
        bull_points += 2
    elif price_location == "BELOW_CLOUD":
        bear_points += 2
    
    # Cloud color
    if cloud_color == "BULLISH":
        bull_points += 1
    else:
        bear_points += 1
    
    # TK relationship
    if curr_tenkan > curr_kijun:
        bull_points += 1
    elif curr_tenkan < curr_kijun:
        bear_points += 1
    
    # Price vs Tenkan
    if curr_price > curr_tenkan:
        bull_points += 1
    elif curr_price < curr_tenkan:
        bear_points += 1
    
    # Determine signal and strength
    total_points = bull_points + bear_points
    if total_points == 0:
        signal = "NEUTRAL"
        strength = 0.5
    elif bull_points >= 4:
        signal = "STRONG_BUY"
        strength = _clamp(bull_points / 5)
    elif bull_points > bear_points:
        signal = "BUY"
        strength = _clamp(bull_points / 5)
    elif bear_points >= 4:
        signal = "STRONG_SELL"
        strength = _clamp(bear_points / 5)
    elif bear_points > bull_points:
        signal = "SELL"
        strength = _clamp(bear_points / 5)
    else:
        signal = "NEUTRAL"
        strength = 0.5
    
    return IchimokuResult(
        tenkan_sen=curr_tenkan,
        kijun_sen=curr_kijun,
        senkou_span_a=curr_senkou_a,
        senkou_span_b=curr_senkou_b,
        chikou_span=curr_chikou,
        cloud_color=cloud_color,
        price_location=price_location,
        tk_cross=tk_cross,
        signal=signal,
        strength=strength,
    )


# =============================================================================
# Williams %R
# =============================================================================

@dataclass
class WilliamsRResult:
    """Williams %R calculation result."""
    value: float  # -100 to 0
    signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    strength: float


def calculate_williams_r(
    df: pd.DataFrame,
    period: int = 14,
) -> Optional[WilliamsRResult]:
    """
    Calculate Williams %R momentum indicator.
    
    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    
    Interpretation:
    - -20 to 0: Overbought
    - -80 to -100: Oversold
    """
    if len(df) < period:
        return None
    
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    
    hl_range = highest_high - lowest_low
    williams_r = ((highest_high - close) / hl_range.replace(0, np.nan)) * -100
    
    curr_value = float(williams_r.iloc[-1])
    
    if not np.isfinite(curr_value):
        return None
    
    # Determine signal
    if curr_value >= -20:
        signal = "OVERBOUGHT"
        strength = _clamp((-curr_value) / 20)
    elif curr_value <= -80:
        signal = "OVERSOLD"
        strength = _clamp((curr_value + 100) / 20)
    else:
        signal = "NEUTRAL"
        strength = _clamp(1.0 - abs(curr_value + 50) / 50)
    
    return WilliamsRResult(
        value=curr_value,
        signal=signal,
        strength=strength,
    )


# =============================================================================
# Chaikin Money Flow (CMF)
# =============================================================================

@dataclass
class CMFResult:
    """Chaikin Money Flow calculation result."""
    value: float  # -1 to +1
    signal: str  # "BUYING_PRESSURE", "SELLING_PRESSURE", "NEUTRAL"
    strength: float


def calculate_cmf(
    df: pd.DataFrame,
    period: int = 21,
) -> Optional[CMFResult]:
    """
    Calculate Chaikin Money Flow for volume-weighted momentum.
    
    CMF = Sum(MFV) / Sum(Volume) over period
    where MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    
    Positive CMF = Buying pressure (accumulation)
    Negative CMF = Selling pressure (distribution)
    """
    if len(df) < period:
        return None
    
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    
    # Money Flow Multiplier
    hl_range = high - low
    mf_multiplier = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    
    # Money Flow Volume
    mfv = mf_multiplier * volume
    
    # CMF
    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    
    curr_cmf = float(cmf.iloc[-1])
    
    if not np.isfinite(curr_cmf):
        return None
    
    # Determine signal
    if curr_cmf > 0.1:
        signal = "BUYING_PRESSURE"
        strength = _clamp(curr_cmf)
    elif curr_cmf < -0.1:
        signal = "SELLING_PRESSURE"
        strength = _clamp(abs(curr_cmf))
    else:
        signal = "NEUTRAL"
        strength = _clamp(1.0 - abs(curr_cmf) * 5)
    
    return CMFResult(
        value=curr_cmf,
        signal=signal,
        strength=strength,
    )


# =============================================================================
# Keltner Channels
# =============================================================================

@dataclass
class KeltnerResult:
    """Keltner Channels calculation result."""
    upper: float
    middle: float
    lower: float
    position: float  # 0 = at lower, 0.5 = at middle, 1 = at upper
    signal: str  # "OVERBOUGHT", "OVERSOLD", "BREAKOUT_UP", "BREAKOUT_DOWN", "NEUTRAL"
    strength: float


def calculate_keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
) -> Optional[KeltnerResult]:
    """
    Calculate Keltner Channels for volatility-based bands.
    
    Middle = EMA(close)
    Upper = Middle + ATR * multiplier
    Lower = Middle - ATR * multiplier
    
    Unlike Bollinger, uses ATR instead of standard deviation.
    """
    if len(df) < max(ema_period, atr_period):
        return None
    
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    
    # Calculate ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    # Calculate channels
    middle = _ema(close, ema_period)
    upper = middle + (atr * atr_multiplier)
    lower = middle - (atr * atr_multiplier)
    
    # Current values
    curr_price = float(close.iloc[-1])
    curr_upper = float(upper.iloc[-1])
    curr_middle = float(middle.iloc[-1])
    curr_lower = float(lower.iloc[-1])
    
    # Calculate position
    channel_range = curr_upper - curr_lower
    if channel_range > 0:
        position = (curr_price - curr_lower) / channel_range
    else:
        position = 0.5
    
    # Determine signal
    if curr_price > curr_upper:
        signal = "BREAKOUT_UP"
        strength = _clamp((curr_price - curr_upper) / (float(atr.iloc[-1]) + 1e-9))
    elif curr_price < curr_lower:
        signal = "BREAKOUT_DOWN"
        strength = _clamp((curr_lower - curr_price) / (float(atr.iloc[-1]) + 1e-9))
    elif position > 0.8:
        signal = "OVERBOUGHT"
        strength = _clamp((position - 0.5) * 2)
    elif position < 0.2:
        signal = "OVERSOLD"
        strength = _clamp((0.5 - position) * 2)
    else:
        signal = "NEUTRAL"
        strength = _clamp(1.0 - abs(position - 0.5) * 2)
    
    return KeltnerResult(
        upper=curr_upper,
        middle=curr_middle,
        lower=curr_lower,
        position=position,
        signal=signal,
        strength=strength,
    )


# =============================================================================
# Aggregate Indicator Analysis
# =============================================================================

@dataclass
class IndicatorSummary:
    """Aggregated summary of all indicators."""
    overall_signal: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    overall_strength: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    indicators: Dict[str, IndicatorResult]


def calculate_all_indicators(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> IndicatorSummary:
    """
    Calculate all indicators and provide an aggregate summary.
    """
    config = config or {}
    indicators: Dict[str, IndicatorResult] = {}
    
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    total_strength = 0.0
    total_weight = 0.0
    
    # MACD
    macd = calculate_macd(
        df,
        fast_period=config.get("macd_fast", 12),
        slow_period=config.get("macd_slow", 26),
        signal_period=config.get("macd_signal", 9),
    )
    if macd:
        indicators["macd"] = IndicatorResult(
            value=macd.histogram,
            signal=macd.signal,
            strength=macd.strength,
            details={"crossover": macd.crossover, "divergence": macd.divergence},
        )
        if macd.signal == "BULLISH":
            bullish_count += 1
        elif macd.signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += macd.strength * 2.5
        total_weight += 2.5
    
    # Bollinger Bands
    bollinger = calculate_bollinger_bands(
        df,
        period=config.get("bollinger_period", 20),
        std_dev=config.get("bollinger_std", 2.0),
    )
    if bollinger:
        bb_signal = "NEUTRAL"
        if bollinger.signal == "OVERSOLD":
            bb_signal = "BULLISH"
        elif bollinger.signal == "OVERBOUGHT":
            bb_signal = "BEARISH"
        
        indicators["bollinger"] = IndicatorResult(
            value=bollinger.percent_b,
            signal=bb_signal,
            strength=bollinger.strength,
            details={"squeeze": bollinger.squeeze_active, "bandwidth": bollinger.bandwidth},
        )
        if bb_signal == "BULLISH":
            bullish_count += 1
        elif bb_signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += bollinger.strength * 1.5
        total_weight += 1.5
    
    # Stochastic RSI
    stoch_rsi = calculate_stoch_rsi(
        df,
        rsi_period=config.get("stoch_rsi_period", 14),
        smooth_k=config.get("stoch_rsi_smooth_k", 3),
        smooth_d=config.get("stoch_rsi_smooth_d", 3),
    )
    if stoch_rsi:
        sr_signal = "NEUTRAL"
        if stoch_rsi.signal == "OVERSOLD":
            sr_signal = "BULLISH"
        elif stoch_rsi.signal == "OVERBOUGHT":
            sr_signal = "BEARISH"
        
        indicators["stoch_rsi"] = IndicatorResult(
            value=stoch_rsi.k,
            signal=sr_signal,
            strength=stoch_rsi.strength,
            details={"k": stoch_rsi.k, "d": stoch_rsi.d, "crossover": stoch_rsi.crossover},
        )
        if sr_signal == "BULLISH":
            bullish_count += 1
        elif sr_signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += stoch_rsi.strength * 2.0
        total_weight += 2.0
    
    # ADX
    adx = calculate_adx(df, period=config.get("adx_period", 14))
    if adx:
        indicators["adx"] = IndicatorResult(
            value=adx.adx,
            signal=adx.trend_direction,
            strength=adx.strength,
            details={"adx_value": adx.adx, "plus_di": adx.plus_di, "minus_di": adx.minus_di, "trend_signal": adx.signal},
        )
        if adx.trend_direction == "BULLISH" and adx.adx >= 25:
            bullish_count += 1
        elif adx.trend_direction == "BEARISH" and adx.adx >= 25:
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += adx.strength * 2.0
        total_weight += 2.0
    
    # OBV
    obv = calculate_obv(df, ema_period=21)
    if obv:
        obv_signal = "NEUTRAL"
        if obv.trend == "ACCUMULATION":
            obv_signal = "BULLISH"
        elif obv.trend == "DISTRIBUTION":
            obv_signal = "BEARISH"
        
        indicators["obv"] = IndicatorResult(
            value=obv.obv,
            signal=obv_signal,
            strength=obv.strength,
            details={"trend": obv.trend, "divergence": obv.divergence},
        )
        if obv_signal == "BULLISH":
            bullish_count += 1
        elif obv_signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += obv.strength * 2.5
        total_weight += 2.5
    
    # Ichimoku
    ichimoku = calculate_ichimoku(
        df,
        tenkan_period=config.get("ichimoku_tenkan", 9),
        kijun_period=config.get("ichimoku_kijun", 26),
        senkou_b_period=config.get("ichimoku_senkou_b", 52),
    )
    if ichimoku:
        ich_signal = "NEUTRAL"
        if ichimoku.signal in ("STRONG_BUY", "BUY"):
            ich_signal = "BULLISH"
        elif ichimoku.signal in ("STRONG_SELL", "SELL"):
            ich_signal = "BEARISH"
        
        indicators["ichimoku"] = IndicatorResult(
            value=0.0,
            signal=ich_signal,
            strength=ichimoku.strength,
            details={
                "cloud_color": ichimoku.cloud_color,
                "price_location": ichimoku.price_location,
                "tk_cross": ichimoku.tk_cross,
                "raw_signal": ichimoku.signal,
            },
        )
        if ich_signal == "BULLISH":
            bullish_count += 1
        elif ich_signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += ichimoku.strength * 2.0
        total_weight += 2.0
    
    # Williams %R
    williams = calculate_williams_r(df, period=14)
    if williams:
        wr_signal = "NEUTRAL"
        if williams.signal == "OVERSOLD":
            wr_signal = "BULLISH"
        elif williams.signal == "OVERBOUGHT":
            wr_signal = "BEARISH"
        
        indicators["williams_r"] = IndicatorResult(
            value=williams.value,
            signal=wr_signal,
            strength=williams.strength,
        )
        if wr_signal == "BULLISH":
            bullish_count += 1
        elif wr_signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += williams.strength * 1.0
        total_weight += 1.0
    
    # CMF
    cmf = calculate_cmf(df, period=21)
    if cmf:
        cmf_signal = "NEUTRAL"
        if cmf.signal == "BUYING_PRESSURE":
            cmf_signal = "BULLISH"
        elif cmf.signal == "SELLING_PRESSURE":
            cmf_signal = "BEARISH"
        
        indicators["cmf"] = IndicatorResult(
            value=cmf.value,
            signal=cmf_signal,
            strength=cmf.strength,
        )
        if cmf_signal == "BULLISH":
            bullish_count += 1
        elif cmf_signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += cmf.strength * 1.5
        total_weight += 1.5
    
    # Keltner
    keltner = calculate_keltner_channels(df, ema_period=20, atr_period=14, atr_multiplier=2.0)
    if keltner:
        kc_signal = "NEUTRAL"
        if keltner.signal in ("OVERSOLD", "BREAKOUT_UP"):
            kc_signal = "BULLISH"
        elif keltner.signal in ("OVERBOUGHT", "BREAKOUT_DOWN"):
            kc_signal = "BEARISH"
        
        indicators["keltner"] = IndicatorResult(
            value=keltner.position,
            signal=kc_signal,
            strength=keltner.strength,
        )
        if kc_signal == "BULLISH":
            bullish_count += 1
        elif kc_signal == "BEARISH":
            bearish_count += 1
        else:
            neutral_count += 1
        total_strength += keltner.strength * 1.0
        total_weight += 1.0
    
    # Determine overall signal
    total_indicators = bullish_count + bearish_count + neutral_count
    if total_indicators == 0:
        overall_signal = "NEUTRAL"
        overall_strength = 0.5
    else:
        bullish_ratio = bullish_count / total_indicators
        bearish_ratio = bearish_count / total_indicators
        
        if bullish_ratio >= 0.7:
            overall_signal = "STRONG_BUY"
        elif bullish_ratio >= 0.5:
            overall_signal = "BUY"
        elif bearish_ratio >= 0.7:
            overall_signal = "STRONG_SELL"
        elif bearish_ratio >= 0.5:
            overall_signal = "SELL"
        else:
            overall_signal = "NEUTRAL"
        
        if total_weight > 0:
            overall_strength = _clamp(total_strength / total_weight)
        else:
            overall_strength = 0.5
    
    return IndicatorSummary(
        overall_signal=overall_signal,
        overall_strength=overall_strength,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        indicators=indicators,
    )
