"""
Multi-Timeframe Analysis Engine

This module provides comprehensive multi-timeframe (MTF) analysis
for professional trading decisions. It aggregates signals across
5 timeframes to determine trend alignment and confluence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple

import numpy as np
import pandas as pd


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""
    timeframe: str
    trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    trend_strength: float
    ema_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    rsi_value: float
    momentum: str  # "BULLISH", "BEARISH", "NEUTRAL"
    structure: str  # "HIGHER_HIGHS", "LOWER_LOWS", "RANGING"
    key_level_distance: float  # Distance to nearest S/R in ATR
    signal: str  # "BUY", "SELL", "NEUTRAL"
    weight: float  # Timeframe weight for aggregation


@dataclass
class MTFAlignment:
    """Multi-timeframe alignment analysis result."""
    overall_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    alignment_score: float  # 0.0 to 1.0
    aligned_timeframes: int
    total_timeframes: int
    weighted_score: float
    dominant_side: str  # "LONG", "SHORT", "NONE"
    confluence_zones: List[Dict[str, Any]]
    timeframe_details: Dict[str, TimeframeAnalysis]
    recommendations: List[str]


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate RSI."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if np.isfinite(val) else 50.0


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR."""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    atr_val = float(tr.rolling(period).mean().iloc[-1])
    return atr_val if np.isfinite(atr_val) else 0.0


def _find_swing_levels(df: pd.DataFrame, lookback: int = 50) -> Tuple[List[float], List[float]]:
    """Find recent swing highs and lows as key levels."""
    if len(df) < lookback:
        lookback = len(df)
    
    recent = df.tail(lookback)
    highs = recent["high"].astype(float).values
    lows = recent["low"].astype(float).values
    
    resistance_levels: List[float] = []
    support_levels: List[float] = []
    
    for i in range(2, len(recent) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(float(highs[i]))
        
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(float(lows[i]))
    
    return resistance_levels, support_levels


def _analyze_structure(df: pd.DataFrame) -> str:
    """Analyze market structure (HH/HL vs LH/LL)."""
    if len(df) < 20:
        return "RANGING"
    
    resistance, support = _find_swing_levels(df, 50)
    
    if len(resistance) >= 2 and len(support) >= 2:
        # Check for higher highs and higher lows
        recent_res = sorted(resistance[-3:], reverse=True)
        recent_sup = sorted(support[-3:])
        
        hh = len(recent_res) >= 2 and recent_res[0] > recent_res[1]
        hl = len(recent_sup) >= 2 and recent_sup[0] > recent_sup[1]
        
        lh = len(recent_res) >= 2 and recent_res[0] < recent_res[1]
        ll = len(recent_sup) >= 2 and recent_sup[0] < recent_sup[1]
        
        if hh and hl:
            return "HIGHER_HIGHS"
        elif lh and ll:
            return "LOWER_LOWS"
    
    return "RANGING"


def analyze_single_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    weight: float = 1.0,
) -> TimeframeAnalysis:
    """
    Analyze a single timeframe and return structured analysis.
    
    Args:
        df: OHLCV DataFrame for the timeframe
        timeframe: Timeframe string (e.g., "1d", "4h", "1h")
        weight: Weight for this timeframe in aggregation
    
    Returns:
        TimeframeAnalysis with trend, momentum, structure analysis
    """
    if len(df) < 50:
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend="NEUTRAL",
            trend_strength=0.5,
            ema_bias="NEUTRAL",
            rsi_value=50.0,
            momentum="NEUTRAL",
            structure="RANGING",
            key_level_distance=0.0,
            signal="NEUTRAL",
            weight=weight,
        )
    
    closes = df["close"].astype(float)
    current_price = float(closes.iloc[-1])
    
    # EMA analysis
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200) if len(df) >= 200 else ema50
    
    curr_ema20 = float(ema20.iloc[-1])
    curr_ema50 = float(ema50.iloc[-1])
    curr_ema200 = float(ema200.iloc[-1])
    
    # Trend determination
    bullish_ema = curr_ema20 > curr_ema50 > curr_ema200
    bearish_ema = curr_ema20 < curr_ema50 < curr_ema200
    
    if bullish_ema:
        trend = "BULLISH"
        ema_bias = "BULLISH"
        ema_alignment = 3
    elif bearish_ema:
        trend = "BEARISH"
        ema_bias = "BEARISH"
        ema_alignment = 3
    elif curr_ema20 > curr_ema50:
        trend = "BULLISH"
        ema_bias = "BULLISH"
        ema_alignment = 2
    elif curr_ema20 < curr_ema50:
        trend = "BEARISH"
        ema_bias = "BEARISH"
        ema_alignment = 2
    else:
        trend = "NEUTRAL"
        ema_bias = "NEUTRAL"
        ema_alignment = 1
    
    trend_strength = min(1.0, ema_alignment / 3.0)
    
    # RSI and momentum
    rsi_value = _rsi(closes, 14)
    
    if rsi_value >= 60:
        momentum = "BULLISH"
    elif rsi_value <= 40:
        momentum = "BEARISH"
    else:
        momentum = "NEUTRAL"
    
    # Structure analysis
    structure = _analyze_structure(df)
    
    # Key level distance
    atr = _atr(df, 14)
    resistance, support = _find_swing_levels(df, 50)
    
    nearest_resistance = min([r for r in resistance if r > current_price], default=current_price * 1.1)
    nearest_support = max([s for s in support if s < current_price], default=current_price * 0.9)
    
    distance_to_resistance = (nearest_resistance - current_price) / atr if atr > 0 else 0
    distance_to_support = (current_price - nearest_support) / atr if atr > 0 else 0
    
    key_level_distance = min(distance_to_resistance, distance_to_support)
    
    # Overall signal for this timeframe
    bullish_score = 0
    bearish_score = 0
    
    if trend == "BULLISH":
        bullish_score += 2
    elif trend == "BEARISH":
        bearish_score += 2
    
    if momentum == "BULLISH":
        bullish_score += 1
    elif momentum == "BEARISH":
        bearish_score += 1
    
    if structure == "HIGHER_HIGHS":
        bullish_score += 1
    elif structure == "LOWER_LOWS":
        bearish_score += 1
    
    if bullish_score > bearish_score:
        signal = "BUY"
    elif bearish_score > bullish_score:
        signal = "SELL"
    else:
        signal = "NEUTRAL"
    
    return TimeframeAnalysis(
        timeframe=timeframe,
        trend=trend,
        trend_strength=trend_strength,
        ema_bias=ema_bias,
        rsi_value=rsi_value,
        momentum=momentum,
        structure=structure,
        key_level_distance=key_level_distance,
        signal=signal,
        weight=weight,
    )


def analyze_mtf_alignment(
    fetch_ohlcv: Callable[[str, str, int], pd.DataFrame],
    symbol: str,
    timeframes: Optional[List[str]] = None,
    timeframe_weights: Optional[Dict[str, float]] = None,
) -> MTFAlignment:
    """
    Perform multi-timeframe alignment analysis.
    
    Args:
        fetch_ohlcv: Function to fetch OHLCV data (symbol, timeframe, limit) -> DataFrame
        symbol: Trading symbol
        timeframes: List of timeframes to analyze (default: 1d, 4h, 1h, 15m, 5m)
        timeframe_weights: Optional custom weights for each timeframe
    
    Returns:
        MTFAlignment with comprehensive cross-timeframe analysis
    """
    if timeframes is None:
        timeframes = ["1d", "4h", "1h", "15m", "5m"]
    
    if timeframe_weights is None:
        # Higher timeframes get more weight
        timeframe_weights = {
            "1d": 3.0,
            "4h": 2.5,
            "1h": 2.0,
            "15m": 1.5,
            "5m": 1.0,
        }
    
    timeframe_details: Dict[str, TimeframeAnalysis] = {}
    bullish_weight = 0.0
    bearish_weight = 0.0
    neutral_weight = 0.0
    total_weight = 0.0
    aligned_count = 0
    
    # Analyze each timeframe
    for tf in timeframes:
        weight = timeframe_weights.get(tf, 1.0)
        
        try:
            df = fetch_ohlcv(symbol, tf, 500)
            analysis = analyze_single_timeframe(df, tf, weight)
        except Exception:
            analysis = TimeframeAnalysis(
                timeframe=tf,
                trend="NEUTRAL",
                trend_strength=0.5,
                ema_bias="NEUTRAL",
                rsi_value=50.0,
                momentum="NEUTRAL",
                structure="RANGING",
                key_level_distance=0.0,
                signal="NEUTRAL",
                weight=weight,
            )
        
        timeframe_details[tf] = analysis
        total_weight += weight
        
        if analysis.signal == "BUY":
            bullish_weight += weight * analysis.trend_strength
        elif analysis.signal == "SELL":
            bearish_weight += weight * analysis.trend_strength
        else:
            neutral_weight += weight
    
    # Calculate alignment
    if total_weight > 0:
        bullish_ratio = bullish_weight / total_weight
        bearish_ratio = bearish_weight / total_weight
    else:
        bullish_ratio = 0.0
        bearish_ratio = 0.0
    
    # Determine overall bias
    if bullish_ratio > bearish_ratio and bullish_ratio > 0.4:
        overall_bias = "BULLISH"
        alignment_score = bullish_ratio
        dominant_side = "LONG"
    elif bearish_ratio > bullish_ratio and bearish_ratio > 0.4:
        overall_bias = "BEARISH"
        alignment_score = bearish_ratio
        dominant_side = "SHORT"
    else:
        overall_bias = "NEUTRAL"
        alignment_score = 0.5
        dominant_side = "NONE"
    
    # Count aligned timeframes
    for tf, analysis in timeframe_details.items():
        if (overall_bias == "BULLISH" and analysis.signal == "BUY") or \
           (overall_bias == "BEARISH" and analysis.signal == "SELL"):
            aligned_count += 1
    
    # Find confluence zones (levels that appear in multiple timeframes)
    confluence_zones: List[Dict[str, Any]] = []
    all_levels: Dict[str, List[Tuple[str, float]]] = {"resistance": [], "support": []}
    
    for tf, analysis in timeframe_details.items():
        try:
            df = fetch_ohlcv(symbol, tf, 500)
            res, sup = _find_swing_levels(df, 50)
            for r in res[-5:]:  # Last 5 resistance levels
                all_levels["resistance"].append((tf, r))
            for s in sup[-5:]:  # Last 5 support levels
                all_levels["support"].append((tf, s))
        except Exception:
            pass
    
    # Group nearby levels as confluence
    def _group_levels(levels: List[Tuple[str, float]], threshold_pct: float = 0.01) -> List[Dict[str, Any]]:
        if not levels:
            return []
        
        sorted_levels = sorted(levels, key=lambda x: x[1])
        groups: List[Dict[str, Any]] = []
        current_group: List[Tuple[str, float]] = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            _, prev_price = sorted_levels[i - 1]
            tf, curr_price = sorted_levels[i]
            
            if abs(curr_price - prev_price) / prev_price < threshold_pct:
                current_group.append((tf, curr_price))
            else:
                if len(current_group) >= 2:  # At least 2 TFs confirm
                    avg_price = sum(p for _, p in current_group) / len(current_group)
                    groups.append({
                        "price": avg_price,
                        "timeframes": [t for t, _ in current_group],
                        "strength": len(current_group) / len(timeframes),
                    })
                current_group = [(tf, curr_price)]
        
        # Don't forget last group
        if len(current_group) >= 2:
            avg_price = sum(p for _, p in current_group) / len(current_group)
            groups.append({
                "price": avg_price,
                "timeframes": [t for t, _ in current_group],
                "strength": len(current_group) / len(timeframes),
            })
        
        return groups
    
    resistance_confluence = _group_levels(all_levels["resistance"])
    support_confluence = _group_levels(all_levels["support"])
    
    for zone in resistance_confluence:
        zone["type"] = "RESISTANCE"
        confluence_zones.append(zone)
    for zone in support_confluence:
        zone["type"] = "SUPPORT"
        confluence_zones.append(zone)
    
    # Generate recommendations
    recommendations: List[str] = []
    
    if alignment_score >= 0.7:
        recommendations.append(f"Strong {overall_bias.lower()} alignment across {aligned_count}/{len(timeframes)} timeframes")
    elif alignment_score >= 0.5:
        recommendations.append(f"Moderate {overall_bias.lower()} bias, consider waiting for better alignment")
    else:
        recommendations.append("Mixed signals across timeframes - consider waiting or trading shorter duration")
    
    # Check for divergences
    htf_signals = [timeframe_details[tf].signal for tf in timeframes[:2] if tf in timeframe_details]
    ltf_signals = [timeframe_details[tf].signal for tf in timeframes[-2:] if tf in timeframe_details]
    
    if htf_signals and ltf_signals:
        htf_bullish = htf_signals.count("BUY") > htf_signals.count("SELL")
        ltf_bullish = ltf_signals.count("BUY") > ltf_signals.count("SELL")
        
        if htf_bullish and not ltf_bullish:
            recommendations.append("HTF bullish but LTF showing weakness - potential pullback entry")
        elif not htf_bullish and ltf_bullish:
            recommendations.append("LTF bullish against HTF - potential counter-trend, use caution")
    
    if confluence_zones:
        recommendations.append(f"Found {len(confluence_zones)} confluence zones across multiple timeframes")
    
    # Calculate weighted score
    weighted_score = (bullish_weight * 2 + neutral_weight) / (total_weight * 2) if total_weight > 0 else 0.5
    if overall_bias == "BEARISH":
        weighted_score = 1.0 - weighted_score
    
    return MTFAlignment(
        overall_bias=overall_bias,
        alignment_score=alignment_score,
        aligned_timeframes=aligned_count,
        total_timeframes=len(timeframes),
        weighted_score=weighted_score,
        dominant_side=dominant_side,
        confluence_zones=confluence_zones,
        timeframe_details=timeframe_details,
        recommendations=recommendations,
    )


def calculate_mtf_score(
    mtf_alignment: MTFAlignment,
    side: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a normalized MTF score for the given trade side.
    
    Args:
        mtf_alignment: Analyzed MTF alignment data
        side: "LONG" or "SHORT"
    
    Returns:
        Tuple of (score 0-1, details dict)
    """
    score = 0.0
    max_score = 0.0
    details: Dict[str, Any] = {}
    
    # Bias alignment (weight: 3)
    max_score += 3.0
    target_bias = "BULLISH" if side == "LONG" else "BEARISH"
    
    if mtf_alignment.overall_bias == target_bias:
        score += 3.0 * mtf_alignment.alignment_score
        details["bias_aligned"] = True
    elif mtf_alignment.overall_bias == "NEUTRAL":
        score += 1.5
        details["bias_aligned"] = None
    else:
        details["bias_aligned"] = False
    
    # Timeframe alignment count (weight: 2)
    max_score += 2.0
    alignment_ratio = mtf_alignment.aligned_timeframes / mtf_alignment.total_timeframes
    score += 2.0 * alignment_ratio
    details["alignment_ratio"] = alignment_ratio
    
    # Check individual timeframe support
    htf_support = 0
    for tf in ["1d", "4h"]:
        if tf in mtf_alignment.timeframe_details:
            analysis = mtf_alignment.timeframe_details[tf]
            if (side == "LONG" and analysis.signal == "BUY") or \
               (side == "SHORT" and analysis.signal == "SELL"):
                htf_support += 1
    
    # HTF support (weight: 2)
    max_score += 2.0
    score += 2.0 * (htf_support / 2)
    details["htf_support"] = htf_support
    
    # Confluence zones available (weight: 1)
    max_score += 1.0
    if mtf_alignment.confluence_zones:
        score += 1.0
        details["confluence_zones"] = len(mtf_alignment.confluence_zones)
    else:
        details["confluence_zones"] = 0
    
    # Weighted score contribution (weight: 2)
    max_score += 2.0
    if side == "LONG":
        score += 2.0 * mtf_alignment.weighted_score
    else:
        score += 2.0 * (1.0 - mtf_alignment.weighted_score)
    
    normalized_score = score / max_score if max_score > 0 else 0.0
    details["raw_score"] = score
    details["max_score"] = max_score
    
    return normalized_score, details


def get_mtf_entry_zones(
    mtf_alignment: MTFAlignment,
    side: str,
    current_price: float,
) -> List[Dict[str, Any]]:
    """
    Get optimal entry zones based on MTF confluence analysis.
    
    Args:
        mtf_alignment: Analyzed MTF alignment data
        side: "LONG" or "SHORT"
        current_price: Current market price
    
    Returns:
        List of entry zone dictionaries with price and strength
    """
    entry_zones: List[Dict[str, Any]] = []
    
    target_type = "SUPPORT" if side == "LONG" else "RESISTANCE"
    
    for zone in mtf_alignment.confluence_zones:
        if zone["type"] == target_type:
            price = zone["price"]
            
            # For LONG, we want zones below current price
            # For SHORT, we want zones above current price
            if (side == "LONG" and price < current_price) or \
               (side == "SHORT" and price > current_price):
                
                distance_pct = abs(current_price - price) / current_price
                
                entry_zones.append({
                    "price": price,
                    "timeframes": zone["timeframes"],
                    "strength": zone["strength"],
                    "distance_pct": distance_pct,
                    "type": zone["type"],
                })
    
    # Sort by nearest first
    entry_zones.sort(key=lambda x: x["distance_pct"])
    
    return entry_zones[:3]  # Return top 3 zones
