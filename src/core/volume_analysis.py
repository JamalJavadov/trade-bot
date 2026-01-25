"""
Deep Volume Analysis Module

This module provides institutional-grade volume analysis including
volume profile, delta analysis, VWAP bands, and cumulative volume 
delta for professional trading decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class VolumeProfileLevel:
    """Represents a price level in the volume profile."""
    price: float
    volume: float
    volume_pct: float  # Percentage of total volume
    is_poc: bool  # Point of Control
    is_vah: bool  # Value Area High
    is_val: bool  # Value Area Low
    is_hvn: bool  # High Volume Node
    is_lvn: bool  # Low Volume Node


@dataclass
class VolumeProfile:
    """Complete volume profile analysis."""
    poc: float  # Point of Control - highest volume price
    vah: float  # Value Area High
    val: float  # Value Area Low
    value_area_volume_pct: float  # Volume within VA as percentage
    levels: List[VolumeProfileLevel]
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]
    developing_poc: float  # POC of recent bars only


@dataclass
class VWAPBands:
    """VWAP with standard deviation bands."""
    vwap: float
    upper_1std: float
    lower_1std: float
    upper_2std: float
    lower_2std: float
    upper_3std: float
    lower_3std: float
    price_position: str  # "ABOVE_2STD", "ABOVE_1STD", "AT_VWAP", "BELOW_1STD", "BELOW_2STD"
    strength: float


@dataclass
class DeltaAnalysis:
    """Volume delta (buying vs selling pressure) analysis."""
    current_delta: float  # Positive = more buying, Negative = more selling
    cumulative_delta: float  # Running total
    delta_ma: float  # Moving average of delta
    divergence: Optional[str]  # "BULLISH_DIV", "BEARISH_DIV", None
    pressure: str  # "BUYING", "SELLING", "NEUTRAL"
    strength: float


@dataclass
class VolumeAnalysis:
    """Complete volume analysis result."""
    profile: VolumeProfile
    vwap_bands: VWAPBands
    delta: DeltaAnalysis
    volume_trend: str  # "INCREASING", "DECREASING", "STABLE"
    relative_volume: float  # Current volume vs average
    signal: str  # "ACCUMULATION", "DISTRIBUTION", "NEUTRAL"
    strength: float
    details: Dict[str, Any] = field(default_factory=dict)


def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
    developing_lookback: int = 20,
) -> VolumeProfile:
    """
    Calculate Volume Profile with Point of Control and Value Area.
    
    The Volume Profile shows the distribution of volume across price levels,
    helping identify significant support/resistance zones.
    
    Args:
        df: OHLCV DataFrame
        num_bins: Number of price levels to divide the range into
        value_area_pct: Percentage of volume to include in Value Area (default 70%)
        developing_lookback: Bars to use for developing POC calculation
    
    Returns:
        VolumeProfile with POC, VAH, VAL, and HVN/LVN levels
    """
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    closes = df["close"].astype(float)
    volumes = df["volume"].astype(float)
    
    # Calculate typical price for each bar
    typical_price = (highs + lows + closes) / 3
    
    # Determine price range
    price_min = float(lows.min())
    price_max = float(highs.max())
    price_range = price_max - price_min
    
    if price_range <= 0:
        # Return default if no price range
        current_price = float(closes.iloc[-1])
        return VolumeProfile(
            poc=current_price,
            vah=current_price,
            val=current_price,
            value_area_volume_pct=1.0,
            levels=[],
            high_volume_nodes=[current_price],
            low_volume_nodes=[],
            developing_poc=current_price,
        )
    
    bin_size = price_range / num_bins
    
    # Create volume profile bins
    volume_by_price: Dict[int, float] = {i: 0.0 for i in range(num_bins)}
    
    for i in range(len(df)):
        bar_high = float(highs.iloc[i])
        bar_low = float(lows.iloc[i])
        bar_volume = float(volumes.iloc[i])
        
        # Distribute volume across price levels the bar touched
        low_bin = int((bar_low - price_min) / bin_size)
        high_bin = int((bar_high - price_min) / bin_size)
        
        low_bin = max(0, min(num_bins - 1, low_bin))
        high_bin = max(0, min(num_bins - 1, high_bin))
        
        bins_touched = high_bin - low_bin + 1
        volume_per_bin = bar_volume / bins_touched if bins_touched > 0 else bar_volume
        
        for b in range(low_bin, high_bin + 1):
            volume_by_price[b] += volume_per_bin
    
    # Calculate total volume
    total_volume = sum(volume_by_price.values())
    if total_volume <= 0:
        total_volume = 1.0
    
    # Find POC (highest volume level)
    poc_bin = max(volume_by_price.keys(), key=lambda x: volume_by_price[x])
    poc_price = price_min + (poc_bin + 0.5) * bin_size
    
    # Calculate Value Area (70% of volume around POC)
    sorted_bins = sorted(volume_by_price.keys(), key=lambda x: volume_by_price[x], reverse=True)
    va_volume = 0.0
    va_bins = set()
    target_volume = total_volume * value_area_pct
    
    for b in sorted_bins:
        va_bins.add(b)
        va_volume += volume_by_price[b]
        if va_volume >= target_volume:
            break
    
    vah_bin = max(va_bins)
    val_bin = min(va_bins)
    vah = price_min + (vah_bin + 1) * bin_size
    val = price_min + val_bin * bin_size
    
    # Identify HVN and LVN
    avg_volume = total_volume / num_bins
    std_volume = np.std(list(volume_by_price.values()))
    
    high_volume_nodes: List[float] = []
    low_volume_nodes: List[float] = []
    levels: List[VolumeProfileLevel] = []
    
    for b in range(num_bins):
        bin_price = price_min + (b + 0.5) * bin_size
        bin_volume = volume_by_price[b]
        volume_pct = bin_volume / total_volume
        
        is_hvn = bin_volume > avg_volume + std_volume
        is_lvn = bin_volume < avg_volume - std_volume * 0.5 and bin_volume > 0
        
        if is_hvn:
            high_volume_nodes.append(bin_price)
        if is_lvn:
            low_volume_nodes.append(bin_price)
        
        levels.append(VolumeProfileLevel(
            price=bin_price,
            volume=bin_volume,
            volume_pct=volume_pct,
            is_poc=(b == poc_bin),
            is_vah=(b == vah_bin),
            is_val=(b == val_bin),
            is_hvn=is_hvn,
            is_lvn=is_lvn,
        ))
    
    # Calculate developing POC (recent bars only)
    if len(df) > developing_lookback:
        recent_df = df.tail(developing_lookback)
        recent_typical = (recent_df["high"].astype(float) + 
                         recent_df["low"].astype(float) + 
                         recent_df["close"].astype(float)) / 3
        developing_poc = float(recent_typical.mean())
    else:
        developing_poc = poc_price
    
    return VolumeProfile(
        poc=poc_price,
        vah=vah,
        val=val,
        value_area_volume_pct=va_volume / total_volume,
        levels=levels,
        high_volume_nodes=high_volume_nodes,
        low_volume_nodes=low_volume_nodes,
        developing_poc=developing_poc,
    )


def calculate_vwap_bands(
    df: pd.DataFrame,
    std_multipliers: Tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> VWAPBands:
    """
    Calculate VWAP with standard deviation bands.
    
    VWAP (Volume Weighted Average Price) is the benchmark price
    for institutional trading. Bands help identify overbought/oversold levels.
    
    Args:
        df: OHLCV DataFrame
        std_multipliers: Multipliers for standard deviation bands
    
    Returns:
        VWAPBands with VWAP and 1/2/3 standard deviation bands
    """
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    closes = df["close"].astype(float)
    volumes = df["volume"].astype(float)
    
    # Calculate typical price
    typical_price = (highs + lows + closes) / 3
    
    # Calculate cumulative values for VWAP
    cum_tp_vol = (typical_price * volumes).cumsum()
    cum_vol = volumes.cumsum()
    
    # VWAP
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    
    # Calculate standard deviation bands
    # Using squared deviation from VWAP weighted by volume
    squared_dev = ((typical_price - vwap) ** 2) * volumes
    cum_squared_dev = squared_dev.cumsum()
    variance = cum_squared_dev / cum_vol.replace(0, np.nan)
    std_dev = np.sqrt(variance)
    
    # Get current values
    curr_vwap = float(vwap.iloc[-1])
    curr_std = float(std_dev.iloc[-1])
    curr_price = float(closes.iloc[-1])
    
    if not np.isfinite(curr_vwap):
        curr_vwap = curr_price
    if not np.isfinite(curr_std):
        curr_std = 0.0
    
    # Calculate bands
    upper_1std = curr_vwap + curr_std * std_multipliers[0]
    lower_1std = curr_vwap - curr_std * std_multipliers[0]
    upper_2std = curr_vwap + curr_std * std_multipliers[1]
    lower_2std = curr_vwap - curr_std * std_multipliers[1]
    upper_3std = curr_vwap + curr_std * std_multipliers[2]
    lower_3std = curr_vwap - curr_std * std_multipliers[2]
    
    # Determine price position
    if curr_price >= upper_2std:
        price_position = "ABOVE_2STD"
        strength = min(1.0, (curr_price - upper_2std) / (upper_3std - upper_2std + 1e-9))
    elif curr_price >= upper_1std:
        price_position = "ABOVE_1STD"
        strength = (curr_price - upper_1std) / (upper_2std - upper_1std + 1e-9)
    elif curr_price <= lower_2std:
        price_position = "BELOW_2STD"
        strength = min(1.0, (lower_2std - curr_price) / (lower_2std - lower_3std + 1e-9))
    elif curr_price <= lower_1std:
        price_position = "BELOW_1STD"
        strength = (lower_1std - curr_price) / (lower_1std - lower_2std + 1e-9)
    else:
        price_position = "AT_VWAP"
        distance_to_vwap = abs(curr_price - curr_vwap)
        strength = 1.0 - min(1.0, distance_to_vwap / (curr_std + 1e-9))
    
    strength = max(0.0, min(1.0, strength))
    
    return VWAPBands(
        vwap=curr_vwap,
        upper_1std=upper_1std,
        lower_1std=lower_1std,
        upper_2std=upper_2std,
        lower_2std=lower_2std,
        upper_3std=upper_3std,
        lower_3std=lower_3std,
        price_position=price_position,
        strength=strength,
    )


def calculate_delta_analysis(
    df: pd.DataFrame,
    ma_period: int = 14,
    divergence_lookback: int = 20,
) -> DeltaAnalysis:
    """
    Calculate Volume Delta (buying vs selling pressure).
    
    Delta is estimated from candle structure:
    - Bullish candle: Assume more buying (add volume)
    - Bearish candle: Assume more selling (subtract volume)
    - Wick analysis: Adjust based on rejection
    
    Args:
        df: OHLCV DataFrame
        ma_period: Period for delta moving average
        divergence_lookback: Bars to check for divergence
    
    Returns:
        DeltaAnalysis with delta, cumulative delta, and divergence detection
    """
    opens = df["open"].astype(float)
    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    volumes = df["volume"].astype(float)
    
    # Estimate delta from candle structure
    # Delta = Volume * (close - open) / (high - low)
    # This approximates buying vs selling pressure
    body = closes - opens
    range_hl = (highs - lows).replace(0, np.nan)
    
    # Normalize body size relative to range
    body_ratio = body / range_hl
    
    # Delta is volume weighted by body ratio
    delta = volumes * body_ratio.fillna(0)
    
    # Cumulative delta
    cumulative_delta = delta.cumsum()
    
    # Delta moving average
    delta_ma = delta.rolling(ma_period).mean()
    
    # Get current values
    curr_delta = float(delta.iloc[-1])
    curr_cum_delta = float(cumulative_delta.iloc[-1])
    curr_delta_ma = float(delta_ma.iloc[-1])
    
    if not np.isfinite(curr_delta):
        curr_delta = 0.0
    if not np.isfinite(curr_cum_delta):
        curr_cum_delta = 0.0
    if not np.isfinite(curr_delta_ma):
        curr_delta_ma = 0.0
    
    # Detect divergence
    divergence = None
    if len(df) >= divergence_lookback:
        price_change = float(closes.iloc[-1]) - float(closes.iloc[-divergence_lookback])
        delta_change = curr_cum_delta - float(cumulative_delta.iloc[-divergence_lookback])
        
        # Bullish divergence: Price down, delta up
        if price_change < 0 and delta_change > 0:
            divergence = "BULLISH_DIV"
        # Bearish divergence: Price up, delta down
        elif price_change > 0 and delta_change < 0:
            divergence = "BEARISH_DIV"
    
    # Determine pressure
    if curr_delta_ma > 0:
        pressure = "BUYING"
        delta_std = float(delta.tail(50).std())
        if delta_std > 0:
            strength = min(1.0, abs(curr_delta_ma) / (delta_std * 2))
        else:
            strength = 0.5
    elif curr_delta_ma < 0:
        pressure = "SELLING"
        delta_std = float(delta.tail(50).std())
        if delta_std > 0:
            strength = min(1.0, abs(curr_delta_ma) / (delta_std * 2))
        else:
            strength = 0.5
    else:
        pressure = "NEUTRAL"
        strength = 0.5
    
    return DeltaAnalysis(
        current_delta=curr_delta,
        cumulative_delta=curr_cum_delta,
        delta_ma=curr_delta_ma,
        divergence=divergence,
        pressure=pressure,
        strength=strength,
    )


def calculate_volume_trend(
    df: pd.DataFrame,
    short_period: int = 10,
    long_period: int = 30,
) -> Tuple[str, float]:
    """
    Determine if volume is increasing, decreasing, or stable.
    
    Returns:
        Tuple of (trend_string, relative_volume_ratio)
    """
    volumes = df["volume"].astype(float)
    
    if len(df) < long_period:
        return "STABLE", 1.0
    
    short_avg = float(volumes.tail(short_period).mean())
    long_avg = float(volumes.tail(long_period).mean())
    
    if long_avg <= 0:
        return "STABLE", 1.0
    
    relative_volume = short_avg / long_avg
    
    if relative_volume > 1.3:
        return "INCREASING", relative_volume
    elif relative_volume < 0.7:
        return "DECREASING", relative_volume
    else:
        return "STABLE", relative_volume


def analyze_volume(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> VolumeAnalysis:
    """
    Perform comprehensive volume analysis.
    
    This is the main entry point for deep volume analysis,
    combining volume profile, VWAP, and delta analysis.
    
    Args:
        df: OHLCV DataFrame
        config: Optional configuration dictionary
    
    Returns:
        VolumeAnalysis with complete volume insights
    """
    config = config or {}
    
    # Configuration
    num_bins = config.get("num_bins", 50)
    value_area_pct = config.get("value_area_pct", 0.70)
    developing_lookback = config.get("developing_lookback", 20)
    delta_ma_period = config.get("delta_ma_period", 14)
    
    # Calculate components
    profile = calculate_volume_profile(df, num_bins, value_area_pct, developing_lookback)
    vwap_bands = calculate_vwap_bands(df)
    delta = calculate_delta_analysis(df, delta_ma_period)
    volume_trend, relative_volume = calculate_volume_trend(df)
    
    # Determine overall signal
    accumulation_score = 0
    distribution_score = 0
    
    # Delta analysis contribution
    if delta.pressure == "BUYING":
        accumulation_score += 2
    elif delta.pressure == "SELLING":
        distribution_score += 2
    
    # Divergence contribution
    if delta.divergence == "BULLISH_DIV":
        accumulation_score += 1
    elif delta.divergence == "BEARISH_DIV":
        distribution_score += 1
    
    # VWAP position contribution
    if vwap_bands.price_position in ("BELOW_1STD", "BELOW_2STD"):
        accumulation_score += 1  # Oversold, potential accumulation
    elif vwap_bands.price_position in ("ABOVE_1STD", "ABOVE_2STD"):
        distribution_score += 1  # Overbought, potential distribution
    
    # Volume trend contribution
    if volume_trend == "INCREASING":
        # High volume confirms the direction
        if accumulation_score > distribution_score:
            accumulation_score += 1
        elif distribution_score > accumulation_score:
            distribution_score += 1
    
    # Determine signal
    if accumulation_score > distribution_score + 1:
        signal = "ACCUMULATION"
        strength = min(1.0, accumulation_score / 5)
    elif distribution_score > accumulation_score + 1:
        signal = "DISTRIBUTION"
        strength = min(1.0, distribution_score / 5)
    else:
        signal = "NEUTRAL"
        strength = 0.5
    
    # Compile details
    details: Dict[str, Any] = {
        "poc": profile.poc,
        "vah": profile.vah,
        "val": profile.val,
        "vwap": vwap_bands.vwap,
        "vwap_position": vwap_bands.price_position,
        "delta_pressure": delta.pressure,
        "delta_divergence": delta.divergence,
        "volume_trend": volume_trend,
        "relative_volume": relative_volume,
        "hvn_count": len(profile.high_volume_nodes),
        "lvn_count": len(profile.low_volume_nodes),
        "accumulation_score": accumulation_score,
        "distribution_score": distribution_score,
    }
    
    return VolumeAnalysis(
        profile=profile,
        vwap_bands=vwap_bands,
        delta=delta,
        volume_trend=volume_trend,
        relative_volume=relative_volume,
        signal=signal,
        strength=strength,
        details=details,
    )


def calculate_volume_score(
    volume_analysis: VolumeAnalysis,
    side: str,
    current_price: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a normalized volume score for the given trade side.
    
    Args:
        volume_analysis: Analyzed volume data
        side: "LONG" or "SHORT"
        current_price: Current market price
    
    Returns:
        Tuple of (score 0-1, details dict)
    """
    score = 0.0
    max_score = 0.0
    details: Dict[str, Any] = {}
    
    # Signal alignment (weight: 3)
    max_score += 3.0
    if side == "LONG" and volume_analysis.signal == "ACCUMULATION":
        score += 3.0 * volume_analysis.strength
        details["signal_aligned"] = True
    elif side == "SHORT" and volume_analysis.signal == "DISTRIBUTION":
        score += 3.0 * volume_analysis.strength
        details["signal_aligned"] = True
    elif volume_analysis.signal == "NEUTRAL":
        score += 1.5
        details["signal_aligned"] = None
    else:
        details["signal_aligned"] = False
    
    # Delta divergence bonus (weight: 2)
    max_score += 2.0
    if side == "LONG" and volume_analysis.delta.divergence == "BULLISH_DIV":
        score += 2.0
        details["divergence_bonus"] = True
    elif side == "SHORT" and volume_analysis.delta.divergence == "BEARISH_DIV":
        score += 2.0
        details["divergence_bonus"] = True
    else:
        details["divergence_bonus"] = False
    
    # VWAP position (weight: 2)
    max_score += 2.0
    vwap = volume_analysis.vwap_bands.vwap
    if side == "LONG":
        # Below VWAP is good for longs (mean reversion opportunity)
        if current_price < vwap:
            discount = (vwap - current_price) / vwap
            score += min(2.0, discount * 20)
            details["vwap_favorable"] = True
        else:
            details["vwap_favorable"] = False
    else:
        # Above VWAP is good for shorts
        if current_price > vwap:
            premium = (current_price - vwap) / vwap
            score += min(2.0, premium * 20)
            details["vwap_favorable"] = True
        else:
            details["vwap_favorable"] = False
    
    # POC proximity (weight: 1.5)
    max_score += 1.5
    poc = volume_analysis.profile.poc
    distance_to_poc = abs(current_price - poc) / poc
    if distance_to_poc < 0.02:  # Within 2% of POC
        score += 1.5
        details["near_poc"] = True
    elif distance_to_poc < 0.05:  # Within 5%
        score += 0.75
        details["near_poc"] = "PARTIAL"
    else:
        details["near_poc"] = False
    
    # Volume trend (weight: 1.5)
    max_score += 1.5
    if volume_analysis.volume_trend == "INCREASING" and volume_analysis.relative_volume > 1.2:
        score += 1.5
        details["volume_confirming"] = True
    elif volume_analysis.volume_trend == "STABLE":
        score += 0.75
        details["volume_confirming"] = "NEUTRAL"
    else:
        details["volume_confirming"] = False
    
    normalized_score = score / max_score if max_score > 0 else 0.0
    details["raw_score"] = score
    details["max_score"] = max_score
    
    return normalized_score, details


def get_key_volume_levels(
    volume_analysis: VolumeAnalysis,
    current_price: float,
) -> Dict[str, List[float]]:
    """
    Get key volume-based support and resistance levels.
    
    Returns:
        Dictionary with support_levels and resistance_levels
    """
    profile = volume_analysis.profile
    vwap = volume_analysis.vwap_bands
    
    all_levels: List[Tuple[float, str]] = []
    
    # Add POC
    all_levels.append((profile.poc, "POC"))
    
    # Add Value Area boundaries
    all_levels.append((profile.vah, "VAH"))
    all_levels.append((profile.val, "VAL"))
    
    # Add VWAP bands
    all_levels.append((vwap.vwap, "VWAP"))
    all_levels.append((vwap.upper_1std, "VWAP+1"))
    all_levels.append((vwap.lower_1std, "VWAP-1"))
    all_levels.append((vwap.upper_2std, "VWAP+2"))
    all_levels.append((vwap.lower_2std, "VWAP-2"))
    
    # Add High Volume Nodes
    for hvn in profile.high_volume_nodes[:3]:  # Top 3 HVNs
        all_levels.append((hvn, "HVN"))
    
    # Separate into support and resistance
    support_levels: List[float] = []
    resistance_levels: List[float] = []
    
    for level, _ in all_levels:
        if level < current_price:
            support_levels.append(level)
        elif level > current_price:
            resistance_levels.append(level)
    
    # Sort appropriately
    support_levels.sort(reverse=True)  # Nearest first
    resistance_levels.sort()  # Nearest first
    
    return {
        "support_levels": support_levels[:5],  # Top 5
        "resistance_levels": resistance_levels[:5],  # Top 5
    }
