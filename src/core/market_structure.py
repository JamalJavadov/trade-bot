"""
Advanced Market Structure Analysis Module

This module provides professional ICT-style (Inner Circle Trader) market
structure analysis including order blocks, breaker blocks, fair value gaps,
and Smart Money Concepts (SMC) pattern detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class OrderBlock:
    """Represents an institutional order block zone."""
    index: int
    price_high: float
    price_low: float
    side: str  # "BULLISH", "BEARISH"
    is_valid: bool
    strength: float  # 0.0 to 1.0
    touches: int
    mitigated: bool
    volume_ratio: float  # Volume relative to average


@dataclass
class BreakerBlock:
    """Represents a breaker block (failed support/resistance)."""
    index: int
    price_high: float
    price_low: float
    side: str  # "BULLISH", "BEARISH"
    origin_type: str  # "FAILED_SUPPORT", "FAILED_RESISTANCE"
    strength: float


@dataclass
class FairValueGap:
    """Represents a fair value gap (imbalance)."""
    index: int
    price_high: float
    price_low: float
    side: str  # "BULLISH", "BEARISH"
    size: float
    size_atr: float  # Size relative to ATR
    filled: bool
    fill_percentage: float


@dataclass
class StructurePoint:
    """Represents a market structure point (swing high/low)."""
    index: int
    price: float
    point_type: str  # "HH", "HL", "LH", "LL", "HIGH", "LOW"
    is_bos: bool  # Break of structure occurred after
    is_choch: bool  # Change of character


@dataclass
class MarketStructure:
    """Complete market structure analysis result."""
    trend: str  # "BULLISH", "BEARISH", "RANGING"
    trend_strength: float
    structure_points: List[StructurePoint]
    order_blocks: List[OrderBlock]
    breaker_blocks: List[BreakerBlock]
    fair_value_gaps: List[FairValueGap]
    last_bos: Optional[StructurePoint]
    last_choch: Optional[StructurePoint]
    displacement_detected: bool
    details: Dict[str, Any] = field(default_factory=dict)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR as a series."""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _find_swing_points(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
) -> Tuple[List[int], List[int]]:
    """
    Find swing highs and lows using pivot detection.
    
    A swing high is a candle with lower highs on both sides.
    A swing low is a candle with higher lows on both sides.
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    
    swing_highs: List[int] = []
    swing_lows: List[int] = []
    
    for i in range(left_bars, len(df) - right_bars):
        # Check swing high
        is_swing_high = True
        for j in range(1, left_bars + 1):
            if highs[i] <= highs[i - j]:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(1, right_bars + 1):
                if highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
        if is_swing_high:
            swing_highs.append(i)
        
        # Check swing low
        is_swing_low = True
        for j in range(1, left_bars + 1):
            if lows[i] >= lows[i - j]:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(1, right_bars + 1):
                if lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
        if is_swing_low:
            swing_lows.append(i)
    
    return swing_highs, swing_lows


def _classify_structure_points(
    df: pd.DataFrame,
    swing_highs: List[int],
    swing_lows: List[int],
) -> List[StructurePoint]:
    """
    Classify swing points as HH, HL, LH, LL based on sequence.
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    
    # Merge and sort all swing points by index
    all_points: List[Tuple[int, str, float]] = []
    for idx in swing_highs:
        all_points.append((idx, "HIGH", highs[idx]))
    for idx in swing_lows:
        all_points.append((idx, "LOW", lows[idx]))
    all_points.sort(key=lambda x: x[0])
    
    if len(all_points) < 2:
        return [StructurePoint(idx, price, pt, False, False) for idx, pt, price in all_points]
    
    result: List[StructurePoint] = []
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    prev_trend: Optional[str] = None
    
    for i, (idx, point_type, price) in enumerate(all_points):
        classified_type = point_type
        is_bos = False
        is_choch = False
        
        if point_type == "HIGH":
            if last_high is not None:
                if price > last_high:
                    classified_type = "HH"  # Higher High
                    is_bos = prev_trend == "BEARISH"  # BOS if we were bearish
                    if is_bos:
                        is_choch = True
                    prev_trend = "BULLISH"
                else:
                    classified_type = "LH"  # Lower High
                    if prev_trend == "BULLISH":
                        is_choch = True
                    prev_trend = "BEARISH"
            last_high = price
        else:  # LOW
            if last_low is not None:
                if price < last_low:
                    classified_type = "LL"  # Lower Low
                    is_bos = prev_trend == "BULLISH"  # BOS if we were bullish
                    if is_bos:
                        is_choch = True
                    prev_trend = "BEARISH"
                else:
                    classified_type = "HL"  # Higher Low
                    if prev_trend == "BEARISH":
                        is_choch = True
                    prev_trend = "BULLISH"
            last_low = price
        
        result.append(StructurePoint(
            index=idx,
            price=price,
            point_type=classified_type,
            is_bos=is_bos,
            is_choch=is_choch,
        ))
    
    return result


def _detect_order_blocks(
    df: pd.DataFrame,
    swing_highs: List[int],
    swing_lows: List[int],
    atr_series: pd.Series,
    lookback: int = 50,
    min_displacement_atr: float = 1.5,
) -> List[OrderBlock]:
    """
    Detect institutional order blocks.
    
    An order block is the last opposing candle before a strong move (displacement).
    
    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move
    """
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    volumes = df["volume"].astype(float).values
    atr_vals = atr_series.values
    
    order_blocks: List[OrderBlock] = []
    vol_avg = np.nanmean(volumes[-lookback:]) if len(volumes) >= lookback else np.nanmean(volumes)
    
    n = len(df)
    start_idx = max(0, n - lookback)
    
    for i in range(start_idx, n - 3):
        atr_val = atr_vals[i] if np.isfinite(atr_vals[i]) else 1.0
        
        # Check for bullish order block
        # Current candle is bearish, followed by strong bullish displacement
        if closes[i] < opens[i]:  # Bearish candle
            # Check for displacement (strong move up)
            future_move = 0.0
            for j in range(i + 1, min(i + 4, n)):
                future_move += closes[j] - opens[j]
            
            if future_move >= atr_val * min_displacement_atr:
                vol_ratio = volumes[i] / vol_avg if vol_avg > 0 else 1.0
                
                # Check if OB has been mitigated (price returned to zone)
                mitigated = False
                touches = 0
                ob_high = highs[i]
                ob_low = lows[i]
                
                for j in range(i + 1, n):
                    if lows[j] <= ob_high and highs[j] >= ob_low:
                        touches += 1
                        if lows[j] < ob_low:
                            mitigated = True
                            break
                
                # Strength based on displacement size and volume
                strength = min(1.0, (future_move / atr_val) / 3.0)
                if vol_ratio > 1.5:
                    strength = min(1.0, strength * 1.2)
                
                order_blocks.append(OrderBlock(
                    index=i,
                    price_high=ob_high,
                    price_low=ob_low,
                    side="BULLISH",
                    is_valid=not mitigated,
                    strength=strength,
                    touches=touches,
                    mitigated=mitigated,
                    volume_ratio=vol_ratio,
                ))
        
        # Check for bearish order block
        if closes[i] > opens[i]:  # Bullish candle
            # Check for displacement (strong move down)
            future_move = 0.0
            for j in range(i + 1, min(i + 4, n)):
                future_move += opens[j] - closes[j]
            
            if future_move >= atr_val * min_displacement_atr:
                vol_ratio = volumes[i] / vol_avg if vol_avg > 0 else 1.0
                
                # Check if OB has been mitigated
                mitigated = False
                touches = 0
                ob_high = highs[i]
                ob_low = lows[i]
                
                for j in range(i + 1, n):
                    if lows[j] <= ob_high and highs[j] >= ob_low:
                        touches += 1
                        if highs[j] > ob_high:
                            mitigated = True
                            break
                
                strength = min(1.0, (future_move / atr_val) / 3.0)
                if vol_ratio > 1.5:
                    strength = min(1.0, strength * 1.2)
                
                order_blocks.append(OrderBlock(
                    index=i,
                    price_high=ob_high,
                    price_low=ob_low,
                    side="BEARISH",
                    is_valid=not mitigated,
                    strength=strength,
                    touches=touches,
                    mitigated=mitigated,
                    volume_ratio=vol_ratio,
                ))
    
    return order_blocks


def _detect_breaker_blocks(
    df: pd.DataFrame,
    structure_points: List[StructurePoint],
    atr_series: pd.Series,
) -> List[BreakerBlock]:
    """
    Detect breaker blocks (failed support/resistance levels).
    
    A breaker block forms when:
    - An order block fails (gets mitigated)
    - The level then acts as the opposite (support becomes resistance or vice versa)
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    
    breaker_blocks: List[BreakerBlock] = []
    
    # Find ChoCH points as potential breaker locations
    choch_points = [sp for sp in structure_points if sp.is_choch]
    
    for sp in choch_points:
        if sp.index >= len(df):
            continue
            
        atr_val = atr_series.iloc[sp.index] if sp.index < len(atr_series) else 1.0
        if not np.isfinite(atr_val):
            atr_val = 1.0
        
        if sp.point_type in ("LH", "HH"):  # High point that caused ChoCH
            # This could be a bearish breaker (old resistance becomes new resistance)
            breaker_blocks.append(BreakerBlock(
                index=sp.index,
                price_high=highs[sp.index],
                price_low=lows[sp.index],
                side="BEARISH",
                origin_type="FAILED_SUPPORT",
                strength=0.7,
            ))
        elif sp.point_type in ("HL", "LL"):  # Low point that caused ChoCH
            # This could be a bullish breaker (old support becomes new support)
            breaker_blocks.append(BreakerBlock(
                index=sp.index,
                price_high=highs[sp.index],
                price_low=lows[sp.index],
                side="BULLISH",
                origin_type="FAILED_RESISTANCE",
                strength=0.7,
            ))
    
    return breaker_blocks


def _detect_fair_value_gaps(
    df: pd.DataFrame,
    atr_series: pd.Series,
    lookback: int = 50,
    min_size_atr: float = 0.3,
) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (FVGs / Imbalances).
    
    Bullish FVG: Gap between candle 1's high and candle 3's low
    Bearish FVG: Gap between candle 1's low and candle 3's high
    """
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    atr_vals = atr_series.values
    
    fair_value_gaps: List[FairValueGap] = []
    n = len(df)
    start_idx = max(0, n - lookback)
    
    for i in range(start_idx + 2, n):
        atr_val = atr_vals[i] if np.isfinite(atr_vals[i]) else 1.0
        
        # Bullish FVG: Candle 1 high < Candle 3 low (gap up)
        if highs[i - 2] < lows[i]:
            gap_low = highs[i - 2]
            gap_high = lows[i]
            gap_size = gap_high - gap_low
            
            if gap_size >= atr_val * min_size_atr:
                # Check if FVG has been filled
                filled = False
                fill_percentage = 0.0
                
                for j in range(i + 1, n):
                    if lows[j] <= gap_high:
                        fill_pct = (gap_high - lows[j]) / gap_size
                        fill_percentage = max(fill_percentage, fill_pct)
                        if lows[j] <= gap_low:
                            filled = True
                            fill_percentage = 1.0
                            break
                
                fair_value_gaps.append(FairValueGap(
                    index=i - 1,  # Middle candle
                    price_high=gap_high,
                    price_low=gap_low,
                    side="BULLISH",
                    size=gap_size,
                    size_atr=gap_size / atr_val,
                    filled=filled,
                    fill_percentage=fill_percentage,
                ))
        
        # Bearish FVG: Candle 1 low > Candle 3 high (gap down)
        if lows[i - 2] > highs[i]:
            gap_high = lows[i - 2]
            gap_low = highs[i]
            gap_size = gap_high - gap_low
            
            if gap_size >= atr_val * min_size_atr:
                # Check if FVG has been filled
                filled = False
                fill_percentage = 0.0
                
                for j in range(i + 1, n):
                    if highs[j] >= gap_low:
                        fill_pct = (highs[j] - gap_low) / gap_size
                        fill_percentage = max(fill_percentage, fill_pct)
                        if highs[j] >= gap_high:
                            filled = True
                            fill_percentage = 1.0
                            break
                
                fair_value_gaps.append(FairValueGap(
                    index=i - 1,  # Middle candle
                    price_high=gap_high,
                    price_low=gap_low,
                    side="BEARISH",
                    size=gap_size,
                    size_atr=gap_size / atr_val,
                    filled=filled,
                    fill_percentage=fill_percentage,
                ))
    
    return fair_value_gaps


def _detect_displacement(
    df: pd.DataFrame,
    atr_series: pd.Series,
    lookback: int = 10,
    min_displacement_atr: float = 2.0,
) -> Tuple[bool, Optional[str]]:
    """
    Detect recent displacement (strong momentum move).
    
    Displacement is characterized by:
    - Large body candles
    - Little to no wicks
    - High volume
    """
    if len(df) < lookback:
        return False, None
    
    opens = df["open"].astype(float).values
    closes = df["close"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    atr_vals = atr_series.values
    
    n = len(df)
    
    for i in range(n - lookback, n):
        atr_val = atr_vals[i] if np.isfinite(atr_vals[i]) else 1.0
        
        body = abs(closes[i] - opens[i])
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        total_wick = upper_wick + lower_wick
        
        # Strong displacement: Large body, small wicks
        if body >= atr_val * min_displacement_atr:
            if total_wick < body * 0.3:  # Wicks less than 30% of body
                if closes[i] > opens[i]:
                    return True, "BULLISH"
                else:
                    return True, "BEARISH"
    
    return False, None


def _determine_trend(
    structure_points: List[StructurePoint],
) -> Tuple[str, float]:
    """
    Determine overall trend from structure points.
    """
    if len(structure_points) < 3:
        return "RANGING", 0.5
    
    recent_points = structure_points[-10:] if len(structure_points) > 10 else structure_points
    
    hh_count = sum(1 for sp in recent_points if sp.point_type == "HH")
    hl_count = sum(1 for sp in recent_points if sp.point_type == "HL")
    lh_count = sum(1 for sp in recent_points if sp.point_type == "LH")
    ll_count = sum(1 for sp in recent_points if sp.point_type == "LL")
    
    bullish_score = hh_count + hl_count
    bearish_score = lh_count + ll_count
    total = bullish_score + bearish_score
    
    if total == 0:
        return "RANGING", 0.5
    
    if bullish_score > bearish_score * 1.5:
        strength = min(1.0, bullish_score / total)
        return "BULLISH", strength
    elif bearish_score > bullish_score * 1.5:
        strength = min(1.0, bearish_score / total)
        return "BEARISH", strength
    else:
        return "RANGING", 0.5


def analyze_market_structure(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> MarketStructure:
    """
    Perform comprehensive market structure analysis.
    
    This is the main entry point for ICT-style market structure analysis.
    """
    config = config or {}
    
    # Get configuration parameters
    swing_left_bars = config.get("swing_left_bars", 3)
    swing_right_bars = config.get("swing_right_bars", 3)
    ob_lookback = config.get("order_block_lookback", 50)
    fvg_lookback = config.get("fvg_lookback", 50)
    min_displacement_atr = config.get("min_displacement_atr", 1.5)
    min_fvg_size_atr = config.get("fvg_min_size_atr", 0.3)
    
    # Calculate ATR
    atr_series = _atr(df, period=14)
    
    # Find swing points
    swing_highs, swing_lows = _find_swing_points(df, swing_left_bars, swing_right_bars)
    
    # Classify structure points
    structure_points = _classify_structure_points(df, swing_highs, swing_lows)
    
    # Determine trend
    trend, trend_strength = _determine_trend(structure_points)
    
    # Detect order blocks
    order_blocks = _detect_order_blocks(
        df, swing_highs, swing_lows, atr_series,
        lookback=ob_lookback,
        min_displacement_atr=min_displacement_atr,
    )
    
    # Detect breaker blocks
    breaker_blocks = _detect_breaker_blocks(df, structure_points, atr_series)
    
    # Detect fair value gaps
    fair_value_gaps = _detect_fair_value_gaps(
        df, atr_series,
        lookback=fvg_lookback,
        min_size_atr=min_fvg_size_atr,
    )
    
    # Find last BOS and ChoCH
    last_bos = None
    last_choch = None
    for sp in reversed(structure_points):
        if sp.is_bos and last_bos is None:
            last_bos = sp
        if sp.is_choch and last_choch is None:
            last_choch = sp
        if last_bos and last_choch:
            break
    
    # Detect displacement
    displacement_detected, displacement_side = _detect_displacement(
        df, atr_series,
        min_displacement_atr=min_displacement_atr * 1.5,
    )
    
    # Compile details
    details: Dict[str, Any] = {
        "swing_highs_count": len(swing_highs),
        "swing_lows_count": len(swing_lows),
        "valid_order_blocks": sum(1 for ob in order_blocks if ob.is_valid),
        "total_order_blocks": len(order_blocks),
        "unfilled_fvgs": sum(1 for fvg in fair_value_gaps if not fvg.filled),
        "total_fvgs": len(fair_value_gaps),
        "displacement_side": displacement_side,
    }
    
    return MarketStructure(
        trend=trend,
        trend_strength=trend_strength,
        structure_points=structure_points,
        order_blocks=order_blocks,
        breaker_blocks=breaker_blocks,
        fair_value_gaps=fair_value_gaps,
        last_bos=last_bos,
        last_choch=last_choch,
        displacement_detected=displacement_detected,
        details=details,
    )


def get_nearest_zones(
    market_structure: MarketStructure,
    current_price: float,
    side: str,
    max_zones: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get the nearest relevant zones for a given trade side.
    
    Args:
        market_structure: Analyzed market structure
        current_price: Current market price
        side: "LONG" or "SHORT"
        max_zones: Maximum number of zones to return per type
    
    Returns:
        Dictionary with nearest order blocks and FVGs
    """
    result: Dict[str, List[Dict[str, Any]]] = {
        "order_blocks": [],
        "fair_value_gaps": [],
    }
    
    # For LONG, we want bullish OBs and bullish FVGs below price
    # For SHORT, we want bearish OBs and bearish FVGs above price
    
    target_side = "BULLISH" if side == "LONG" else "BEARISH"
    
    # Filter and sort order blocks
    relevant_obs = [
        ob for ob in market_structure.order_blocks
        if ob.side == target_side and ob.is_valid
    ]
    
    if side == "LONG":
        # Get zones below current price
        relevant_obs = [ob for ob in relevant_obs if ob.price_high < current_price]
        relevant_obs.sort(key=lambda x: x.price_high, reverse=True)  # Nearest first
    else:
        # Get zones above current price
        relevant_obs = [ob for ob in relevant_obs if ob.price_low > current_price]
        relevant_obs.sort(key=lambda x: x.price_low)  # Nearest first
    
    for ob in relevant_obs[:max_zones]:
        result["order_blocks"].append({
            "high": ob.price_high,
            "low": ob.price_low,
            "mid": (ob.price_high + ob.price_low) / 2,
            "strength": ob.strength,
            "touches": ob.touches,
            "volume_ratio": ob.volume_ratio,
        })
    
    # Filter and sort FVGs
    relevant_fvgs = [
        fvg for fvg in market_structure.fair_value_gaps
        if fvg.side == target_side and not fvg.filled
    ]
    
    if side == "LONG":
        relevant_fvgs = [fvg for fvg in relevant_fvgs if fvg.price_high < current_price]
        relevant_fvgs.sort(key=lambda x: x.price_high, reverse=True)
    else:
        relevant_fvgs = [fvg for fvg in relevant_fvgs if fvg.price_low > current_price]
        relevant_fvgs.sort(key=lambda x: x.price_low)
    
    for fvg in relevant_fvgs[:max_zones]:
        result["fair_value_gaps"].append({
            "high": fvg.price_high,
            "low": fvg.price_low,
            "mid": (fvg.price_high + fvg.price_low) / 2,
            "size_atr": fvg.size_atr,
            "fill_percentage": fvg.fill_percentage,
        })
    
    return result


def calculate_structure_score(
    market_structure: MarketStructure,
    side: str,
    current_price: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a normalized structure score for the given trade side.
    
    Returns:
        Tuple of (score 0-1, details dict)
    """
    score = 0.0
    max_score = 0.0
    details: Dict[str, Any] = {}
    
    # Trend alignment (weight: 3)
    max_score += 3.0
    if side == "LONG" and market_structure.trend == "BULLISH":
        score += 3.0 * market_structure.trend_strength
        details["trend_aligned"] = True
    elif side == "SHORT" and market_structure.trend == "BEARISH":
        score += 3.0 * market_structure.trend_strength
        details["trend_aligned"] = True
    elif market_structure.trend == "RANGING":
        score += 1.5  # Neutral
        details["trend_aligned"] = None
    else:
        details["trend_aligned"] = False
    
    # Valid order blocks available (weight: 2)
    max_score += 2.0
    zones = get_nearest_zones(market_structure, current_price, side)
    if zones["order_blocks"]:
        avg_ob_strength = sum(ob["strength"] for ob in zones["order_blocks"]) / len(zones["order_blocks"])
        score += 2.0 * avg_ob_strength
        details["order_blocks_available"] = len(zones["order_blocks"])
    else:
        details["order_blocks_available"] = 0
    
    # Unfilled FVGs (weight: 1.5)
    max_score += 1.5
    if zones["fair_value_gaps"]:
        score += 1.5
        details["fvgs_available"] = len(zones["fair_value_gaps"])
    else:
        details["fvgs_available"] = 0
    
    # Recent structure confirmation (weight: 2)
    max_score += 2.0
    if market_structure.last_bos:
        if side == "LONG" and market_structure.last_bos.point_type in ("HH", "HL"):
            score += 2.0
            details["bos_confirms"] = True
        elif side == "SHORT" and market_structure.last_bos.point_type in ("LH", "LL"):
            score += 2.0
            details["bos_confirms"] = True
        else:
            details["bos_confirms"] = False
    else:
        details["bos_confirms"] = None
    
    # Displacement (weight: 1.5)
    max_score += 1.5
    if market_structure.displacement_detected:
        displacement_side = market_structure.details.get("displacement_side")
        if (side == "LONG" and displacement_side == "BULLISH") or \
           (side == "SHORT" and displacement_side == "BEARISH"):
            score += 1.5
            details["displacement_aligned"] = True
        else:
            details["displacement_aligned"] = False
    else:
        details["displacement_aligned"] = None
    
    normalized_score = score / max_score if max_score > 0 else 0.0
    details["raw_score"] = score
    details["max_score"] = max_score
    
    return normalized_score, details
