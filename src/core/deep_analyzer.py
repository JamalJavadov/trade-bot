"""
Deep Analysis Integration Module

This module integrates all analysis components (indicators, market structure,
volume analysis, MTF analysis) into a unified professional analysis system
with comprehensive scoring and probability calculation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple

import numpy as np
import pandas as pd

from .indicators import calculate_all_indicators, IndicatorSummary
from .market_structure import analyze_market_structure, calculate_structure_score, MarketStructure
from .volume_analysis import analyze_volume, calculate_volume_score, VolumeAnalysis
from .mtf_analyzer import analyze_mtf_alignment, calculate_mtf_score, MTFAlignment


@dataclass
class DeepAnalysisResult:
    """Complete deep analysis result with all components."""
    symbol: str
    side: str  # "LONG", "SHORT", "NONE"
    signal: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    confidence: float  # 0-100%
    quality_score: float  # 0-100%
    
    # Component scores (0-1)
    indicator_score: float
    structure_score: float
    volume_score: float
    mtf_score: float
    market_data_score: float
    
    # Component analysis objects
    indicators: Optional[IndicatorSummary]
    market_structure: Optional[MarketStructure]
    volume_analysis: Optional[VolumeAnalysis]
    mtf_alignment: Optional[MTFAlignment]
    market_data: Optional[Dict[str, Any]]
    
    # Trade setup
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_1: float
    risk_reward_2: float
    
    # Reasoning
    reasons: List[str]
    warnings: List[str]
    
    # Detailed breakdown
    details: Dict[str, Any] = field(default_factory=dict)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp value between low and high."""
    return max(low, min(high, value))


def _calculate_entry_sl_tp(
    df: pd.DataFrame,
    side: str,
    market_structure: Optional[MarketStructure],
    volume_analysis: Optional[VolumeAnalysis],
    atr: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate optimal entry, stop loss, and take profit levels.
    """
    current_price = float(df["close"].iloc[-1])
    
    if atr <= 0:
        atr = current_price * 0.02  # Default 2% volatility
    
    # Default levels based on ATR
    if side == "LONG":
        entry = current_price
        sl = current_price - atr * 1.5
        tp1 = current_price + atr * 2.0
        tp2 = current_price + atr * 4.0
        
        # Try to improve with structure zones
        if market_structure and market_structure.order_blocks:
            bullish_obs = [ob for ob in market_structure.order_blocks 
                          if ob.side == "BULLISH" and ob.is_valid and ob.price_high < current_price]
            if bullish_obs:
                nearest_ob = max(bullish_obs, key=lambda x: x.price_high)
                entry = (nearest_ob.price_high + nearest_ob.price_low) / 2
                sl = nearest_ob.price_low - atr * 0.5
        
        # Use volume levels for TP
        if volume_analysis and volume_analysis.profile:
            if volume_analysis.profile.vah > current_price:
                tp1 = volume_analysis.profile.vah
            
    else:  # SHORT
        entry = current_price
        sl = current_price + atr * 1.5
        tp1 = current_price - atr * 2.0
        tp2 = current_price - atr * 4.0
        
        # Try to improve with structure zones
        if market_structure and market_structure.order_blocks:
            bearish_obs = [ob for ob in market_structure.order_blocks 
                          if ob.side == "BEARISH" and ob.is_valid and ob.price_low > current_price]
            if bearish_obs:
                nearest_ob = min(bearish_obs, key=lambda x: x.price_low)
                entry = (nearest_ob.price_high + nearest_ob.price_low) / 2
                sl = nearest_ob.price_high + atr * 0.5
        
        # Use volume levels for TP
        if volume_analysis and volume_analysis.profile:
            if volume_analysis.profile.val < current_price:
                tp1 = volume_analysis.profile.val
    
    return entry, sl, tp1, tp2


def _calculate_rr(entry: float, sl: float, tp: float) -> float:
    """Calculate risk/reward ratio."""
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return 0.0
    return reward / risk


def _calculate_market_data_score(
    market_data: Optional[Dict[str, Any]],
    side: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate score from market data (funding, OI, order book).
    """
    if not market_data:
        return 0.5, {"available": False}
    
    score = 0.0
    max_score = 0.0
    details: Dict[str, Any] = {"available": True}
    
    # Order book pressure (weight: 2)
    max_score += 2.0
    order_book = market_data.get("orderBook", {})
    pressure = order_book.get("pressure", "NEUTRAL")
    
    if (side == "LONG" and pressure == "BUYING") or (side == "SHORT" and pressure == "SELLING"):
        score += 2.0
        details["order_book_aligned"] = True
    elif pressure == "NEUTRAL":
        score += 1.0
        details["order_book_aligned"] = None
    else:
        details["order_book_aligned"] = False
    
    # Funding rate contrarian (weight: 1.5)
    max_score += 1.5
    funding = market_data.get("funding", {})
    funding_sentiment = funding.get("sentiment", "NEUTRAL")
    
    # Contrarian: crowded longs = bearish, crowded shorts = bullish
    if side == "LONG" and funding_sentiment == "BEARISH_CROWDED":
        score += 1.5
        details["funding_contrarian"] = True
    elif side == "SHORT" and funding_sentiment == "BULLISH_CROWDED":
        score += 1.5
        details["funding_contrarian"] = True
    elif funding_sentiment == "NEUTRAL":
        score += 0.75
        details["funding_contrarian"] = None
    else:
        details["funding_contrarian"] = False
    
    # Taker ratio (weight: 1.5)
    max_score += 1.5
    taker = market_data.get("takerRatio", {})
    taker_pressure = taker.get("pressure", "BALANCED")
    
    if (side == "LONG" and taker_pressure == "BUYING") or (side == "SHORT" and taker_pressure == "SELLING"):
        score += 1.5
        details["taker_aligned"] = True
    elif taker_pressure == "BALANCED":
        score += 0.75
        details["taker_aligned"] = None
    else:
        details["taker_aligned"] = False
    
    normalized_score = score / max_score if max_score > 0 else 0.5
    details["raw_score"] = score
    details["max_score"] = max_score
    
    return normalized_score, details


def _aggregate_signal(
    indicator_signal: str,
    structure_trend: str,
    volume_signal: str,
    mtf_bias: str,
) -> Tuple[str, str]:
    """
    Aggregate signals from all components to determine overall signal and side.
    """
    bullish_count = 0
    bearish_count = 0
    
    # Indicator signal
    if indicator_signal in ("STRONG_BUY", "BUY"):
        bullish_count += 2 if indicator_signal == "STRONG_BUY" else 1
    elif indicator_signal in ("STRONG_SELL", "SELL"):
        bearish_count += 2 if indicator_signal == "STRONG_SELL" else 1
    
    # Structure trend
    if structure_trend == "BULLISH":
        bullish_count += 1
    elif structure_trend == "BEARISH":
        bearish_count += 1
    
    # Volume signal
    if volume_signal == "ACCUMULATION":
        bullish_count += 1
    elif volume_signal == "DISTRIBUTION":
        bearish_count += 1
    
    # MTF bias
    if mtf_bias == "BULLISH":
        bullish_count += 2
    elif mtf_bias == "BEARISH":
        bearish_count += 2
    
    # Determine signal
    total = bullish_count + bearish_count
    if total == 0:
        return "NEUTRAL", "NONE"
    
    if bullish_count >= 5:
        return "STRONG_BUY", "LONG"
    elif bullish_count > bearish_count:
        return "BUY", "LONG"
    elif bearish_count >= 5:
        return "STRONG_SELL", "SHORT"
    elif bearish_count > bullish_count:
        return "SELL", "SHORT"
    else:
        return "NEUTRAL", "NONE"


def _calculate_confidence(
    indicator_score: float,
    structure_score: float,
    volume_score: float,
    mtf_score: float,
    market_data_score: float,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Calculate overall confidence percentage using weighted components.
    """
    config = config or {}
    
    # Weights for each component
    w_indicator = config.get("w_indicator", 2.5)
    w_structure = config.get("w_structure", 3.0)
    w_volume = config.get("w_volume", 2.0)
    w_mtf = config.get("w_mtf", 3.0)
    w_market = config.get("w_market_data", 1.5)
    
    total_weight = w_indicator + w_structure + w_volume + w_mtf + w_market
    
    weighted_sum = (
        indicator_score * w_indicator +
        structure_score * w_structure +
        volume_score * w_volume +
        mtf_score * w_mtf +
        market_data_score * w_market
    )
    
    confidence = (weighted_sum / total_weight) * 100
    return _clamp(confidence, 5.0, 95.0)


def _generate_reasons(
    side: str,
    indicators: Optional[IndicatorSummary],
    market_structure: Optional[MarketStructure],
    volume_analysis: Optional[VolumeAnalysis],
    mtf_alignment: Optional[MTFAlignment],
    market_data: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    Generate human-readable reasons and warnings.
    """
    reasons: List[str] = []
    warnings: List[str] = []
    
    target_trend = "BULLISH" if side == "LONG" else "BEARISH"
    
    # Indicator reasons
    if indicators:
        if indicators.overall_signal in ("STRONG_BUY", "BUY") and side == "LONG":
            reasons.append(f"Technical indicators favor {side}: {indicators.bullish_count} bullish, {indicators.bearish_count} bearish")
        elif indicators.overall_signal in ("STRONG_SELL", "SELL") and side == "SHORT":
            reasons.append(f"Technical indicators favor {side}: {indicators.bearish_count} bearish, {indicators.bullish_count} bullish")
        
        # Check for specific indicator signals
        if "macd" in indicators.indicators:
            macd = indicators.indicators["macd"]
            if macd.details and macd.details.get("crossover"):
                reasons.append(f"MACD {macd.details['crossover'].replace('_', ' ').lower()}")
            if macd.details and macd.details.get("divergence"):
                reasons.append(f"MACD {macd.details['divergence'].replace('_', ' ').lower()} detected")
        
        if "adx" in indicators.indicators:
            adx = indicators.indicators["adx"]
            if adx.details and adx.details.get("adx_value", 0) >= 25:
                reasons.append(f"Strong trend confirmed (ADX={adx.details['adx_value']:.1f})")
    
    # Structure reasons
    if market_structure:
        if market_structure.trend == target_trend:
            reasons.append(f"Market structure is {target_trend.lower()} (trend strength: {market_structure.trend_strength:.0%})")
        else:
            warnings.append(f"Market structure shows {market_structure.trend.lower()} bias - counter-trend trade")
        
        valid_obs = [ob for ob in market_structure.order_blocks if ob.is_valid]
        matching_obs = [ob for ob in valid_obs if ob.side == target_trend]
        if matching_obs:
            reasons.append(f"{len(matching_obs)} valid {target_trend.lower()} order block(s) detected")
        
        unfilled_fvgs = [fvg for fvg in market_structure.fair_value_gaps if not fvg.filled]
        if unfilled_fvgs:
            reasons.append(f"{len(unfilled_fvgs)} unfilled fair value gap(s) for potential entry")
        
        if market_structure.displacement_detected:
            reasons.append("Recent displacement (momentum) detected")
    
    # Volume reasons
    if volume_analysis:
        if (side == "LONG" and volume_analysis.signal == "ACCUMULATION") or \
           (side == "SHORT" and volume_analysis.signal == "DISTRIBUTION"):
            reasons.append(f"Volume analysis shows {volume_analysis.signal.lower()}")
        
        if volume_analysis.delta.divergence:
            reasons.append(f"Volume delta {volume_analysis.delta.divergence.replace('_', ' ').lower()}")
        
        if volume_analysis.volume_trend == "INCREASING":
            reasons.append("Volume trending higher (confirms conviction)")
        elif volume_analysis.volume_trend == "DECREASING":
            warnings.append("Volume declining - watch for false breakouts")
    
    # MTF reasons
    if mtf_alignment:
        if (side == "LONG" and mtf_alignment.overall_bias == "BULLISH") or \
           (side == "SHORT" and mtf_alignment.overall_bias == "BEARISH"):
            reasons.append(f"MTF alignment: {mtf_alignment.aligned_timeframes}/{mtf_alignment.total_timeframes} timeframes agree")
        else:
            warnings.append(f"Limited MTF alignment ({mtf_alignment.aligned_timeframes}/{mtf_alignment.total_timeframes})")
        
        if mtf_alignment.confluence_zones:
            reasons.append(f"{len(mtf_alignment.confluence_zones)} confluence zone(s) across timeframes")
    
    # Market data reasons
    if market_data:
        funding = market_data.get("funding", {})
        if funding.get("sentiment") == "BEARISH_CROWDED" and side == "LONG":
            reasons.append("Funding rate negative (shorts crowded) - contrarian bullish")
        elif funding.get("sentiment") == "BULLISH_CROWDED" and side == "SHORT":
            reasons.append("Funding rate positive (longs crowded) - contrarian bearish")
        elif funding.get("sentiment") == "BULLISH_CROWDED" and side == "LONG":
            warnings.append("Funding rate elevated - longs may be crowded")
        elif funding.get("sentiment") == "BEARISH_CROWDED" and side == "SHORT":
            warnings.append("Funding rate negative - shorts may be crowded")
        
        order_book = market_data.get("orderBook", {})
        if (side == "LONG" and order_book.get("pressure") == "BUYING") or \
           (side == "SHORT" and order_book.get("pressure") == "SELLING"):
            reasons.append(f"Order book shows {order_book['pressure'].lower()} pressure")
    
    return reasons, warnings


def perform_deep_analysis(
    symbol: str,
    fetch_ohlcv: Callable[[str, str, int], pd.DataFrame],
    settings: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Any]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> DeepAnalysisResult:
    """
    Perform comprehensive deep analysis on a symbol.
    
    This is the main entry point for professional-grade analysis,
    integrating all components into a unified result.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        fetch_ohlcv: Function to fetch OHLCV data (symbol, timeframe, limit) -> DataFrame
        settings: Optional configuration settings
        market_data: Optional pre-fetched market data from binance_data
        on_stage: Optional callback for progress updates
    
    Returns:
        DeepAnalysisResult with comprehensive analysis
    """
    settings = settings or {}
    deep_config = settings.get("deep_analysis", {})
    scoring_config = settings.get("advanced_scoring", {})
    
    # Fetch primary timeframe data
    if on_stage:
        on_stage("Fetching data")
    
    primary_tf = settings.get("strategy", {}).get("impulse_timeframe", "1h")
    
    try:
        df = fetch_ohlcv(symbol, primary_tf, 500)
    except Exception as e:
        return DeepAnalysisResult(
            symbol=symbol,
            side="NONE",
            signal="NEUTRAL",
            confidence=0.0,
            quality_score=0.0,
            indicator_score=0.0,
            structure_score=0.0,
            volume_score=0.0,
            mtf_score=0.0,
            market_data_score=0.0,
            indicators=None,
            market_structure=None,
            volume_analysis=None,
            mtf_alignment=None,
            market_data=None,
            entry=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_1=0.0,
            risk_reward_2=0.0,
            reasons=[],
            warnings=[f"Failed to fetch data: {e}"],
            details={"error": str(e)},
        )
    
    if len(df) < 50:
        return DeepAnalysisResult(
            symbol=symbol,
            side="NONE",
            signal="NEUTRAL",
            confidence=0.0,
            quality_score=0.0,
            indicator_score=0.0,
            structure_score=0.0,
            volume_score=0.0,
            mtf_score=0.0,
            market_data_score=0.0,
            indicators=None,
            market_structure=None,
            volume_analysis=None,
            mtf_alignment=None,
            market_data=None,
            entry=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_1=0.0,
            risk_reward_2=0.0,
            reasons=[],
            warnings=["Insufficient data for analysis"],
            details={"error": "Insufficient data"},
        )
    
    current_price = float(df["close"].iloc[-1])
    
    # Calculate ATR
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        atr = current_price * 0.02
    
    # 1. Technical Indicators Analysis
    if on_stage:
        on_stage("Analyzing indicators")
    
    indicators = calculate_all_indicators(df, deep_config)
    
    # 2. Market Structure Analysis
    if on_stage:
        on_stage("Analyzing market structure")
    
    market_structure = analyze_market_structure(df, settings.get("market_structure", {}))
    
    # 3. Volume Analysis
    if on_stage:
        on_stage("Analyzing volume")
    
    volume_analysis = analyze_volume(df, settings.get("volume_profile", {}))
    
    # 4. Multi-Timeframe Analysis
    if on_stage:
        on_stage("Analyzing MTF alignment")
    
    try:
        mtf_alignment = analyze_mtf_alignment(fetch_ohlcv, symbol)
    except Exception:
        mtf_alignment = None
    
    # Aggregate signals to determine side
    indicator_signal = indicators.overall_signal if indicators else "NEUTRAL"
    structure_trend = market_structure.trend if market_structure else "NEUTRAL"
    volume_signal = volume_analysis.signal if volume_analysis else "NEUTRAL"
    mtf_bias = mtf_alignment.overall_bias if mtf_alignment else "NEUTRAL"
    
    signal, side = _aggregate_signal(indicator_signal, structure_trend, volume_signal, mtf_bias)
    
    if side == "NONE":
        # Default to structure trend if no clear side
        if structure_trend == "BULLISH":
            side = "LONG"
        elif structure_trend == "BEARISH":
            side = "SHORT"
        else:
            side = "LONG"  # Default fallback
    
    # 5. Calculate component scores
    if on_stage:
        on_stage("Calculating scores")
    
    # Indicator score
    indicator_score = indicators.overall_strength if indicators else 0.5
    if indicators:
        target_count = indicators.bullish_count if side == "LONG" else indicators.bearish_count
        total = indicators.bullish_count + indicators.bearish_count + indicators.neutral_count
        if total > 0:
            indicator_score = _clamp(target_count / total + indicator_score * 0.5)
    
    # Structure score
    structure_score_val, structure_details = calculate_structure_score(
        market_structure, side, current_price
    ) if market_structure else (0.5, {})
    
    # Volume score
    volume_score_val, volume_details = calculate_volume_score(
        volume_analysis, side, current_price
    ) if volume_analysis else (0.5, {})
    
    # MTF score
    mtf_score_val, mtf_details = calculate_mtf_score(
        mtf_alignment, side
    ) if mtf_alignment else (0.5, {})
    
    # Market data score
    market_data_score_val, market_data_details = _calculate_market_data_score(market_data, side)
    
    # 6. Calculate confidence
    confidence = _calculate_confidence(
        indicator_score,
        structure_score_val,
        volume_score_val,
        mtf_score_val,
        market_data_score_val,
        scoring_config,
    )
    
    # 7. Calculate entry, SL, TP levels
    entry, sl, tp1, tp2 = _calculate_entry_sl_tp(
        df, side, market_structure, volume_analysis, atr
    )
    
    rr1 = _calculate_rr(entry, sl, tp1)
    rr2 = _calculate_rr(entry, sl, tp2)

    # Enforce strict 3:1 RR if enabled
    risk_cfg = settings.get("risk", {})
    enforce_strict = bool(risk_cfg.get("enforce_strict_rr", False))
    rr_buffer = float(risk_cfg.get("rr_buffer_pct", 1.0))
    if enforce_strict and side in ("LONG", "SHORT"):
        risk_dist = abs(entry - sl)
        if risk_dist > 0:
            direction = 1 if side == "LONG" else -1
            target_rr = 3.0 * rr_buffer
            tp2 = float(entry + direction * (risk_dist * target_rr))
            rr2 = _calculate_rr(entry, sl, tp2)
            # Adjust TP1 to 1:1 if it was further
            if _calculate_rr(entry, sl, tp1) > 1.0:
                tp1 = float(entry + direction * risk_dist)
                rr1 = 1.0

    # 8. Generate reasons and warnings
    reasons, warnings = _generate_reasons(
        side, indicators, market_structure, volume_analysis, mtf_alignment, market_data
    )
    if enforce_strict and side in ("LONG", "SHORT"):
        reasons.append(f"Strict 3:1 RR applied (buffer={rr_buffer})")
    
    # Calculate quality score (similar to confidence but with RR factor)
    quality_score = confidence
    if rr2 >= 3.0:
        quality_score = min(95.0, quality_score * 1.1)
    elif rr2 < 2.0:
        quality_score = quality_score * 0.9
        warnings.append(f"Risk/Reward ratio ({rr2:.2f}) is below ideal (>2.0)")
    
    # Compile details
    details: Dict[str, Any] = {
        "current_price": current_price,
        "atr": atr,
        "primary_timeframe": primary_tf,
        "indicator_details": {
            "signal": indicator_signal,
            "bullish": indicators.bullish_count if indicators else 0,
            "bearish": indicators.bearish_count if indicators else 0,
            "neutral": indicators.neutral_count if indicators else 0,
        },
        "structure_details": structure_details,
        "volume_details": volume_details,
        "mtf_details": mtf_details,
        "market_data_details": market_data_details,
        "score_weights": scoring_config,
    }
    
    return DeepAnalysisResult(
        symbol=symbol,
        side=side,
        signal=signal,
        confidence=confidence,
        quality_score=quality_score,
        indicator_score=indicator_score,
        structure_score=structure_score_val,
        volume_score=volume_score_val,
        mtf_score=mtf_score_val,
        market_data_score=market_data_score_val,
        indicators=indicators,
        market_structure=market_structure,
        volume_analysis=volume_analysis,
        mtf_alignment=mtf_alignment,
        market_data=market_data,
        entry=entry,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_reward_1=rr1,
        risk_reward_2=rr2,
        reasons=reasons,
        warnings=warnings,
        details=details,
    )


def format_deep_analysis_report(result: DeepAnalysisResult) -> str:
    """
    Format deep analysis result as a professional text report.
    """
    lines: List[str] = []
    
    # Header
    lines.append("=" * 60)
    lines.append("     PROFESSIONAL DEEP ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Market Overview
    lines.append("📊 MARKET OVERVIEW")
    lines.append(f"├─ Symbol: {result.symbol}")
    lines.append(f"├─ Current Price: ${result.details.get('current_price', 0):,.4f}")
    lines.append(f"├─ Signal: {result.signal}")
    lines.append(f"└─ Side: {result.side}")
    lines.append("")
    
    # Confidence & Quality
    lines.append("📈 CONFIDENCE ANALYSIS")
    lines.append(f"├─ Overall Confidence: {result.confidence:.1f}%")
    lines.append(f"├─ Quality Score: {result.quality_score:.1f}%")
    lines.append("│")
    lines.append("│  Component Scores (0-100%):")
    lines.append(f"├─ Indicators:      {result.indicator_score * 100:.1f}%  {'█' * int(result.indicator_score * 10)}{'░' * (10 - int(result.indicator_score * 10))}")
    lines.append(f"├─ Structure:       {result.structure_score * 100:.1f}%  {'█' * int(result.structure_score * 10)}{'░' * (10 - int(result.structure_score * 10))}")
    lines.append(f"├─ Volume:          {result.volume_score * 100:.1f}%  {'█' * int(result.volume_score * 10)}{'░' * (10 - int(result.volume_score * 10))}")
    lines.append(f"├─ MTF Alignment:   {result.mtf_score * 100:.1f}%  {'█' * int(result.mtf_score * 10)}{'░' * (10 - int(result.mtf_score * 10))}")
    lines.append(f"└─ Market Data:     {result.market_data_score * 100:.1f}%  {'█' * int(result.market_data_score * 10)}{'░' * (10 - int(result.market_data_score * 10))}")
    lines.append("")
    
    # Technical Indicators Summary
    if result.indicators:
        lines.append("📉 TECHNICAL INDICATORS")
        lines.append(f"├─ Overall: {result.indicators.overall_signal}")
        lines.append(f"├─ Bullish: {result.indicators.bullish_count} | Bearish: {result.indicators.bearish_count} | Neutral: {result.indicators.neutral_count}")
        
        # Show key indicators
        for name, ind in list(result.indicators.indicators.items())[:5]:
            lines.append(f"├─ {name.upper()}: {ind.signal} (strength: {ind.strength:.0%})")
        lines.append("")
    
    # Market Structure Summary
    if result.market_structure:
        lines.append("🏗️ MARKET STRUCTURE")
        lines.append(f"├─ Trend: {result.market_structure.trend} (strength: {result.market_structure.trend_strength:.0%})")
        valid_obs = sum(1 for ob in result.market_structure.order_blocks if ob.is_valid)
        lines.append(f"├─ Order Blocks: {valid_obs} valid")
        unfilled_fvgs = sum(1 for fvg in result.market_structure.fair_value_gaps if not fvg.filled)
        lines.append(f"├─ FVGs: {unfilled_fvgs} unfilled")
        lines.append(f"└─ Displacement: {'Yes' if result.market_structure.displacement_detected else 'No'}")
        lines.append("")
    
    # Volume Summary
    if result.volume_analysis:
        lines.append("📊 VOLUME ANALYSIS")
        lines.append(f"├─ Signal: {result.volume_analysis.signal}")
        lines.append(f"├─ POC: ${result.volume_analysis.profile.poc:,.4f}")
        lines.append(f"├─ VAH: ${result.volume_analysis.profile.vah:,.4f}")
        lines.append(f"├─ VAL: ${result.volume_analysis.profile.val:,.4f}")
        lines.append(f"├─ VWAP: ${result.volume_analysis.vwap_bands.vwap:,.4f}")
        lines.append(f"├─ Delta Pressure: {result.volume_analysis.delta.pressure}")
        lines.append(f"└─ Volume Trend: {result.volume_analysis.volume_trend}")
        lines.append("")
    
    # MTF Alignment Summary
    if result.mtf_alignment:
        lines.append("🔄 MULTI-TIMEFRAME ALIGNMENT")
        lines.append(f"├─ Bias: {result.mtf_alignment.overall_bias}")
        lines.append(f"├─ Aligned: {result.mtf_alignment.aligned_timeframes}/{result.mtf_alignment.total_timeframes} timeframes")
        lines.append(f"└─ Confluence Zones: {len(result.mtf_alignment.confluence_zones)}")
        lines.append("")
    
    # Trade Setup
    lines.append("🎯 TRADE SETUP")
    lines.append(f"├─ Direction: {result.side}")
    lines.append(f"├─ Entry: ${result.entry:,.6f}")
    lines.append(f"├─ Stop Loss: ${result.stop_loss:,.6f}")
    lines.append(f"├─ TP1: ${result.take_profit_1:,.6f} (RR: {result.risk_reward_1:.2f})")
    lines.append(f"├─ TP2: ${result.take_profit_2:,.6f} (RR: {result.risk_reward_2:.2f})")
    lines.append("")
    
    # Reasons
    if result.reasons:
        lines.append("✅ BULLISH/BEARISH FACTORS")
        for reason in result.reasons:
            lines.append(f"  • {reason}")
        lines.append("")
    
    # Warnings
    if result.warnings:
        lines.append("⚠️ WARNINGS & CONSIDERATIONS")
        for warning in result.warnings:
            lines.append(f"  • {warning}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
