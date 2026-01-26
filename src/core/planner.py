from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd

from . import binance_data as bd
from .analyzer import analyze_symbol, Analysis

# Import deep analysis module (professional deep analysis)
try:
    from .deep_analyzer import perform_deep_analysis, format_deep_analysis_report, DeepAnalysisResult
    DEEP_ANALYSIS_AVAILABLE = True
except ImportError:
    DEEP_ANALYSIS_AVAILABLE = False

# Import parallel analyzer module (professional multithreading)
try:
    from .parallel_analyzer import (
        ProfessionalParallelAnalyzer,
        ParallelAnalysisResult,
        ParallelAnalysisConfig,
        run_parallel_deep_analysis,
    )
    from .thread_manager import get_thread_manager, shutdown_thread_manager, TaskMetrics
    PARALLEL_ANALYSIS_AVAILABLE = True
except ImportError:
    PARALLEL_ANALYSIS_AVAILABLE = False


@dataclass
class ScanResult:
    symbol: str
    status: str   # OK / SETUP / NO_TRADE
    side: str
    rr2: float
    score: float
    probability: float
    reason: str
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    details: Optional[Dict[str, Any]] = None


@dataclass
class Plan:
    symbol: str
    status: str   # OK / SETUP
    side: str     # LONG / SHORT
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr1: float
    rr2: float
    score: float
    probability: float
    reason: str
    qty: float
    leverage: int
    risk_target: float
    risk_actual: float
    details: Optional[Dict[str, Any]] = None


def load_settings(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        # minimal default
        return {
            "symbols": {
                "list": ["BTCUSDT", "ETHUSDT"],
                "auto_all_usdtm": False,
                "auto_top_usdtm": True,
                "top_limit": 200,
            },
            "budget": {"default_usdt": 5.0},
            "risk": {
                "risk_pct": 0.10,
                "leverage": 3,
                "min_rr2": 3.0,
                "max_entry_distance_atr": 2.0,
                "sl_atr_mult": 1.2,
            },
            "scan": {"limit_4h": 500, "limit_1h": 500, "limit_15m": 500, "sleep_ms": 0},
            "strategy": {
                "impulse_lookback": 240,
                "htf_range_lookback": 200,
                "htf_bias_timeframes": ["4h"],
                "htf_min_alignment": 1.0,
                "impulse_timeframe": "4h",
                "confirm_timeframe": "4h",
                "measurement_timeframe": "15m",
                "micro_confirm_timeframe": "15m",
                "min_confluence": 1,
                "require_confluence": True,
                "allow_setup_if_no_confirm": False,
                "best_requires_ok": True,
                "force_best_plan": True,
                "allow_weak_confluence": False,
                "allow_pre_zone": False,
                "max_pre_zone_atr": 1.5,
                "zone_tolerance_atr": 0.25,
                "zone_tolerance_pct": 0.002,
                "sl_atr_mult": 0.0,
                "fib_zone_mode": "golden",
                "respect_body_atr_ratio": 0.6,
                "allow_measurement_setup": True,
            },
            "scoring": {
                "w_rr2": 10.0,
                "w_trend": 5.0,
                "w_trend_strength": 3.0,
                "w_confluence": 3.0,
                "w_confluence_count": 1.0,
                "w_confirmation": 2.0,
                "w_micro_confirmation": 1.5,
                "w_alignment": 2.0,
                "w_entry_distance": 1.0,
                "w_zone": 2.0,
                "w_zone_balance": 1.0,
                "w_status": 2.0,
                "w_momentum": 2.0,
                "w_volume": 1.5,
                "w_volatility": 1.5,
                "w_liquidity": 1.0,
                "rr2_target": 6.0,
                "confluence_target": 3,
                "min_setup_quality_pct": 55.0,
                "rsi_period": 14,
                "volume_lookback": 30,
                "volume_ratio_min": 0.6,
                "volume_ratio_max": 2.0,
                "volatility_lookback": 120,
                "volatility_target_pctile": 0.6,
            },
        }
    return json.loads(p.read_text(encoding="utf-8"))


def save_settings(path: str, data: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float((x // step) * step)


def _calc_qty(symbol: str, entry: float, sl: float, budget_usdt: float, risk_pct: float, leverage: int) -> Tuple[float, float, float]:
    filters = bd.get_symbol_filters(symbol)
    step = float(filters.get("stepSize", 0.0)) or 0.0
    min_qty = float(filters.get("minQty", 0.0)) or 0.0

    risk_target = budget_usdt * risk_pct
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return 0.0, risk_target, 0.0

    # risk-based qty
    qty_risk = risk_target / risk_per_unit

    # notional cap
    notional_cap = budget_usdt * float(leverage)
    qty_notional = notional_cap / entry if entry > 0 else 0.0

    qty = min(qty_risk, qty_notional)
    qty = _round_step(qty, step)
    if qty < min_qty:
        qty = 0.0

    risk_actual = qty * risk_per_unit
    return qty, risk_target, risk_actual


def _fit_probability(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 50.0
    pct = (score / max_score) * 100.0
    return float(max(5.0, min(95.0, pct)))


def _fit_probability_distribution(score: float, mean_score: float, std_score: float) -> float:
    if std_score <= 0:
        return _fit_probability(score, max_score=mean_score if mean_score > 0 else 1.0)
    z = (score - mean_score) / std_score
    sigmoid = 1.0 / (1.0 + math.exp(-z))
    return float(max(5.0, min(95.0, sigmoid * 100.0)))


def run_scan_and_build_best_plan(
    binance_data,
    analyzer,
    settings: Dict[str, Any],
    symbols: List[str],
    budget_usdt: float,
    risk_pct: float,
    leverage: int,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> Tuple[List[ScanResult], Optional[Plan]]:

    scan_cfg = settings.get("scan", {})
    sleep_ms = int(scan_cfg.get("sleep_ms", 0))

    def fetch(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return binance_data.get_ohlcv(symbol, interval, limit=limit, sleep_ms=sleep_ms)

    results: List[ScanResult] = []
    best_ok: Optional[Plan] = None
    best_setup: Optional[Plan] = None

    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        sym = sym.upper().strip()
        if on_progress:
            on_progress(idx, total, sym)

        # validate symbol (professional filter)
        if hasattr(binance_data, "is_valid_usdtm_perp") and not binance_data.is_valid_usdtm_perp(sym):
            results.append(ScanResult(sym, "NO_TRADE", "-", 0.0, 0.0, 0.0, "Symbol USDT-M PERP deyil / TRADING deyil"))
            continue

        try:
            a: Analysis = analyzer.analyze_symbol(sym, fetch_ohlcv=fetch, settings=settings, on_stage=on_stage)

            r = ScanResult(
                symbol=sym,
                status=a.status,
                side=a.side,
                rr2=float(a.rr2),
                score=float(a.score),
                probability=0.0,
                reason=a.reason,
                entry=float(a.entry),
                sl=float(a.sl),
                tp1=float(a.tp1),
                tp2=float(a.tp2),
                details=a.details,
            )
            results.append(r)

            if a.status in ("OK", "SETUP"):
                qty, risk_target, risk_actual = _calc_qty(sym, a.entry, a.sl, budget_usdt, risk_pct, leverage)
                if qty <= 0:
                    # can't size => treat as NO_TRADE (exchange minQty/step)
                    r.status = "NO_TRADE"
                    r.reason = "MinQty/step səbəbilə qty=0 (budget çox kiçik ola bilər)"
                    continue

                plan = Plan(
                    symbol=sym,
                    status=a.status,
                    side=a.side,
                    entry=a.entry,
                    sl=a.sl,
                    tp1=a.tp1,
                    tp2=a.tp2,
                    rr1=a.rr1,
                    rr2=a.rr2,
                    score=a.score,
                    probability=0.0,
                    reason=a.reason,
                    qty=qty,
                    leverage=int(leverage),
                    risk_target=risk_target,
                    risk_actual=risk_actual,
                    details=a.details,
                )

                if a.status == "OK":
                    if (best_ok is None) or (plan.score > best_ok.score):
                        best_ok = plan
                else:
                    if (best_setup is None) or (plan.score > best_setup.score):
                        best_setup = plan

        except Exception as e:
            results.append(ScanResult(sym, "NO_TRADE", "-", 0.0, 0.0, 0.0, f"ERROR: {e}"))

    scored = [r.score for r in results if r.status in ("OK", "SETUP")]
    if not scored:
        scored = [r.score for r in results]
    max_score = max(scored) if scored else 0.0
    min_score = min(scored) if scored else 0.0
    mean_score = (sum(scored) / len(scored)) if scored else 0.0
    std_score = math.sqrt(sum((s - mean_score) ** 2 for s in scored) / len(scored)) if scored else 0.0

    def _range_probability(score: float) -> float:
        if max_score <= min_score:
            return 50.0
        pct = (score - min_score) / (max_score - min_score)
        return float(max(5.0, min(95.0, pct * 100.0)))

    def _blend_probability(score: float) -> float:
        dist_prob = _fit_probability_distribution(score, mean_score, std_score)
        range_prob = _range_probability(score)
        blended = (dist_prob * 0.6) + (range_prob * 0.4)
        return float(max(5.0, min(95.0, blended)))

    for r in results:
        # Use blended probability for all coins to show relative strength
        r.probability = _blend_probability(r.score)
    if best_ok:
        best_ok.probability = _blend_probability(best_ok.score)
    if best_setup:
        best_setup.probability = _blend_probability(best_setup.score)

    prefer_ok = bool(settings.get("strategy", {}).get("best_requires_ok", False))
    if prefer_ok and best_ok:
        return results, best_ok
    return results, (best_ok or best_setup)


def save_scan_results(
    results: List[ScanResult],
    best: Optional[Plan],
    settings: Dict[str, Any],
    output_dir: str = "outputs/scans",
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": timestamp,
        "settings": {
            "symbols": settings.get("symbols", {}),
            "risk": settings.get("risk", {}),
            "scoring": settings.get("scoring", {}),
            "strategy": settings.get("strategy", {}),
            "timeframes": settings.get("timeframes", {}),
            "scan": settings.get("scan", {}),
        },
        "best": asdict(best) if best else None,
        "results": [asdict(r) for r in results],
    }

    json_path = out_dir / f"scan_{timestamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame([{
        "symbol": r.symbol,
        "status": r.status,
        "side": r.side,
        "rr2": r.rr2,
        "score": r.score,
        "probability": r.probability,
        "entry": r.entry,
        "sl": r.sl,
        "tp1": r.tp1,
        "tp2": r.tp2,
        "reason": r.reason,
    } for r in results])
    csv_path = out_dir / f"scan_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    return json_path


def format_report(
    results: List[ScanResult],
    best: Optional[Plan],
    settings: Dict[str, Any],
    snapshot_path: Optional[Path] = None,
) -> str:
    ok = [r for r in results if r.status == "OK"]
    setups = [r for r in results if r.status == "SETUP"]
    no = [r for r in results if r.status == "NO_TRADE"]

    # reason stats (top)
    reason_count: Dict[str, int] = {}
    for r in no:
        reason_count[r.reason] = reason_count.get(r.reason, 0) + 1
    top_reasons = sorted(reason_count.items(), key=lambda x: x[1], reverse=True)[:8]

    lines: List[str] = []
    lines.append("Scan başladı.\n")
    lines.append("=== Scan Summary ===")
    lines.append(f"OK: {len(ok)} | SETUP: {len(setups)} | NO_TRADE: {len(no)} | TOTAL: {len(results)}\n")
    min_rr2 = float(settings.get("risk", {}).get("min_rr2", 3.0))
    lines.append(f"Risk/Reward qaydası: minimum RR2 = {min_rr2:.2f} (3x1)")
    lines.append("")

    if top_reasons:
        lines.append("=== NO_TRADE Top Səbəblər ===")
        for k, v in top_reasons:
            lines.append(f"- {v}x: {k}")
        lines.append("")

    # show coin list sorted by probability
    sorted_results = sorted(results, key=lambda r: r.probability, reverse=True)
    lines.append("=== COIN LIST (Ehtimala görə sıralı) ===")
    for r in sorted_results:
        lines.append(
            f"{r.symbol}: {r.status} | {r.side} | fit={r.probability:.1f}% | RR2={r.rr2:.2f} | score={r.score:.2f}"
        )
    lines.append("")

    lines.append("=== BEST PLAN ===")
    if not best:
        lines.append("NO TRADE\n")
        if snapshot_path:
            lines.append(f"Snapshot: {snapshot_path}")
        return "\n".join(lines)

    lines.append(f"{best.symbol} | {best.side} | {best.status}")
    lines.append(f"Entry={best.entry:.6f} SL={best.sl:.6f} TP1={best.tp1:.6f} TP2={best.tp2:.6f}")
    lines.append(f"RR1={best.rr1:.2f} RR2={best.rr2:.2f} | fit={best.probability:.1f}% | score={best.score:.2f}")
    lines.append(f"Səbəb: {best.reason}\n")

    if best.details:
        lines.append("=== ANALİZ DETALLARI ===")
        for key, value in best.details.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    # Manual form guidance
    exp_days = int(settings.get("plan", {}).get("expiry_days", 7))
    lines.append("=== BINANCE FUTURES FORM (manual doldurma) ===")
    lines.append(f"Market: {best.symbol}")
    lines.append("Margin: Isolated")
    lines.append(f"Leverage: {best.leverage}x\n")
    lines.append("Tab: Limit")
    lines.append(f"Price (Entry): {best.entry:.6f}")
    lines.append(f"Size (Qty): {best.qty:.6f}")
    lines.append("TP/SL: ON")
    lines.append(f"  Take Profit: {best.tp2:.6f}   Trigger: Mark")
    lines.append(f"  Stop Loss:   {best.sl:.6f}   Trigger: Mark")
    lines.append("Reduce-Only: OFF (entry açırsan)")
    lines.append("TIF: GTC")
    lines.append("Action: Buy/Long" if best.side == "LONG" else "Action: Sell/Short")
    lines.append(f"\nExpiry: {exp_days} gün (trigger olmazsa cancel)")
    lines.append(f"Risk target: {best.risk_target:.4f} USDT | Risk actual: {best.risk_actual:.4f} USDT")

    if best.status == "SETUP":
        lines.append("\n[WATCH] Bu SETUP-dur. Qaydaya görə kor-koranə girmə. Price zone-a gələndə 5m sweep+close təsdiqi gözlə.")

    if snapshot_path:
        lines.append(f"\nSnapshot: {snapshot_path}")

    return "\n".join(lines)


# =============================================================================
# Deep Analysis Functions (Professional-Grade)
# =============================================================================

def run_deep_analysis_scan(
    symbols: List[str],
    settings: Dict[str, Any],
    budget_usdt: float,
    risk_pct: float,
    leverage: int,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Run professional deep analysis scan on symbols.
    
    This leverages all analysis modules for comprehensive results:
    - Technical indicators (MACD, Bollinger, Stoch RSI, ADX, OBV, Ichimoku, etc.)
    - Market structure (order blocks, FVG, BOS/ChoCH)
    - Volume analysis (profile, delta, VWAP)
    - Multi-timeframe alignment
    - Market data (funding, OI, order book)
    
    Returns:
        Tuple of (results list, best plan dict)
    """
    if not DEEP_ANALYSIS_AVAILABLE:
        return [], None
    
    scan_cfg = settings.get("scan", {})
    sleep_ms = int(scan_cfg.get("sleep_ms", 0))
    deep_enabled = settings.get("deep_analysis", {}).get("enabled", True)
    
    if not deep_enabled:
        return [], None
    
    def fetch(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return bd.get_ohlcv(symbol, interval, limit=limit, sleep_ms=sleep_ms)
    
    results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_score: float = 0.0
    
    total = len(symbols)
    
    for idx, sym in enumerate(symbols, start=1):
        sym = sym.upper().strip()
        if on_progress:
            on_progress(idx, total, sym)
        
        # Validate symbol
        if not bd.is_valid_usdtm_perp(sym):
            results.append({
                "symbol": sym,
                "status": "NO_TRADE",
                "side": "-",
                "confidence": 0.0,
                "quality_score": 0.0,
                "reason": "Symbol USDT-M PERP deyil / TRADING deyil",
            })
            continue
        
        try:
            # Get comprehensive market data
            if on_stage:
                on_stage(f"Fetching market data for {sym}")
            
            market_data = bd.get_comprehensive_market_data(sym)
            
            # Run deep analysis
            deep_result: DeepAnalysisResult = perform_deep_analysis(
                symbol=sym,
                fetch_ohlcv=fetch,
                settings=settings,
                market_data=market_data,
                on_stage=on_stage,
            )
            
            # Calculate position size
            qty, risk_target, risk_actual = _calc_qty(
                sym, deep_result.entry, deep_result.stop_loss,
                budget_usdt, risk_pct, leverage
            )
            
            # Determine status
            if deep_result.confidence >= 70 and deep_result.risk_reward_2 >= 2.5 and qty > 0:
                status = "OK"
            elif deep_result.confidence >= 50 and deep_result.risk_reward_2 >= 2.0 and qty > 0:
                status = "SETUP"
            else:
                status = "NO_TRADE"
            
            result_dict = {
                "symbol": sym,
                "status": status,
                "side": deep_result.side,
                "signal": deep_result.signal,
                "confidence": deep_result.confidence,
                "quality_score": deep_result.quality_score,
                "entry": deep_result.entry,
                "sl": deep_result.stop_loss,
                "tp1": deep_result.take_profit_1,
                "tp2": deep_result.take_profit_2,
                "rr1": deep_result.risk_reward_1,
                "rr2": deep_result.risk_reward_2,
                "qty": qty,
                "leverage": leverage,
                "risk_target": risk_target,
                "risk_actual": risk_actual,
                "reasons": deep_result.reasons,
                "warnings": deep_result.warnings,
                "indicator_score": deep_result.indicator_score,
                "structure_score": deep_result.structure_score,
                "volume_score": deep_result.volume_score,
                "mtf_score": deep_result.mtf_score,
                "market_data_score": deep_result.market_data_score,
                "details": deep_result.details,
            }
            
            results.append(result_dict)
            
            # Track best result
            if status in ("OK", "SETUP") and deep_result.quality_score > best_score:
                best_score = deep_result.quality_score
                best_result = result_dict
                
        except Exception as e:
            results.append({
                "symbol": sym,
                "status": "NO_TRADE",
                "side": "-",
                "confidence": 0.0,
                "quality_score": 0.0,
                "reason": f"ERROR: {e}",
            })
    
    return results, best_result


def format_deep_analysis_scan_report(
    results: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    settings: Dict[str, Any],
) -> str:
    """
    Format deep analysis scan results as a professional report.
    """
    lines: List[str] = []
    
    # Header
    lines.append("=" * 70)
    lines.append("        PROFESSIONAL DEEP ANALYSIS SCAN REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary
    ok = [r for r in results if r.get("status") == "OK"]
    setups = [r for r in results if r.get("status") == "SETUP"]
    no = [r for r in results if r.get("status") == "NO_TRADE"]
    
    lines.append("📊 SCAN SUMMARY")
    lines.append(f"├─ OK Signals: {len(ok)}")
    lines.append(f"├─ SETUP: {len(setups)}")
    lines.append(f"├─ NO_TRADE: {len(no)}")
    lines.append(f"└─ TOTAL: {len(results)}")
    lines.append("")
    
    # Top opportunities (sorted by confidence)
    sorted_results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    top_results = [r for r in sorted_results if r.get("status") in ("OK", "SETUP")][:10]
    
    if top_results:
        lines.append("🎯 TOP OPPORTUNITIES (by confidence)")
        lines.append("-" * 70)
        lines.append(f"{'Symbol':<12} {'Side':<6} {'Status':<8} {'Conf':<8} {'RR2':<6} {'Signal':<12}")
        lines.append("-" * 70)
        
        for r in top_results:
            lines.append(
                f"{r.get('symbol', '-'):<12} "
                f"{r.get('side', '-'):<6} "
                f"{r.get('status', '-'):<8} "
                f"{r.get('confidence', 0):.1f}%{'':<3} "
                f"{r.get('rr2', 0):.2f}{'':<2} "
                f"{r.get('signal', '-'):<12}"
            )
        lines.append("")
    
    # Best plan details
    if best:
        lines.append("=" * 70)
        lines.append("                    BEST TRADE PLAN")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"📈 {best.get('symbol', '-')} | {best.get('side', '-')} | {best.get('signal', '-')}")
        lines.append("")
        
        lines.append("📊 CONFIDENCE BREAKDOWN")
        lines.append(f"├─ Overall Confidence: {best.get('confidence', 0):.1f}%")
        lines.append(f"├─ Quality Score: {best.get('quality_score', 0):.1f}%")
        lines.append(f"├─ Indicators: {best.get('indicator_score', 0) * 100:.1f}%")
        lines.append(f"├─ Structure: {best.get('structure_score', 0) * 100:.1f}%")
        lines.append(f"├─ Volume: {best.get('volume_score', 0) * 100:.1f}%")
        lines.append(f"├─ MTF Alignment: {best.get('mtf_score', 0) * 100:.1f}%")
        lines.append(f"└─ Market Data: {best.get('market_data_score', 0) * 100:.1f}%")
        lines.append("")
        
        lines.append("🎯 TRADE LEVELS")
        lines.append(f"├─ Entry: ${best.get('entry', 0):,.6f}")
        lines.append(f"├─ Stop Loss: ${best.get('sl', 0):,.6f}")
        lines.append(f"├─ TP1: ${best.get('tp1', 0):,.6f} (RR: {best.get('rr1', 0):.2f})")
        lines.append(f"├─ TP2: ${best.get('tp2', 0):,.6f} (RR: {best.get('rr2', 0):.2f})")
        lines.append(f"├─ Qty: {best.get('qty', 0):.6f}")
        lines.append(f"└─ Leverage: {best.get('leverage', 1)}x")
        lines.append("")
        
        if best.get("reasons"):
            lines.append("✅ ANALYSIS REASONS")
            for reason in best["reasons"][:8]:  # Limit to 8
                lines.append(f"  • {reason}")
            lines.append("")
        
        if best.get("warnings"):
            lines.append("⚠️ WARNINGS")
            for warning in best["warnings"][:5]:  # Limit to 5
                lines.append(f"  • {warning}")
            lines.append("")
        
        # Trading form
        lines.append("=" * 70)
        lines.append("            BINANCE FUTURES ORDER FORM")
        lines.append("=" * 70)
        lines.append(f"Market: {best.get('symbol', '-')}")
        lines.append("Margin: Isolated")
        lines.append(f"Leverage: {best.get('leverage', 1)}x")
        lines.append("")
        lines.append("Tab: Limit")
        lines.append(f"Price (Entry): {best.get('entry', 0):.6f}")
        lines.append(f"Size (Qty): {best.get('qty', 0):.6f}")
        lines.append("TP/SL: ON")
        lines.append(f"  Take Profit: {best.get('tp2', 0):.6f}   Trigger: Mark")
        lines.append(f"  Stop Loss: {best.get('sl', 0):.6f}   Trigger: Mark")
        lines.append("Reduce-Only: OFF (entry açırsan)")
        lines.append("TIF: GTC")
        lines.append(f"Action: {'Buy/Long' if best.get('side') == 'LONG' else 'Sell/Short'}")
        lines.append("")
        lines.append(f"Risk target: {best.get('risk_target', 0):.4f} USDT")
        lines.append(f"Risk actual: {best.get('risk_actual', 0):.4f} USDT")
        
        if best.get("status") == "SETUP":
            lines.append("")
            lines.append("[WATCH] Bu SETUP-dur - price zone-a gələndə confirmation gözlə!")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# Professional Parallel Scan Functions (Multithreaded)
# =============================================================================

def run_parallel_scan(
    symbols: List[str],
    settings: Dict[str, Any],
    budget_usdt: float,
    risk_pct: float,
    leverage: int,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_result: Optional[Callable[[str, Any], None]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Run professional parallel scan on multiple symbols using multithreading.
    
    This function leverages the thread pool for maximum performance,
    analyzing multiple symbols concurrently with parallel component execution.
    
    Features:
    - Parallel symbol processing
    - Concurrent indicator, structure, and volume analysis
    - Thread-safe result aggregation
    - Performance metrics reporting
    
    Args:
        symbols: List of symbols to scan
        settings: Analysis and threading configuration
        budget_usdt: Trading budget
        risk_pct: Risk percentage per trade
        leverage: Leverage multiplier
        on_progress: Progress callback (completed, total, current_symbol)
        on_result: Called when each result is ready
    
    Returns:
        Tuple of (results list, best result dict, performance metrics)
    """
    if not PARALLEL_ANALYSIS_AVAILABLE:
        # Fallback to sequential if parallel not available
        return run_deep_analysis_scan(
            symbols, settings, budget_usdt, risk_pct, leverage,
            on_progress=on_progress,
        ) + ({},)
    
    threading_config = settings.get("threading", {})
    
    if not threading_config.get("enabled", True):
        # Threading disabled, use sequential
        return run_deep_analysis_scan(
            symbols, settings, budget_usdt, risk_pct, leverage,
            on_progress=on_progress,
        ) + ({},)
    
    scan_cfg = settings.get("scan", {})
    sleep_ms = int(scan_cfg.get("sleep_ms", 0))
    
    def fetch(symbol: str, interval: str, limit: int) -> pd.DataFrame:
        return bd.get_ohlcv(symbol, interval, limit=limit, sleep_ms=sleep_ms)
    
    # Create parallel analyzer with config
    parallel_config = ParallelAnalysisConfig(
        max_concurrent_symbols=threading_config.get("max_concurrent_symbols", 10),
        max_concurrent_timeframes=threading_config.get("max_concurrent_timeframes", 5),
        enable_caching=threading_config.get("enable_caching", True),
        cache_ttl_seconds=threading_config.get("cache_ttl_seconds", 30.0),
        data_fetch_retries=threading_config.get("data_fetch_retries", 2),
    )
    
    analyzer = ProfessionalParallelAnalyzer(
        fetch_ohlcv=fetch,
        settings=settings,
        config=parallel_config,
    )
    
    # Run parallel analysis
    parallel_results = analyzer.analyze_symbols_parallel(
        symbols=symbols,
        on_progress=on_progress,
        on_result=on_result,
    )
    
    # Convert to result dictionaries and find best
    results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_score: float = 0.0
    
    for pr in parallel_results:
        # Validate symbol for trading
        if not bd.is_valid_usdtm_perp(pr.symbol):
            results.append({
                "symbol": pr.symbol,
                "status": "NO_TRADE",
                "side": "-",
                "confidence": 0.0,
                "quality_score": 0.0,
                "reason": "Symbol USDT-M PERP deyil / TRADING deyil",
            })
            continue
        
        if not pr.success:
            results.append({
                "symbol": pr.symbol,
                "status": "NO_TRADE",
                "side": "-",
                "confidence": 0.0,
                "quality_score": 0.0,
                "reason": "; ".join(pr.errors) if pr.errors else "Xəta baş verdi",
                "execution_time_ms": pr.execution_time_ms,
            })
            continue
        
        # Calculate entry/SL/TP from analysis
        entry = 0.0
        sl = 0.0
        tp1 = 0.0
        tp2 = 0.0
        rr1 = 0.0
        rr2 = 0.0
        
        # Use market structure or basic ATR-based levels
        if pr.market_structure and pr.indicators:
            current_price = 0.0
            if pr.volume_analysis:
                current_price = pr.volume_analysis.vwap_bands.vwap
            
            # Basic level calculation (can be enhanced)
            side = "LONG" if pr.overall_signal in ("STRONG_BUY", "BUY") else "SHORT"
            
            if side == "LONG":
                entry = current_price if current_price > 0 else 0.0
                # These would normally come from the deep analyzer
            else:
                entry = current_price if current_price > 0 else 0.0
        
        # Calculate position size
        if entry > 0 and sl > 0:
            qty, risk_target, risk_actual = _calc_qty(
                pr.symbol, entry, sl, budget_usdt, risk_pct, leverage
            )
        else:
            qty = 0.0
            risk_target = 0.0
            risk_actual = 0.0
        
        # Determine status
        if pr.confidence >= 70 and rr2 >= 2.5 and qty > 0:
            status = "OK"
        elif pr.confidence >= 50 and rr2 >= 2.0:
            status = "SETUP"
        else:
            status = "NO_TRADE"
        
        result_dict = {
            "symbol": pr.symbol,
            "status": status,
            "side": "LONG" if pr.overall_signal in ("STRONG_BUY", "BUY") else "SHORT" if pr.overall_signal in ("STRONG_SELL", "SELL") else "-",
            "signal": pr.overall_signal,
            "confidence": pr.confidence,
            "quality_score": pr.confidence,  # Use confidence as quality
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr1": rr1,
            "rr2": rr2,
            "qty": qty,
            "leverage": leverage,
            "risk_target": risk_target,
            "risk_actual": risk_actual,
            "execution_time_ms": pr.execution_time_ms,
            "indicator_signal": pr.indicators.overall_signal if pr.indicators else "UNKNOWN",
            "structure_trend": pr.market_structure.trend if pr.market_structure else "UNKNOWN",
            "volume_signal": pr.volume_analysis.signal if pr.volume_analysis else "UNKNOWN",
            "mtf_count": len(pr.timeframe_analyses),
            "errors": pr.errors,
        }
        
        results.append(result_dict)
        
        # Track best
        if status in ("OK", "SETUP") and pr.confidence > best_score:
            best_score = pr.confidence
            best_result = result_dict
    
    # Get performance metrics
    metrics = analyzer.get_performance_metrics()
    
    return results, best_result, metrics


def format_parallel_scan_report(
    results: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    settings: Dict[str, Any],
) -> str:
    """
    Format parallel scan results as a professional report with threading metrics.
    """
    lines: List[str] = []
    
    # Header
    lines.append("=" * 70)
    lines.append("      PROFESSIONAL PARALLEL ANALYSIS REPORT (MULTITHREADED)")
    lines.append("=" * 70)
    lines.append("")
    
    # Performance metrics
    lines.append("⚡ THREADING PERFORMANCE")
    lines.append(f"├─ Total Tasks: {metrics.get('total_tasks', 0)}")
    lines.append(f"├─ Completed: {metrics.get('completed_tasks', 0)}")
    lines.append(f"├─ Failed: {metrics.get('failed_tasks', 0)}")
    lines.append(f"├─ Avg Execution: {metrics.get('avg_execution_time_ms', 0):.1f}ms")
    lines.append(f"├─ Peak Threads: {metrics.get('peak_threads', 0)}")
    lines.append(f"└─ Active Threads: {metrics.get('active_threads', 0)}")
    lines.append("")
    
    # Summary
    ok = [r for r in results if r.get("status") == "OK"]
    setups = [r for r in results if r.get("status") == "SETUP"]
    no = [r for r in results if r.get("status") == "NO_TRADE"]
    
    lines.append("📊 SCAN SUMMARY")
    lines.append(f"├─ OK Signals: {len(ok)}")
    lines.append(f"├─ SETUP: {len(setups)}")
    lines.append(f"├─ NO_TRADE: {len(no)}")
    lines.append(f"└─ TOTAL: {len(results)}")
    lines.append("")
    
    # Top opportunities
    sorted_results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    top_results = [r for r in sorted_results if r.get("status") in ("OK", "SETUP")][:10]
    
    if top_results:
        lines.append("🎯 TOP OPPORTUNITIES (by confidence)")
        lines.append("-" * 70)
        lines.append(f"{'Symbol':<12} {'Side':<6} {'Status':<8} {'Conf':<8} {'Time':<10} {'Signal':<12}")
        lines.append("-" * 70)
        
        for r in top_results:
            exec_time = r.get("execution_time_ms", 0)
            lines.append(
                f"{r.get('symbol', '-'):<12} "
                f"{r.get('side', '-'):<6} "
                f"{r.get('status', '-'):<8} "
                f"{r.get('confidence', 0):.1f}%{'':<3} "
                f"{exec_time:.0f}ms{'':<4} "
                f"{r.get('signal', '-'):<12}"
            )
        lines.append("")
    
    # Best plan
    if best:
        lines.append("=" * 70)
        lines.append("                    BEST TRADE PLAN")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"📈 {best.get('symbol', '-')} | {best.get('side', '-')} | {best.get('signal', '-')}")
        lines.append(f"   Confidence: {best.get('confidence', 0):.1f}%")
        lines.append(f"   Execution Time: {best.get('execution_time_ms', 0):.0f}ms")
        lines.append("")
        lines.append(f"├─ Indicator Signal: {best.get('indicator_signal', '-')}")
        lines.append(f"├─ Structure Trend: {best.get('structure_trend', '-')}")
        lines.append(f"├─ Volume Signal: {best.get('volume_signal', '-')}")
        lines.append(f"└─ MTF Timeframes: {best.get('mtf_count', 0)}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)
