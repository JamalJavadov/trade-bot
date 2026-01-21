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
                "htf_bias_timeframes": ["1d", "4h", "1h"],
                "htf_min_alignment": 0.6,
                "impulse_timeframe": "1h",
                "confirm_timeframe": "15m",
                "measurement_timeframe": "15m",
                "micro_confirm_timeframe": "5m",
                "min_confluence": 2,
                "require_confluence": True,
                "allow_setup_if_no_confirm": False,
                "best_requires_ok": True,
                "allow_weak_confluence": False,
                "allow_pre_zone": False,
                "max_pre_zone_atr": 1.5,
                "zone_tolerance_atr": 0.25,
                "zone_tolerance_pct": 0.002,
                "sl_atr_mult": 1.2,
                "allow_measurement_setup": False,
            },
            "scoring": {
                "w_rr2": 10.0,
                "w_trend": 5.0,
                "w_confluence": 3.0,
                "w_confluence_count": 1.0,
                "w_confirmation": 2.0,
                "w_micro_confirmation": 1.5,
                "w_alignment": 2.0,
                "w_entry_distance": 1.0,
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

    scored = [r.score for r in results]
    max_score = max(scored) if scored else 0.0
    mean_score = (sum(scored) / len(scored)) if scored else 0.0
    std_score = math.sqrt(sum((s - mean_score) ** 2 for s in scored) / len(scored)) if scored else 0.0
    for r in results:
        r.probability = _fit_probability_distribution(r.score, mean_score, std_score)
    if best_ok:
        best_ok.probability = _fit_probability_distribution(best_ok.score, mean_score, std_score)
    if best_setup:
        best_setup.probability = _fit_probability_distribution(best_setup.score, mean_score, std_score)

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
