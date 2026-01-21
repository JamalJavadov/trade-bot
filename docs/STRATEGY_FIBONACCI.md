# Fibonacci Strategy Blueprint (Codex Reference)

> **Purpose:** This document translates the provided Fibonacci trading videos into a precise, repeatable strategy that the bot can follow and that a human can audit. It is a professional, rule-based blueprint with clear entries, confirmations, exits, and risk controls.

## 1) Core Philosophy
- Fibonacci levels are **not magic**; they are *measurement tools* for identifying **structured pullbacks** within a trend.
- A valid setup needs **structure + confluence + confirmation + risk control**.
- Levels are **zones**, not single price lines. We treat them as *areas of reaction*.

## 2) Chart Setup & Tools
**Timeframes**
1. **HTF Bias Stack (default: 1D + 4H)** – trend and premium/discount alignment across multiple higher timeframes.
2. **Impulse TF (default: 1H)** – main impulse leg selection + golden zone setup.
3. **Confirm TF (default: 15M)** – confirmation signals (engulfing, star, rejection).
4. **Measurement TF (default: 15M)** – BOS/FVG measurement setups (can differ from confirm TF).

**Indicators / Tools**
- Fibonacci Retracement (standard levels).
- 50 EMA (trend + dynamic S/R).
- Anchored VWAP (from impulse start).
- Fractals (5-period) to locate swing highs/lows.
- Optional: Fib Time Zones (timing confluence, optional).

**HTF Alignment Rule**
- The bot takes the **majority bias** across HTF timeframes. If the alignment ratio is below the configured threshold, it stands aside.

## 3) Fibonacci Settings
Retracement zones (choose by regime or let the bot auto-select):
- **Golden Zone:** 0.50–0.618 (balanced trend pullback)
- **Strong Trend Zone:** 0.382–0.50 (shallow pullback in aggressive trends)
- **Beginner Zone:** 0.382–0.618 (broadest reaction band)
- Optional: 0.786 (deep retrace filter only)

Extensions for targets:
- **1.0 measured move** (swing high/low retest)
- **1.618 extension** (primary TP2 target)
- Optional: 1.272 extension (aggressive but closer TP2)

Measurement entry zone:
- **0.71 – 0.75** (precision retrace zone)

## 4) Swing Selection (Precision Anchor)
1. Identify the **impulse leg** in the trend direction.
2. Anchor Fibonacci **from wick to wick**:
   - **Uptrend:** swing low wick → swing high wick.
   - **Downtrend:** swing high wick → swing low wick.
3. If price ignores the first anchor (no reactions), adjust to the **next significant swing**.

## 5) Fibonacci Retracement Strategy (Primary)
**Idea:** In a trend, price often retraces into a Fibonacci **reaction zone** before continuation.

### 5.1 Entry Zone (Adaptive)
- **Golden Zone (default classic):** 0.50–0.618.
- **Strong Trend Zone:** 0.382–0.50.
- **Beginner Zone:** 0.382–0.618.

### 5.2 Confluence Filters (must have at least one)
- **EMA50**: price retraces into golden zone + EMA50 agrees with trend slope.
- **Anchored VWAP**: anchored from impulse start, overlapping the zone.
- **Swap Zone**: prior support turned resistance (or vice versa) aligns with zone.

### 5.3 Confirmation (required for “OK”)
Choose one of:
- **Bullish/Bearish Engulfing** inside the zone.
- **Morning/Evening Star** inside the zone.
- **Rejection Wick** (wick through zone, close back inside).

Plus: **Golden Respect Rule**
- Require 2–3 closes that **respect** the zone (no clean break through it).

### 5.4 Stop & Targets
- **Stop Loss:** beyond the swing point (ATR buffered).
  - Long: below swing low.
  - Short: above swing high.
- **TP1:** return to swing high/low.
- **TP2:** 1.618 extension (optional 1.272 for conservative exits).

### 5.5 Setup vs. OK
- **OK:** in zone + confirmation + confluence.
- **SETUP:** in zone but confirmation pending, or very near zone (watchlist).

## 6) Measurement Strategy (71% Entry)
**Idea:** Use Fibonacci as a *measurement tool* after liquidity sweep + BOS.

### 6.1 Preconditions
1. **Liquidity Sweep** (stop-hunt).
2. **Break of Structure** (BOS) in intended direction.
3. Identify the impulse leg from sweep → BOS.

### 6.2 Entry Zone
- **Short:** retrace into **0.71–0.75** zone.
- **Long:** retrace into **0.71–0.75** zone.

### 6.3 Confluence Boost
- **Fair Value Gap (FVG) overlap** with the 0.71–0.75 zone.

### 6.4 Stop & Targets
- **Stop:** beyond the impulse high/low.
- **TP:** opposite end of the impulse (range low/high).

## 7) Risk Management Rules
- Risk per trade: **1–2%** (configurable by bot).
- Minimum **RR2** threshold (default 2.0).
- Reject entries if **distance to entry > 2 ATR** (avoids chasing).

## 8) How the Bot Scores “Best Coin”
The bot does not “predict guaranteed success.” It ranks trades by **objective scoring**:
- **RR2** (reward/risk).
- **Trend alignment** (EMA bias).
- **Confluence count** (EMA50, AVWAP, swap zone, FVG overlap).
- **Confirmation quality** (engulfing / star / rejection).

The coin with the **highest score** is presented as the **BEST PLAN**.

**Fit Probability (Uyğunluq faizi)**
- Each coin gets a **fit percentage** based on its score relative to the best score in the scan.
- This is **not a guaranteed win rate**; it reflects *how well the coin matches the strategy rules*.

## 9) Operational Checklist (Human or Bot)
1. Identify 4H trend bias (EMA or premium/discount).
2. Find impulse leg on 1H and draw Fibonacci.
3. Check if price is in the selected Fib zone (Golden/Strong/Beginner).
4. Require at least one confluence.
5. Confirm with a candlestick reversal or rejection wick.
6. Compute SL/TPs and RR.
7. Use **pending limit orders** at the planned entry price (no market chase).
8. If all rules pass → **OK**. Else → **SETUP** (watch).

## 10) Parameters (Bot Settings Reference)
Key settings in `settings.json`:
- `strategy.min_confluence`
- `strategy.zone_tolerance_atr`
- `strategy.allow_pre_zone`
- `strategy.htf_bias_timeframes`
- `strategy.htf_min_alignment`
- `strategy.impulse_timeframe`
- `strategy.confirm_timeframe`
- `strategy.measurement_timeframe`
- `risk.min_rr2`
- `risk.max_entry_distance_atr`
- `scoring.w_confluence_count`
- `scoring.w_confirmation`

---

### Final Notes
This is a professional, rules-first strategy. It **reduces false signals** by requiring structure, confluence, and confirmation. It is designed to be **auditable**, reproducible, and safe for algorithmic execution.
