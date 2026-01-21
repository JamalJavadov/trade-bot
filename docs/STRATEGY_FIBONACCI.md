# Fibonacci Strategy Blueprint (Codex Reference)

> **Purpose:** This document translates the provided Fibonacci trading videos into a precise, repeatable strategy that the bot can follow and that a human can audit. It is a professional, rule-based blueprint with clear entries, confirmations, exits, and risk controls.

## 1) Core Philosophy
- Fibonacci levels are **not magic**; they are *measurement tools* for identifying **structured pullbacks** within a trend.
- A valid setup needs **structure + confluence + confirmation + risk control**.
- Levels are **zones**, not single price lines. We treat them as *areas of reaction*.

## 2) Chart Setup & Tools
**Timeframes**
1. **4H** – higher-timeframe bias (trend + premium/discount).
2. **1H** – main impulse leg selection + golden zone setup.
3. **15M** – confirmation + measurement strategy with BOS/FVG.

**Indicators / Tools**
- Fibonacci Retracement (standard levels).
- 50 EMA (trend + dynamic S/R).
- Anchored VWAP (from impulse start).
- Fractals (5-period) to locate swing highs/lows.
- Optional: Fib Time Zones (timing confluence, optional).

## 3) Fibonacci Settings
Use the following retracement levels:
- 0.382, 0.50, **0.618** (Golden Zone)
- Optional: 0.786 (deep retrace)

Extensions for targets:
- **-0.272** (first extension)
- **-0.618** (aggressive extension)

Measurement entry zone:
- **0.71 – 0.75** (precision retrace zone)

## 4) Swing Selection (Precision Anchor)
1. Identify the **impulse leg** in the trend direction.
2. Anchor Fibonacci **from wick to wick**:
   - **Uptrend:** swing low wick → swing high wick.
   - **Downtrend:** swing high wick → swing low wick.
3. If price ignores the first anchor (no reactions), adjust to the **next significant swing**.

## 5) Golden Zone Strategy (Primary)
**Idea:** In a trend, price often retraces into the **50–61.8% zone** before continuation.

### 5.1 Entry Zone
- **Long:** 0.50–0.618 retracement of the impulse.
- **Short:** 0.50–0.618 retracement of the impulse.

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
- **TP2:** extension -0.272 (and optionally -0.618).

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

## 9) Operational Checklist (Human or Bot)
1. Identify 4H trend bias (EMA or premium/discount).
2. Find impulse leg on 1H and draw Fibonacci.
3. Check if price is in the Golden Zone (50–61.8).
4. Require at least one confluence.
5. Confirm with a candlestick reversal or rejection wick.
6. Compute SL/TPs and RR.
7. If all rules pass → **OK**. Else → **SETUP** (watch).

## 10) Parameters (Bot Settings Reference)
Key settings in `settings.json`:
- `strategy.min_confluence`
- `strategy.zone_tolerance_atr`
- `strategy.allow_pre_zone`
- `risk.min_rr2`
- `risk.max_entry_distance_atr`
- `scoring.w_confluence_count`
- `scoring.w_confirmation`

---

### Final Notes
This is a professional, rules-first strategy. It **reduces false signals** by requiring structure, confluence, and confirmation. It is designed to be **auditable**, reproducible, and safe for algorithmic execution.
