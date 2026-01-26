# Fibonacci Golden Zone Pullback Strategy (Bot Blueprint)

> **Purpose:** Convert the user's Fibonacci pullback system into a precise, auditable strategy for the bot. The system only trades the **Golden Zone (50%–61.8%)** with strict structure, confluence, and engulfing confirmation. It provides both **Swing Mode (4H)** and **Precision Mode (15M execution + 4H bias)**.

## 1) Core Idea (Non-Negotiables)
- Fibonacci is a **measurement + mapping tool**, not a prediction tool.
- The bot only trades when price shows:
  1. **Correct anchoring (wick-to-wick)**
  2. **Confluence (at least one reason to react)**
  3. **Respect + confirmation**
  4. **Structured exits**
- **Golden Zone only:** 0.50–0.618.
- **Entry trigger only:** Engulfing candles (bullish or bearish).

## 2) Chart Setup
### A) Swing Mode (Recommended)
- **Timeframe:** 4H
- **Tools:**
  - Fibonacci Retracement (custom levels)
  - Fractals (5-period) for swing identification
  - Optional confluence: Swap Zones (S/R flip), 50 EMA, Anchored VWAP (from impulse start)

### B) Precision Mode (Sniper Execution)
- **Bias timeframe:** 4H
- **Execution timeframe:** 15M
- **Tools:**
  - Fibonacci Retracement (includes 0.71 level)
  - Optional: Fair Value Gap (FVG)

## 3) Market Conditions (Trade Permissions)
- **Uptrend:** Higher highs + higher lows.
- **Downtrend:** Lower highs + lower lows.
- If structure is **choppy, range-bound, or unclear → NO TRADE**.

## 4) Step 1 — Precision Anchor Rule (Wick-to-Wick)
### Uptrend
- Identify the **most recent impulsive swing up**.
- Anchor Fibonacci **0% at swing high wick**, **100% at swing low wick**.

### Downtrend
- Identify the **most recent impulsive swing down**.
- Anchor Fibonacci **0% at swing low wick**, **100% at swing high wick**.

**Market Respect Adjustment:**
- If price ignores your fib, **keep the true extreme fixed** and adjust the starting anchor to the next most obvious significant swing that price respects.

## 5) Step 2 — The Only Zone We Trade (Golden Zone)
- **Golden Zone = 50% to 61.8%**.
- 38.2% is **not** a primary entry zone.

## 6) Step 3 — Confluence Filters (Choose at least ONE)
The Golden Zone is tradable only when it aligns with at least one confluence:
- **Swap Zone (S/R flip)**
- **50 EMA dynamic support/resistance**
- **Anchored VWAP** (anchored from impulse start)

**Best setups:** Golden Zone + **2+ confluences**.

## 7) Step 4 — “Golden Respect” Rule (Mandatory)
A touch is not enough. We require **respect**:
- Price slows into the zone.
- Wicks reject the zone.
- Multiple candles fail to break deeper.
- Bodies stay relatively small (hesitation).

If price slices cleanly through 50% and 61.8% with no hesitation → **NO TRADE**.

## 8) Step 5 — Entry Trigger (Mandatory)
**Entry Trigger = Engulfing Candle at the Golden Zone.**

### Buy Entry (Long)
- Price retraces into Golden Zone + confluence.
- Shows respect.
- **Bullish Engulfing** candle forms.
- Entry options:
  - Aggressive: close of engulfing candle.
  - Safer: break of engulfing high.

### Sell Entry (Short)
- Price retraces into Golden Zone + confluence.
- Shows respect.
- **Bearish Engulfing** candle forms.
- Entry options:
  - Aggressive: close of engulfing candle.
  - Safer: break of engulfing low.

## 9) Risk Management (Stop Loss)
- **Buy:** stop below the **swing low wick** that anchored the fib.
- **Sell:** stop above the **swing high wick** that anchored the fib.

## 10) Targets & Exits
- **TP1:** 0% fib level (impulse extreme / prior swing point).
- **Partial take profit:** 50–70% at TP1.
- **Runner exit:** ride until structure exhaustion (rejections, failed pushes, momentum stall).

## 11) Precision Mode Bonus — 15M “71% + SMC” Execution
Use this when you want tighter entries with less drawdown.

**Checklist (all required):**
1. **HTF Alignment:**
   - Above 50% of 4H range = premium → look for sells.
   - Below 50% = discount → look for buys.
2. **Liquidity Sweep** (wick through a major high/low).
3. **Break of Structure (BOS)** + **Imbalance/FVG**.
4. **Entry at 71% retracement** (best if aligned with FVG).

**Execution Rules:**
- Limit order at 71%.
- Stop at 100% extreme.
- TP at 0% extreme.
- If price reacts early but does not break the fib extremes, the setup remains valid.

## 12) Optional Timing Layer — Fibonacci Time Zones
- Plot time zones from a swing high → swing low.
- Prioritize setups only when price hits Golden Zone or 71% **during a time-zone reaction window**.

## 13) Bot Implementation Notes
- The bot enforces **Golden Zone only** and **engulfing-only confirmation**.
- Confluence is required (minimum 1).
- Stops are anchored to the swing wick extremes without ATR padding.
- Measurement mode uses 71% retracement after sweep + BOS.

---

### Final Notes
This is a professional, rules-first strategy built for **clarity, repeatability, and auditability**. It avoids impulsive trades, prioritizes confluence, and executes only when price **proves respect** at the Golden Zone.
