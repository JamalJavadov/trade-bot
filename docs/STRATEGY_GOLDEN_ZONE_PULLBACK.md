# Golden Zone Pullback Strategy (Execution Checklist)

## Quick Purpose
A concise, professional checklist for executing Fibonacci Golden Zone pullback trades with strict structure, confluence, and engulfing confirmation. This is the same strategy the bot enforces.

---

## A) Swing Mode (4H)
**1) Trend Filter**
- Uptrend = HH/HL, Downtrend = LH/LL.
- If structure is choppy or range-bound → **NO TRADE**.

**2) Impulse Leg**
- Select the most recent clean expansion in trend direction.

**3) Fibonacci Anchor (Wick-to-Wick)**
- Uptrend: 0% at swing high wick, 100% at swing low wick.
- Downtrend: 0% at swing low wick, 100% at swing high wick.
- If price ignores the fib, keep the true extreme fixed and adjust the start to a swing that price respects.

**4) Golden Zone**
- Only trade 50%–61.8%.
- Ignore 38.2% as a primary entry.

**5) Confluence (need ≥ 1)**
- Swap Zone (S/R flip)
- 50 EMA slope + price position
- Anchored VWAP from impulse start

**6) Golden Respect (mandatory)**
- Slowing into zone, wick rejections, multiple candles failing to break deeper, small bodies.
- No reaction → no trade.

**7) Entry Trigger (mandatory)**
- Bullish/Bearish Engulfing candle at the zone.
- Aggressive entry: close of engulfing.
- Safer entry: break of engulfing high/low.

**8) Stop & Targets**
- Stop: beyond fib anchor swing wick extreme.
- TP1: 0% level (impulse extreme).
- Take 50–70% at TP1, move stop to BE or structure.
- Runner: exit on structure exhaustion (stall, rejection, failed push).

---

## B) Precision Mode (15M execution + 4H bias)
**Required checklist**
1. HTF premium/discount alignment on 4H.
2. Liquidity sweep (wick through major high/low).
3. BOS + FVG.
4. 71% retracement entry (best if aligned with FVG).

**Execution rules**
- Limit order at 71%.
- Stop at 100% extreme.
- Take profit at 0% extreme.

---

## Notes
- This framework prioritizes **structure, confluence, and confirmation** over frequency.
- If any rule fails, the setup is invalid—no discretionary overrides.
