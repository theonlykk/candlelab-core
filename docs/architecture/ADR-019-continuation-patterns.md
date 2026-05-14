# ADR-019 — Inside Bar Breakout and 1-Candle Flag Continuation Patterns

**Status:** Accepted

## Context

Three Soldiers/Crows was permanently dropped (−5.533 pips/trade, 4-candle lag). The home system needed replacement continuation patterns validated on institutional OneTick data before porting to OANDA BAM.

## Decision

Inside Bar Breakout and 1-Candle Flag were added to `candlelab_core`. Specs were locked by Gemini Staff Architect review. Both use strict close-based breakout triggers, strict containment (`<` and `>` only), `body_ratio >= 0.50` on the impulse candle, and a no-doji flag requirement (opposite body colour via strict `is_bull` / `is_bear`). All boolean masks are built as pandas `Series` using `.shift()` only—no manual numpy slicing—to avoid double-shift alignment bugs.

## Consequences

`detect_all()` now returns seven columns. Existing call sites stay valid; new columns appear automatically. The strategy wizard can offer these as continuation options.

## What this does NOT do

Does not change existing pattern detectors. Does not modify `signal_engine.py` or `indicator_utils.py`. Does not alter the live executor or any deployed strategy.
