# ADR-098 — Pattern Context Fixes

**Status:** Accepted

**Version:** candlelab-core v1.4.0

## Context

Several reversal pattern detectors in `candlelab_core/patterns.py` fired without sufficient contextual confirmation. Pin bars used a coarse 3-bar close diff; engulfing had no prior-trend filter; morning/evening star lacked FX-appropriate body-gap rules. These gaps produced false positives on EUR/USD 5-minute continuous data.

## Decision

Apply three targeted fixes in `patterns.py` and bump the package version to **1.4.0**.

### 1. `morning_evening_star` — FX body gap requirement

Add institutional FX body-gap checks before the morning and evening boolean blocks:

- **Morning star:** star body entirely at or below the prior bearish close (`gap_down_into_star`), and the current bar opens above the star body (`gap_up_from_star`).
- **Evening star:** star body entirely at or above the prior bullish close (`gap_up_into_star`), and the current bar opens below the star body (`gap_down_from_star`).

The existing `star` definition (middle candle body &lt; 30% of candle-2 body) and midpoint close penetration rules are unchanged.

#### FX body gap vs equity gap

On equity markets, overnight session gaps are discrete open-to-close discontinuities. FX trades continuously; “gaps” are expressed as **body separation** between adjacent candles rather than empty price space between sessions. ADR-098 uses body-bound comparisons (`star_body_high`, `star_body_low`) against the prior candle’s close and the current open — the standard institutional interpretation for 24-hour FX star patterns.

### 2. `engulfing` — prior trend requirement

Require a 5-bar net directional move before the engulfing candle:

```python
prior_trend = df["close"].shift(1).diff(5)
```

- Bullish engulfing: `prior_trend < 0` (prior downtrend).
- Bearish engulfing: `prior_trend > 0` (prior uptrend).

#### `.shift(1)` rationale

Without `.shift(1)`, `diff(5)` at bar `t` includes the engulfing candle’s close in the trend window. Shifting by one measures `close[t-1] - close[t-6]`, so the trend ends at the bar **before** the engulfing candle — the correct context for a reversal signal.

### 3. `hammer_hanging_man` and `shooting_star_inverted_hammer` — stronger trend detection

Replace the 3-bar `close.diff(3)` trend filter with:

1. **5-bar net move:** `prior_trend = df["close"].shift(1).diff(5)` — trend ending at the bar before the pin bar.
2. **Majority direction:** count declining or rising close-to-close steps over bars `t-1` through `t-5`; require at least 3 of 5.

| Pattern | Net trend | Majority closes |
|---------|-----------|-----------------|
| Hammer | `prior_trend < 0` | `declining >= 3` |
| Hanging Man | `prior_trend > 0` | `rising >= 3` |
| Shooting Star | `prior_trend > 0` | `rising >= 3` |
| Inverted Hammer | `prior_trend < 0` | `declining >= 3` |

The `.shift(1)` on `prior_trend` and on each `close_changes.shift(n)` ensures the pin bar itself is excluded from trend measurement.

## Consequences

- **Positive:** Reversal patterns require demonstrable prior trend and FX-correct star gaps, reducing context-free false positives.
- **Positive:** Trend measurement is consistent across engulfing and pin-bar detectors (5-bar net + shift exclusion).
- **Note:** Signal counts for affected patterns will decrease; backtest baselines should be re-run.
- **Unchanged:** `three_soldiers_crows`, `detect_inside_bar_breakout`, `detect_1_candle_flag`, and all `_body` / `_bull` / `_bear` helpers.

## What this does NOT do

Does not modify `__init__.py`, `indicators.py`, or `signal_engine.py`. Does not change continuation pattern detectors or the `PATTERNS` registry membership.
