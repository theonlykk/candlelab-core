# ADR-018 — RSI Annotation Refactor: View Layer Decoupling

**Status:** Accepted

## Context

`render_trade_panels()` was recomputing `_rsi()` internally to paint an “RSI N” label at the entry bar, which violates separation of concerns between the chart view and the signal/indicator core. The `indicator_meta` pipeline exists specifically to carry computed state from `candlelab_core` into the view layer without the renderer performing indicator math.

## Decision

- Extend `_check_rsi_extreme_detailed` to return the RSI value at the extreme bar as a third tuple element (`float | None`), alongside the existing pass/fail flag and bar index.
- Extend `passes_indicator_detailed()` metadata with `rsi_extreme_val` and `rsi_momentum_dir`, populated from core logic and (for momentum) the same Wilder RSI series used elsewhere, keyed off entry-bar context.
- Update `chart_renderer.py` so trade panels read only these meta keys for the RSI extreme label and momentum glyph—no `_rsi()` calls in the removed entry-bar RSI text path.

## Consequences

- The renderer stays dumb for this concern: all RSI math lives in `candlelab_core`.
- Future RSI display tweaks can be shipped via the core package and meta contract without editing the chart renderer’s math.
- Callers must pass `indicator_meta` through trades when they want the new labels; missing meta continues to mean no extra RSI UI.

## What this does NOT do

- Does not persist indicator metadata in the database.
- Does not modify `passes_indicator()` or other boolean hot paths.
- Does not change executor or `strategy_runner` wiring.
