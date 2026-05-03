"""
indicators.py — MA and RSI confirmation filters for backtest signals.
All implementations are pure numpy, no scipy, no pandas built-in TA functions.
"""
"""
Three-Phase Hysteresis Framework
=================================
All momentum indicators in candlelab-core follow this state-transition model:

Phase 1 — Setup (Exhaustion/Trend):
    The market enters an exhausted or trending state.
    RSI: crosses below oversold threshold (e.g. < 30)
    MA Cross: fast MA crosses below slow MA

Phase 2 — Confirmation:
    Reversal patterns fire while the exhausted state holds.
    The indicator remains in the exhausted state or is just beginning to turn.

Phase 3 — Pivot:
    Momentum direction changes — this is the entry signal.
    RSI: RSI[i] > RSI[i-1] (momentum pivot) or RSI crosses back above threshold
    MA Cross: fast MA crosses above slow MA within lookback window

Design principles:
- Direction change is the signal, not level crossing
- Confluence on the signal bar is a feature, not a bug
- Staleness is the real enemy — lookback windows guard against stale signals
- Exit threshold != entry threshold (hysteresis)
- If an indicator cannot be expressed in Setup/Confirmation/Pivot terms,
  it does not belong in a reversal strategy filter

New indicators must implement check_{name}() following this contract:
    def check_{name}(df, sig_idx, dir_str, anchor_idx, has_continuation, **kwargs) -> bool
"""
import numpy as np
import pandas as pd


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Simple moving average implemented via convolution.

    Returns an array of the same length as `arr` padded with NaNs until `period-1`.
    """
    result = np.full(len(arr), np.nan)
    if len(arr) < period:
        return result
    kernel = np.ones(period) / period
    valid = np.convolve(arr, kernel, mode="valid")
    result[period - 1 :] = valid
    return result


def _rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI implementation using Wilder-style exponential smoothing.

    Returns an array of the same length as `arr` padded with NaNs until `period`.
    """
    result = np.full(len(arr), np.nan)
    if len(arr) < period + 1:
        return result
    delta = np.diff(arr)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return result


def check_ma_cross_direction(
    df: pd.DataFrame,
    signal_idx: int,
    required: str,
    anchor_idx: int = -1,
    has_continuation: bool = False,
    fast_period: int = 5,
    slow_period: int = 20,
    lookback: int = 3,
) -> bool:
    """
    Three-Phase Hysteresis MA cross: setup (opposite alignment), then pivot
    (crossover) within the scan window ending on the signal bar (inclusive).
    """
    close = df["close"].to_numpy(dtype=float)
    sma_fast = _sma(close[: signal_idx + 1], fast_period)
    sma_slow = _sma(close[: signal_idx + 1], slow_period)

    if has_continuation and anchor_idx >= 0:
        start = anchor_idx
    else:
        start = max(0, signal_idx - lookback)
    end = signal_idx + 1

    s_fast = sma_fast[start:end]
    s_slow = sma_slow[start:end]
    valid = ~(np.isnan(s_fast) | np.isnan(s_slow))
    if np.sum(valid) < 2:
        return False
    s_fast_v = s_fast[valid]
    s_slow_v = s_slow[valid]

    req = (required or "bullish").strip().lower()
    if req == "bullish":
        setup_ok = bool(np.any(s_fast_v < s_slow_v))
    else:
        setup_ok = bool(np.any(s_fast_v > s_slow_v))
    if not setup_ok:
        return False

    if req == "bullish":
        cross = bool(
            np.any(
                (s_fast_v[:-1] < s_slow_v[:-1]) & (s_fast_v[1:] >= s_slow_v[1:])
            )
        )
    else:
        cross = bool(
            np.any(
                (s_fast_v[:-1] > s_slow_v[:-1]) & (s_fast_v[1:] <= s_slow_v[1:])
            )
        )
    return cross


def check_ma_cross(df: pd.DataFrame, signal_idx: int) -> bool:
    """Backward-compatible: bullish cross only."""
    return check_ma_cross_direction(df, signal_idx, "bullish")


def check_rsi_extreme(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    oversold: float = 30.0,
    overbought: float = 70.0,
    lookback: int = 10,
) -> bool:
    """
    Returns True if RSI(14) was below oversold (long) or above overbought (short)
    at any point in the lookback candles before signal_idx.
    """
    close = df["close"].to_numpy(dtype=float)
    end = signal_idx
    start = max(0, end - lookback)

    rsi = _rsi(close[:end])
    window = rsi[start:end]
    window = window[~np.isnan(window)]

    if len(window) == 0:
        return False

    if direction == "long":
        return bool(np.any(window < float(oversold)))
    return bool(np.any(window > float(overbought)))


def _ma_stable_progressive_closes(close: np.ndarray, end: int, direction: str) -> bool:
    """
    Last 7 closes before `end` give 6 consecutive step comparisons.
    Long: count steps where close[i] > close[i-1]. Short: close[i] < close[i-1].
    Require at least 4 of 6.
    """
    if end < 7:
        return False
    window = close[end - 7 : end]
    d = str(direction).strip().lower()
    if d == "long":
        n_prog = int(np.sum(window[1:] > window[:-1]))
    else:
        n_prog = int(np.sum(window[1:] < window[:-1]))
    return n_prog >= 4


def check_ma_stable(
    df: pd.DataFrame, signal_idx: int, direction: str, lookback: int = 10
) -> bool:
    """
    Returns True if both:
    - the 5-period and 20-period SMA are sloping in the direction of the trade
      for ALL lookback candles before signal_idx, and
    - at least 4 of the last 6 close-to-close steps (from 7 closes ending at the
      bar before signal_idx) are progressive in the trade direction.
    """
    close = df["close"].to_numpy(dtype=float)
    end = signal_idx
    start = max(0, end - lookback)
    if end - start < 2:
        return False

    sma5 = _sma(close[:end], 5)
    sma20 = _sma(close[:end], 20)

    s5 = sma5[start:end]
    s20 = sma20[start:end]

    valid = ~(np.isnan(s5) | np.isnan(s20))
    if np.sum(valid) < 2:
        return False

    s5v = s5[valid]
    s20v = s20[valid]

    d = str(direction).strip().lower()
    if d == "long":
        ma_ok = bool(np.all(np.diff(s5v) > 0) and np.all(np.diff(s20v) > 0))
    else:
        ma_ok = bool(np.all(np.diff(s5v) < 0) and np.all(np.diff(s20v) < 0))
    if not ma_ok:
        return False
    return _ma_stable_progressive_closes(close, end, direction)


def check_ma_alignment(
    df: pd.DataFrame,
    sig_idx: int,
    dir_str: str,
    fast_period: int = 5,
    slow_period: int = 20,
) -> bool:
    """
    Point-in-time MA alignment check for continuation strategies.
    Phase 3 (Pivot) only — no lookback window.
    Long: fast_ma > slow_ma at sig_idx - 1.
    Short: fast_ma < slow_ma at sig_idx - 1.
    """
    if sig_idx < 1:
        return False
    close = df["close"].to_numpy(dtype=float)
    fast = _sma(close[:sig_idx], fast_period)
    slow = _sma(close[:sig_idx], slow_period)
    f_val = fast[-1]
    s_val = slow[-1]
    if np.isnan(f_val) or np.isnan(s_val):
        return False
    if dir_str == "long":
        return bool(f_val > s_val)
    return bool(f_val < s_val)


def check_rsi_envelope(
    df: pd.DataFrame,
    sig_idx: int,
    dir_str: str,
    anchor_idx: int,
    has_continuation: bool,
    exhaustion_threshold: float = 30.0,
    recovery_threshold: float = 30.0,
    exhaustion_lookback: int = 5,
    recovery_window: int = 5,
    window_reversal_pair: int = 10,
) -> bool:
    """
    Three-Phase Hysteresis RSI filter.

    Phase 1 (Exhaustion): RSI must touch below exhaustion_threshold (long)
        or above 100-exhaustion_threshold (short) within
        [anchor_idx - exhaustion_lookback, proxy_reversal_end].

    Phase 3 (Recovery):
        has_continuation=False: RSI[sig_idx] > RSI[sig_idx-1] (momentum pivot)
        has_continuation=True: RSI crosses back through recovery_threshold
            in [proxy_reversal_end + 1, sig_idx]

    If anchor_idx < 0: log warning and return False — should not occur in
    normal operation after Phase 1b-i anchor tracking is wired in.

    window_reversal_pair: passed in to avoid circular import with signal_engine.
    """
    import logging

    log = logging.getLogger(__name__)

    if anchor_idx < 0:
        log.warning(
            "check_rsi_envelope called with anchor_idx=%d — returning False",
            anchor_idx,
        )
        return False

    close = df["close"].to_numpy(dtype=float)
    rsi = _rsi(close[: sig_idx + 1])

    proxy_reversal_end = min(anchor_idx + window_reversal_pair, sig_idx - 1)
    exhaust_start = max(0, anchor_idx - exhaustion_lookback)
    exhaust_window = rsi[exhaust_start : proxy_reversal_end + 1]
    exhaust_window = exhaust_window[~np.isnan(exhaust_window)]

    if len(exhaust_window) == 0:
        return False

    if dir_str == "long":
        exhausted = bool(np.any(exhaust_window < float(exhaustion_threshold)))
    else:
        exhausted = bool(np.any(exhaust_window > (100.0 - float(exhaustion_threshold))))

    if not exhausted:
        return False

    if not has_continuation:
        if sig_idx < 1:
            return False
        if dir_str == "long":
            return bool(rsi[sig_idx] > rsi[sig_idx - 1])
        return bool(rsi[sig_idx] < rsi[sig_idx - 1])
    else:
        recovery_start = proxy_reversal_end + 1
        recovery_end = sig_idx + 1
        if recovery_start >= recovery_end:
            return False
        rec_window = rsi[recovery_start:recovery_end]
        rec_window = rec_window[~np.isnan(rec_window)]
        if len(rec_window) == 0:
            return False
        if dir_str == "long":
            return bool(np.any(rec_window > float(recovery_threshold)))
        else:
            return bool(np.any(rec_window < (100.0 - float(recovery_threshold))))
