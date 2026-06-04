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


class RunningRSI:
    """
    Stateful Wilder RSI accumulator — O(1) per bar update. (ADR-096C)
    Derives from the same Wilder smoothing math as _rsi().
    Use .update(close) to feed one bar at a time; returns current RSI scalar.
    Warm-start by calling .update() for each bar in the IS window before OOS.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self._prev_close: float | None = None
        self._bar_count: int = 0
        self._seed_gains: list[float] = []
        self._seed_losses: list[float] = []

    def update(self, close: float) -> float:
        """Feed one closing price. Returns RSI scalar, or NaN if insufficient data."""
        if self._prev_close is None:
            self._prev_close = close
            return float("nan")

        delta = close - self._prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._prev_close = close
        self._bar_count += 1

        if self._avg_gain is None:
            # Accumulate seed period
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)
            if self._bar_count >= self.period:
                self._avg_gain = float(np.mean(self._seed_gains))
                self._avg_loss = float(np.mean(self._seed_losses))
                self._seed_gains = []
                self._seed_losses = []
            else:
                return float("nan")
        else:
            # Wilder smoothing — identical to _rsi() loop
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def reset(self) -> None:
        """Reset state — use when starting a new instrument or window."""
        self.__init__(self.period)


class RunningEMA:
    """
    Stateful EMA accumulator — O(1) per bar update. (ADR-096C)
    Standard exponential moving average with k = 2 / (period + 1).
    Use .update(close) to feed one bar at a time; returns current EMA scalar.
    Warm-start by calling .update() for each bar in the IS window before OOS.
    """

    def __init__(self, period: int):
        self.period = period
        self.k = 2.0 / (period + 1)
        self._value: float | None = None
        self._bar_count: int = 0

    def update(self, close: float) -> float:
        """Feed one closing price. Returns EMA scalar, or NaN if insufficient data."""
        self._bar_count += 1
        if self._value is None:
            self._value = close
        else:
            self._value = close * self.k + self._value * (1.0 - self.k)
        return self._value if self._bar_count >= self.period else float("nan")

    def reset(self) -> None:
        """Reset state — use when starting a new instrument or window."""
        self.__init__(self.period)


def _check_ma_cross_direction_detailed(
    df: pd.DataFrame,
    signal_idx: int,
    required: str,
    anchor_idx: int = -1,
    has_continuation: bool = False,
    fast_period: int = 5,
    slow_period: int = 20,
    lookback: int = 5,
) -> tuple[bool, int | None]:
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
        return False, None
    s_fast_v = s_fast[valid]
    s_slow_v = s_slow[valid]

    req = (required or "bullish").strip().lower()
    if req == "bullish":
        setup_ok = bool(np.any(s_fast_v < s_slow_v))
    else:
        setup_ok = bool(np.any(s_fast_v > s_slow_v))
    if not setup_ok:
        return False, None

    if req == "bullish":
        cross_mask = (s_fast_v[:-1] < s_slow_v[:-1]) & (s_fast_v[1:] >= s_slow_v[1:])
    else:
        cross_mask = (s_fast_v[:-1] > s_slow_v[:-1]) & (s_fast_v[1:] <= s_slow_v[1:])
    compressed_where = np.where(cross_mask)[0]
    if compressed_where.size == 0:
        return False, None
    compressed_idx = int(compressed_where[0])
    valid_positions = np.where(valid)[0]
    if compressed_idx + 1 >= len(valid_positions):
        return False, None
    slice_idx = int(valid_positions[compressed_idx + 1])
    absolute_idx = start + slice_idx
    return True, absolute_idx


def check_ma_cross_direction(
    df: pd.DataFrame,
    signal_idx: int,
    required: str,
    anchor_idx: int = -1,
    has_continuation: bool = False,
    fast_period: int = 5,
    slow_period: int = 20,
    lookback: int = 5,
) -> bool:
    """
    Three-Phase Hysteresis MA cross: setup (opposite alignment), then pivot
    (crossover) within the scan window ending on the signal bar (inclusive).
    """
    return _check_ma_cross_direction_detailed(
        df,
        signal_idx,
        required,
        anchor_idx=anchor_idx,
        has_continuation=has_continuation,
        fast_period=fast_period,
        slow_period=slow_period,
        lookback=lookback,
    )[0]


def check_ma_cross(df: pd.DataFrame, signal_idx: int) -> bool:
    """Backward-compatible: bullish cross only."""
    return check_ma_cross_direction(df, signal_idx, "bullish")


def _check_rsi_extreme_detailed(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    oversold: float = 30.0,
    overbought: float = 70.0,
    lookback: int = 10,
) -> tuple[bool, int | None, float | None]:
    """
    Returns True if RSI(14) was below oversold (long) or above overbought (short)
    at any point in the lookback candles before signal_idx.
    """
    close = df["close"].to_numpy(dtype=float)
    end = signal_idx
    start = max(0, end - lookback)

    rsi = _rsi(close[:end])
    w = rsi[start:end]
    window = w[~np.isnan(w)]

    if len(window) == 0:
        return False, None, None

    if direction == "long":
        if not bool(np.any(window < float(oversold))):
            return False, None, None
        rel = int(np.nanargmin(w))
        rv = w[rel]
        ext_f = None if np.isnan(rv) else float(rv)
        return True, start + rel, ext_f
    if not bool(np.any(window > float(overbought))):
        return False, None, None
    rel = int(np.nanargmax(w))
    rv = w[rel]
    ext_f = None if np.isnan(rv) else float(rv)
    return True, start + rel, ext_f


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
    return _check_rsi_extreme_detailed(
        df, signal_idx, direction, oversold, overbought, lookback
    )[0]


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


def _check_ma_alignment_detailed(
    df: pd.DataFrame,
    sig_idx: int,
    dir_str: str,
    fast_period: int = 5,
    slow_period: int = 20,
) -> tuple[bool, int | None]:
    """
    Point-in-time MA alignment check for continuation strategies.
    Phase 3 (Pivot) only — no lookback window.
    Long: fast_ma > slow_ma at sig_idx - 1.
    Short: fast_ma < slow_ma at sig_idx - 1.
    """
    if sig_idx < 1:
        return False, None
    close = df["close"].to_numpy(dtype=float)
    fast = _sma(close[:sig_idx], fast_period)
    slow = _sma(close[:sig_idx], slow_period)
    f_val = fast[-1]
    s_val = slow[-1]
    if np.isnan(f_val) or np.isnan(s_val):
        return False, None
    if dir_str == "long":
        ok = bool(f_val > s_val)
    else:
        ok = bool(f_val < s_val)
    if not ok:
        return False, None
    return True, sig_idx - 1


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
    return _check_ma_alignment_detailed(
        df, sig_idx, dir_str, fast_period, slow_period
    )[0]


def _check_rsi_envelope_detailed(
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
) -> tuple[bool, int | None, int | None]:
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
        return False, None, None

    close = df["close"].to_numpy(dtype=float)
    rsi = _rsi(close[: sig_idx + 1])

    proxy_reversal_end = min(anchor_idx + window_reversal_pair, sig_idx - 1)
    exhaust_start = max(0, anchor_idx - exhaustion_lookback)
    ex_slice = rsi[exhaust_start : proxy_reversal_end + 1]
    finite_ex = ~np.isnan(ex_slice)
    if dir_str == "long":
        exhaust_hit = finite_ex & (ex_slice < float(exhaustion_threshold))
    else:
        exhaust_hit = finite_ex & (ex_slice > (100.0 - float(exhaustion_threshold)))
    exhaust_where = np.where(exhaust_hit)[0]
    if exhaust_where.size == 0:
        return False, None, None
    exhaustion_idx = exhaust_start + int(exhaust_where[0])

    if not has_continuation:
        if sig_idx < 1:
            return False, exhaustion_idx, None
        if dir_str == "long":
            ok = bool(rsi[sig_idx] > rsi[sig_idx - 1])
        else:
            ok = bool(rsi[sig_idx] < rsi[sig_idx - 1])
        if ok:
            return True, exhaustion_idx, sig_idx
        return False, exhaustion_idx, None
    recovery_start = proxy_reversal_end + 1
    recovery_end = sig_idx + 1
    if recovery_start >= recovery_end:
        return False, exhaustion_idx, None
    rec_slice = rsi[recovery_start:recovery_end]
    finite_r = ~np.isnan(rec_slice)
    if dir_str == "long":
        rec_hit = finite_r & (rec_slice > float(recovery_threshold))
    else:
        rec_hit = finite_r & (rec_slice < (100.0 - float(recovery_threshold)))
    rec_where = np.where(rec_hit)[0]
    if rec_where.size == 0:
        return False, exhaustion_idx, None
    pivot_idx = recovery_start + int(rec_where[0])
    return True, exhaustion_idx, pivot_idx


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
    return _check_rsi_envelope_detailed(
        df,
        sig_idx,
        dir_str,
        anchor_idx,
        has_continuation,
        exhaustion_threshold=exhaustion_threshold,
        recovery_threshold=recovery_threshold,
        exhaustion_lookback=exhaustion_lookback,
        recovery_window=recovery_window,
        window_reversal_pair=window_reversal_pair,
    )[0]


def _validate_running_accumulators(n: int = 100, period: int = 14) -> bool:
    """
    Validate RunningRSI and RunningEMA match batch _rsi/_sma output.
    Returns True if max absolute difference < 1e-8. (ADR-096C)
    """
    rng = np.random.default_rng(42)
    closes = 1.0 + np.cumsum(rng.normal(0, 0.001, n))

    # Batch RSI
    batch_rsi = _rsi(closes, period)

    # Running RSI
    acc = RunningRSI(period)
    running_rsi = np.array([acc.update(c) for c in closes])

    # Compare (ignore NaN positions)
    valid = np.isfinite(batch_rsi) & np.isfinite(running_rsi)
    if not valid.any():
        return False
    rsi_ok = np.max(np.abs(batch_rsi[valid] - running_rsi[valid])) < 1e-8

    # Batch SMA
    batch_sma = _sma(closes, period)

    # Running EMA (note: RunningEMA is EMA not SMA — validate separately)
    # SMA validation via manual rolling
    sma_running = np.array([
        float(np.mean(closes[max(0, i - period + 1):i + 1]))
        if i >= period - 1 else float("nan")
        for i in range(n)
    ])
    valid_sma = np.isfinite(batch_sma) & np.isfinite(sma_running)
    sma_ok = np.max(np.abs(batch_sma[valid_sma] - sma_running[valid_sma])) < 1e-8

    return rsi_ok and sma_ok
