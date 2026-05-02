"""
Position sizing math for CandleLab strategies.
INST_CONFIG stays app-side — pip and pip_val are passed as arguments (dependency injection).
"""

from __future__ import annotations

DEFAULT_UNITS = 10_000


def calculate_position_units(
    direction: str,
    sl_dist: float,
    pip: float,
    pip_val: float,
) -> int:
    """
    Position size in OANDA units (signed). Matches ``_position_units`` in strategy_executor.
    """
    if sl_dist <= 0 or pip <= 0 or pip_val <= 0:
        u = DEFAULT_UNITS
    else:
        sl_pips = sl_dist / pip
        if sl_pips <= 0:
            u = DEFAULT_UNITS
        else:
            try:
                u = int(100.0 / (sl_pips * pip_val) * 100_000.0)
            except ZeroDivisionError:
                u = DEFAULT_UNITS
    if u < 1:
        u = DEFAULT_UNITS
    cap = 75_000
    u = min(max(u, 1), cap)
    d = str(direction).upper()
    return u if d in ("BUY", "LONG") else -u
