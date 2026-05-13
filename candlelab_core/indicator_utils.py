import json
import logging

import numpy as np
import pandas as pd

from .signal_engine import WINDOW_REVERSAL_PAIR
from .indicators import (
    _check_ma_alignment_detailed,
    _check_ma_cross_direction_detailed,
    _check_rsi_envelope_detailed,
    _check_rsi_extreme_detailed,
    check_ma_alignment,
    check_ma_cross_direction,
    check_ma_stable,
    check_rsi_envelope,
    check_rsi_extreme,
)

log = logging.getLogger(__name__)


def _index_to_timestamp(df: pd.DataFrame, idx: int | None):
    if idx is None or idx >= len(df) or idx < 0:
        return None
    return df.index[idx]


def _parse_indicator_filter_config(raw) -> dict | list | None:
    """
    Normalize candlelab_strategies_live.indicator_filter to a dict.
    Plain strings 'rsi' / 'ma_cross' use defaults; JSON objects use stored thresholds.
    """
    if isinstance(raw, list):
        result = []
        for item in raw:
            parsed = _parse_indicator_filter_config(item)
            if parsed is not None:
                result.append(parsed)
        return result if result else None
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        d = dict(raw)
    else:
        s = str(raw).strip()
        if s.startswith("{"):
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                return None
        else:
            typ = s.lower().replace(" ", "_")
            if typ == "rsi":
                return {"type": "rsi", "oversold": 30.0, "overbought": 70.0}
            if typ == "ma_cross":
                return {"type": "ma_cross", "direction": None}
            if typ == "ma_stable":
                return {"type": "ma_stable"}
            return None
    t = str(d.get("type", "")).strip().lower().replace(" ", "_")
    if not t:
        return None
    out: dict = {"type": t}
    if t == "rsi":
        out["oversold"] = float(d.get("oversold", 30))
        out["overbought"] = float(d.get("overbought", 70))
    elif t == "ma_cross":
        dr = d.get("direction")
        out["direction"] = str(dr).strip().lower() if dr is not None else None
        out["fast_period"] = int(d.get("fast_period", 5))
        out["slow_period"] = int(d.get("slow_period", 20))
        out["lookback"] = int(d.get("lookback", 5))
    elif t == "rsi_envelope":
        out["exhaustion_threshold"] = float(d.get("exhaustion_threshold", 30.0))
        out["recovery_threshold"] = float(d.get("recovery_threshold", 30.0))
        out["exhaustion_lookback"] = int(d.get("exhaustion_lookback", 5))
        out["recovery_window"] = int(d.get("recovery_window", 5))
    elif t == "ma_alignment":
        out["fast"] = int(d.get("fast", 5))
        out["slow"] = int(d.get("slow", 20))
    return out


def passes_indicator(
    ind_cfg: dict | list | None,
    df: pd.DataFrame,
    sig_idx: int,
    dir_str: str,
    anchor_idx: int = -1,
    has_continuation: bool = False,
) -> bool:
    if not ind_cfg:
        return True

    # Normalise to list — dict→list wrapping only here, never in parser
    if isinstance(ind_cfg, dict):
        filters = [ind_cfg]
    elif isinstance(ind_cfg, list):
        filters = ind_cfg
    else:
        return True

    # AND logic — all filters must pass
    for f in filters:
        it = str(f.get("type", "")).lower()

        if it == "rsi_envelope":
            if not check_rsi_envelope(
                df,
                sig_idx,
                dir_str,
                anchor_idx,
                has_continuation,
                exhaustion_threshold=float(f.get("exhaustion_threshold", 30.0)),
                recovery_threshold=float(f.get("recovery_threshold", 30.0)),
                exhaustion_lookback=int(f.get("exhaustion_lookback", 5)),
                recovery_window=int(f.get("recovery_window", 5)),
                window_reversal_pair=WINDOW_REVERSAL_PAIR,
            ):
                return False

        elif it == "ma_alignment":
            if not check_ma_alignment(
                df,
                sig_idx,
                dir_str,
                fast_period=int(f.get("fast", 5)),
                slow_period=int(f.get("slow", 20)),
            ):
                return False

        elif it == "ma_cross":
            cross_dir = f.get("direction")
            if cross_dir is None:
                cross_dir = "bullish" if dir_str.lower() in ("long", "buy") else "bearish"
            else:
                cross_dir = str(cross_dir).strip().lower()
            if not check_ma_cross_direction(
                df,
                sig_idx,
                cross_dir,
                anchor_idx=anchor_idx,
                has_continuation=has_continuation,
                fast_period=int(f.get("fast_period", 5)),
                slow_period=int(f.get("slow_period", 20)),
                lookback=int(f.get("lookback", 5)),
            ):
                return False

        elif it == "rsi":
            if not check_rsi_extreme(
                df,
                sig_idx,
                dir_str,
                float(f.get("oversold", 30)),
                float(f.get("overbought", 70)),
            ):
                return False

        elif it == "ma_stable":
            if not check_ma_stable(df, sig_idx, dir_str):
                return False

        else:
            log.debug("Unknown indicator_filter type %r — treating as pass", f)

    return True


def passes_indicator_detailed(
    ind_cfg: dict | list | None,
    df: pd.DataFrame,
    sig_idx: int,
    dir_str: str,
    anchor_idx: int = -1,
    has_continuation: bool = False,
) -> tuple[bool, dict]:
    meta = {
        "rsi_exhaustion_ts": None,
        "rsi_pivot_ts": None,
        "rsi_extreme_ts": None,
        "rsi_extreme_val": None,
        "rsi_momentum_dir": None,
        "ma_cross_ts": None,
        "ma_alignment_ts": None,
    }
    if not ind_cfg:
        return True, meta

    if isinstance(ind_cfg, dict):
        filters = [ind_cfg]
    elif isinstance(ind_cfg, list):
        filters = ind_cfg
    else:
        return True, meta

    for f in filters:
        it = str(f.get("type", "")).lower()

        if it == "rsi_envelope":
            ok, ex_i, piv_i = _check_rsi_envelope_detailed(
                df,
                sig_idx,
                dir_str,
                anchor_idx,
                has_continuation,
                exhaustion_threshold=float(f.get("exhaustion_threshold", 30.0)),
                recovery_threshold=float(f.get("recovery_threshold", 30.0)),
                exhaustion_lookback=int(f.get("exhaustion_lookback", 5)),
                recovery_window=int(f.get("recovery_window", 5)),
                window_reversal_pair=WINDOW_REVERSAL_PAIR,
            )
            meta["rsi_exhaustion_ts"] = _index_to_timestamp(df, ex_i)
            meta["rsi_pivot_ts"] = _index_to_timestamp(df, piv_i)
            if not ok:
                return False, meta

        elif it == "ma_alignment":
            ok, align_i = _check_ma_alignment_detailed(
                df,
                sig_idx,
                dir_str,
                fast_period=int(f.get("fast", 5)),
                slow_period=int(f.get("slow", 20)),
            )
            meta["ma_alignment_ts"] = _index_to_timestamp(df, align_i)
            if not ok:
                return False, meta

        elif it == "ma_cross":
            cross_dir = f.get("direction")
            if cross_dir is None:
                cross_dir = "bullish" if dir_str.lower() in ("long", "buy") else "bearish"
            else:
                cross_dir = str(cross_dir).strip().lower()
            ok, cross_i = _check_ma_cross_direction_detailed(
                df,
                sig_idx,
                cross_dir,
                anchor_idx=anchor_idx,
                has_continuation=has_continuation,
                fast_period=int(f.get("fast_period", 5)),
                slow_period=int(f.get("slow_period", 20)),
                lookback=int(f.get("lookback", 5)),
            )
            meta["ma_cross_ts"] = _index_to_timestamp(df, cross_i)
            if not ok:
                return False, meta

        elif it == "rsi":
            ok, ext_i, ext_val = _check_rsi_extreme_detailed(
                df,
                sig_idx,
                dir_str,
                float(f.get("oversold", 30)),
                float(f.get("overbought", 70)),
            )
            meta["rsi_extreme_ts"] = _index_to_timestamp(df, ext_i)
            if ext_val is not None and not np.isnan(ext_val):
                meta["rsi_extreme_val"] = round(float(ext_val), 1)
            if not ok:
                return False, meta

        elif it == "ma_stable":
            if not check_ma_stable(df, sig_idx, dir_str):
                return False, meta

        else:
            log.debug("Unknown indicator_filter type %r — treating as pass", f)

    if meta.get("rsi_extreme_ts") is not None or meta.get("rsi_exhaustion_ts") is not None:
        if "close" in df.columns and sig_idx >= 1:
            from candlelab_core.indicators import _rsi as _rsi_fn
            _close = df["close"].to_numpy(dtype=float)
            _rsi_arr = _rsi_fn(_close[: sig_idx + 1])
            if len(_rsi_arr) >= 2 and not np.isnan(_rsi_arr[-1]) and not np.isnan(_rsi_arr[-2]):
                meta["rsi_momentum_dir"] = "UP" if _rsi_arr[-1] > _rsi_arr[-2] else "DOWN"

    return True, meta
