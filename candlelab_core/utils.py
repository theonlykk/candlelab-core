"""
utils.py — Shared string normalization utilities for candlelab_core.
Used by both the live executor and the backtest engine to ensure
pattern name resolution is identical across all consumers.
"""


def normalise_pattern(label: str) -> str:
    """
    Stable slug for comparing DB pattern labels to detect_all column names.
    Handles slashes, spaces, dots, hyphens and collapsed underscores.
    """
    raw = str(label or "").strip()
    if not raw:
        return ""
    s = (
        raw.lower()
        .replace(".", "")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")
