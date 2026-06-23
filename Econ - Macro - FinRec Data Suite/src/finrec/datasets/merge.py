from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd


def build_merged_dataset(
    *,
    inputs: list[dict],
    merge_how: str = "outer",
    ffill: bool = False,
    log: Callable[[str, str], None] | None = None,
) -> pd.DataFrame:
    """
    Build a merged dataset from a list of input specs.

    Each spec:
      - path: csv path
      - date_col: column to interpret as date-like
      - value_col: numeric column to include
      - alias: output column name
    """

    def _log(level: str, msg: str) -> None:
        if log is not None:
            log(level, msg)

    if merge_how not in {"outer", "inner"}:
        raise ValueError("merge_how must be 'outer' or 'inner'")

    merged: pd.DataFrame | None = None
    for spec in inputs:
        path = str(spec["path"])
        date_col = str(spec["date_col"])
        value_col = str(spec["value_col"])
        alias = str(spec["alias"])

        _log("INFO", f"Loading input CSV: {path}")
        df = pd.read_csv(path)

        if date_col not in df.columns:
            raise ValueError(f"Missing date column '{date_col}' in {path}")
        if value_col not in df.columns:
            raise ValueError(f"Missing value column '{value_col}' in {path}")
        if not alias:
            raise ValueError("alias must be a non-empty string")

        s = pd.DataFrame()
        s["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
        s[alias] = pd.to_numeric(df[value_col], errors="coerce")
        s = s.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")

        merged = s if merged is None else merged.merge(s, on="date", how=merge_how)

    if merged is None:
        raise ValueError("No inputs provided.")

    merged = merged.sort_values("date").reset_index(drop=True)
    if ffill:
        merged = merged.set_index("date").sort_index().ffill().reset_index()

    return merged

