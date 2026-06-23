"""Regime-duration analytics for simulated two-state Markov paths."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_spells(shock_idx: np.ndarray) -> pd.DataFrame:
    """Return one row per contiguous run of the same shock state."""
    shocks = np.asarray(shock_idx, dtype=int)
    if shocks.size == 0:
        return pd.DataFrame(columns=["state", "start", "end", "length"])

    states = [int(shocks[0])]
    starts = [0]
    lengths = []

    current = int(shocks[0])
    run_start = 0
    for t in range(1, shocks.size):
        if int(shocks[t]) != current:
            lengths.append(t - run_start)
            current = int(shocks[t])
            states.append(current)
            starts.append(t)
            run_start = t
    lengths.append(shocks.size - run_start)

    ends = [s + l - 1 for s, l in zip(starts, lengths)]
    return pd.DataFrame(
        {"state": states, "start": starts, "end": ends, "length": lengths}
    )


def compute_transition_summary(shock_idx: np.ndarray, n_states: int = 2) -> dict[str, pd.DataFrame]:
    """Transition counts and empirical probabilities from a simulated path."""
    shocks = np.asarray(shock_idx, dtype=int)
    counts = np.zeros((n_states, n_states), dtype=int)
    for i, j in zip(shocks[:-1], shocks[1:]):
        counts[int(i), int(j)] += 1

    probs = counts.astype(float)
    row_sums = probs.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(probs, row_sums, where=row_sums > 0)
    probs[row_sums[:, 0] == 0] = np.nan

    return {
        "counts": pd.DataFrame(counts),
        "probabilities": pd.DataFrame(probs),
    }


def summarize_spell_stats(spells: pd.DataFrame, state_labels: list[str] | None = None) -> pd.DataFrame:
    """Aggregate spell durations by regime."""
    if spells.empty:
        return pd.DataFrame(
            columns=["state", "n_spells", "mean_duration", "median_duration", "max_duration", "share_of_time"]
        )

    total_periods = float(spells["length"].sum())
    grouped = spells.groupby("state")["length"]
    summary = grouped.agg(["count", "mean", "median", "max", "sum"]).reset_index()
    summary.columns = [
        "state",
        "n_spells",
        "mean_duration",
        "median_duration",
        "max_duration",
        "time_in_state",
    ]
    summary["share_of_time"] = summary["time_in_state"] / total_periods
    summary = summary.drop(columns="time_in_state")
    if state_labels:
        summary["state"] = summary["state"].map(
            {idx: label for idx, label in enumerate(state_labels)}
        )
    return summary


def conditional_series_stats(
    sim: dict[str, np.ndarray],
    shock_idx: np.ndarray,
    state_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Mean and volatility of each simulated series by regime."""
    shocks = np.asarray(shock_idx, dtype=int)
    labels = state_labels or [f"State {idx}" for idx in sorted(np.unique(shocks))]
    rows: list[dict] = []
    for key, values in sim.items():
        if key == "shock_idx":
            continue
        arr = np.asarray(values, dtype=float)
        for idx, label in enumerate(labels):
            mask = shocks == idx
            if np.any(mask):
                rows.append(
                    {
                        "variable": key,
                        "state": label,
                        "mean": float(np.mean(arr[mask])),
                        "std": float(np.std(arr[mask])),
                        "min": float(np.min(arr[mask])),
                        "max": float(np.max(arr[mask])),
                    }
                )
    return pd.DataFrame(rows)
