"""Optional empirical and structural calibration helpers."""

from __future__ import annotations

from urllib.parse import urlencode

import numpy as np
import pandas as pd

from analysis.scorecards import build_moment_scorecard


def classify_two_state_series(
    raw_series,
    method: str = "median",
    threshold: float | None = None,
) -> np.ndarray:
    """Map a numeric series into a two-state sequence in {0, 1}."""
    values = np.asarray(raw_series, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need a one-dimensional series with at least two observations.")

    if method == "threshold":
        if threshold is None:
            raise ValueError("A threshold is required for threshold classification.")
        cut = float(threshold)
    else:
        cut = float(np.median(values))
    return (values >= cut).astype(int)


def estimate_transition_matrix_from_states(state_series) -> tuple[np.ndarray, pd.DataFrame]:
    """Frequency-estimate a 2x2 transition matrix from a binary state path."""
    states = np.asarray(state_series, dtype=int)
    if states.ndim != 1 or states.size < 2:
        raise ValueError("Need at least two state observations.")
    if not np.isin(states, [0, 1]).all():
        raise ValueError("State series must contain only 0 and 1.")

    counts = np.zeros((2, 2), dtype=int)
    for i, j in zip(states[:-1], states[1:]):
        counts[int(i), int(j)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_hat = np.divide(counts, row_sums, where=row_sums > 0)
    for row_idx in range(2):
        if row_sums[row_idx, 0] == 0:
            P_hat[row_idx, :] = np.array([0.5, 0.5])
    return P_hat, pd.DataFrame(counts, index=["Low", "High"], columns=["Low", "High"])


def download_fred_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download a single FRED series as a two-column DataFrame.

    Uses FRED's public CSV endpoint, so no API key is required for a simple
    one-series download.
    """
    clean_id = str(series_id).strip().upper()
    if not clean_id:
        raise ValueError("Provide a FRED series ID.")

    query = {"id": clean_id}
    if start_date:
        query["cosd"] = str(start_date)
    if end_date:
        query["coed"] = str(end_date)

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode(query)
    df = pd.read_csv(url, na_values=".")
    date_col = None
    for candidate in ("DATE", "date", "observation_date"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None or clean_id not in df.columns:
        raise ValueError("Unexpected FRED response format.")

    out = df.rename(columns={date_col: "date", clean_id: "value"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("FRED returned no usable observations for that series/date range.")
    out["series_id"] = clean_id
    return out[["date", "value", "series_id"]]


def parameter_bounds_for_model(model_key: str, include_assets: bool = True) -> dict[str, tuple[float, float]]:
    """Safe parameter bounds for optional structural calibration."""
    bounds = {
        "model1": {"beta": (0.85, 0.99), "sigma": (0.8, 4.0), "r": (0.0, 0.08)},
        "model2": {"beta": (0.85, 0.99), "alpha": (0.2, 0.5), "delta": (0.03, 0.2)},
        "model3": {"psi": (0.2, 3.0), "nu": (0.8, 4.0), "r": (0.0, 0.08)},
    }
    if model_key == "model3" and not include_assets:
        bounds["model3"] = {"psi": (0.2, 3.0), "nu": (0.8, 4.0)}
    return bounds.get(model_key, {})


def objective_from_moments(sim_moments: dict, target_moments: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Weighted distance between model-implied and target moments."""
    scorecard, objective = build_moment_scorecard(sim_moments, target_moments)
    return objective, scorecard


def run_structural_calibration(
    base_params: dict,
    selected_params: list[str],
    bounds: dict[str, tuple[float, float]],
    n_evals: int,
    evaluate_candidate,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Small-footprint random/grid search over selected parameters."""
    if not selected_params:
        raise ValueError("Select at least one parameter to calibrate.")

    rng = np.random.default_rng(int(random_seed))
    candidates = [dict(base_params)]

    if len(selected_params) == 1:
        name = selected_params[0]
        low, high = bounds[name]
        for value in np.linspace(low, high, max(2, n_evals - 1)):
            candidate = dict(base_params)
            candidate[name] = float(value)
            candidates.append(candidate)
    else:
        for _ in range(max(1, n_evals - 1)):
            candidate = dict(base_params)
            for name in selected_params:
                low, high = bounds[name]
                candidate[name] = float(rng.uniform(low, high))
            candidates.append(candidate)

    rows = []
    for idx, candidate in enumerate(candidates[:n_evals], start=1):
        objective, extras = evaluate_candidate(candidate)
        row = {"evaluation": idx, "objective": float(objective)}
        for name in selected_params:
            row[name] = float(candidate[name])
        if isinstance(extras, dict):
            row.update(extras)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("objective", ascending=True).reset_index(drop=True)
    return out
