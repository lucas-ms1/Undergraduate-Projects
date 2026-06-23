"""Cross-model comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_series(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < 1e-12:
        return x * 0.0
    return (x - mu) / sd


def summarize_simulation_bundle(sim_by_model: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_key, sim in sim_by_model.items():
        for var, values in sim.items():
            if var == "shock_idx":
                continue
            x = np.asarray(values, dtype=float)
            if x.ndim != 1:
                continue
            rows.append(
                {
                    "model": model_key,
                    "variable": var,
                    "mean": float(np.nanmean(x)),
                    "std": float(np.nanstd(x)),
                    "min": float(np.nanmin(x)),
                    "max": float(np.nanmax(x)),
                }
            )
    return pd.DataFrame(rows)


def normalized_forecast_panel(forecast_by_model: dict[str, dict], variable_map: dict[str, str]) -> pd.DataFrame:
    """Build long panel with normalized trajectories for model-comparison plots."""
    rows = []
    for model_key, forecast in forecast_by_model.items():
        var = variable_map.get(model_key)
        if not var or var not in forecast:
            continue
        x = normalize_series(np.asarray(forecast[var], dtype=float))
        for t, val in enumerate(x):
            rows.append({"model": model_key, "t": t, "normalized_value": float(val), "variable": var})
    return pd.DataFrame(rows)
