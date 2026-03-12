"""Monte Carlo wrappers around the existing simulation module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.simulate import simulate_model


def run_distributional_simulation(
    solver_result: dict,
    shock_vals: np.ndarray,
    P: np.ndarray,
    initial_states: list[float],
    seeds: list[int],
    model_name: str,
    model_params: dict,
    T_sim: int,
) -> pd.DataFrame:
    """Repeated simulations over seeds and optional initial states."""
    rows: list[dict] = []
    for initial_state in initial_states:
        for seed in seeds:
            sim = simulate_model(
                solver_result,
                shock_vals,
                P,
                initial_state=initial_state,
                T_sim=T_sim,
                seed=seed,
                model_name=model_name,
                model_params=model_params,
            )
            row = {
                "initial_state": float(initial_state),
                "seed": int(seed),
                "low_state_share": float(np.mean(np.asarray(sim["shock_idx"]) == 0)),
            }
            for key, arr in sim.items():
                if key == "shock_idx":
                    continue
                values = np.asarray(arr, dtype=float)
                row[f"{key}_mean"] = float(np.mean(values))
                row[f"{key}_std"] = float(np.std(values))
                row[f"{key}_final"] = float(values[-1])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_distribution_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-run distributions for numeric summary columns."""
    if df.empty:
        return pd.DataFrame()

    metric_cols = [c for c in df.columns if c not in ("seed", "initial_state")]
    rows = []
    for col in metric_cols:
        values = df[col].to_numpy(dtype=float)
        rows.append(
            {
                "metric": col,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }
        )
    return pd.DataFrame(rows)
