"""Scenario engine and fan-chart helpers for advanced mode."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.forecast import forecast_model


def build_named_shock_path(name: str, horizon: int, low_state: int = 0, high_state: int = 1) -> list[int]:
    h = int(max(1, horizon))
    if name == "Baseline alternating":
        return [high_state if t % 2 else low_state for t in range(h)]
    if name == "Persistent recession":
        half = max(2, h // 2)
        return [low_state] * half + [high_state] * (h - half)
    if name == "Boom then reversal":
        half = max(2, h // 2)
        return [high_state] * half + [low_state] * (h - half)
    if name == "All low":
        return [low_state] * h
    return [high_state] * h


def scenario_compare(
    solver_result: dict,
    shock_vals: np.ndarray,
    current_state: float,
    model_name: str,
    model_params: dict,
    baseline_path: list[int],
    alternative_path: list[int],
) -> dict[str, dict]:
    base = forecast_model(
        solver_result=solver_result,
        shock_vals=shock_vals,
        shock_path=baseline_path,
        current_state=float(current_state),
        model_name=model_name,
        model_params=model_params,
    )
    alt = forecast_model(
        solver_result=solver_result,
        shock_vals=shock_vals,
        shock_path=alternative_path,
        current_state=float(current_state),
        model_name=model_name,
        model_params=model_params,
    )
    return {"baseline": base, "alternative": alt}


def fan_chart_percentiles(
    solver_result: dict,
    shock_vals: np.ndarray,
    current_state: float,
    model_name: str,
    model_params: dict,
    P: np.ndarray,
    horizon: int = 20,
    n_paths: int = 200,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Generate p10/p50/p90 percentile paths for each forecast variable."""
    rng = np.random.default_rng(int(seed))
    P = np.asarray(P, dtype=float)
    horizon = int(horizon)
    all_paths: dict[str, list[np.ndarray]] = {}
    n_states = P.shape[0]

    for _ in range(int(n_paths)):
        s = 0
        shock_path = []
        for _t in range(horizon):
            shock_path.append(s)
            s = int(rng.choice(n_states, p=P[s]))
        fc = forecast_model(
            solver_result=solver_result,
            shock_vals=shock_vals,
            shock_path=shock_path,
            current_state=float(current_state),
            model_name=model_name,
            model_params=model_params,
        )
        for k, v in fc.items():
            if k == "shock_idx":
                continue
            arr = np.asarray(v, dtype=float)
            if arr.shape[0] != horizon:
                continue
            all_paths.setdefault(k, []).append(arr)

    out = {}
    for var, stack_list in all_paths.items():
        stack = np.vstack(stack_list)
        out[var] = pd.DataFrame(
            {
                "t": np.arange(horizon),
                "p10": np.percentile(stack, 10, axis=0),
                "p50": np.percentile(stack, 50, axis=0),
                "p90": np.percentile(stack, 90, axis=0),
            }
        )
    return out
