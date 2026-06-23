"""Optional welfare summaries and comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.interpolation import interp_policy


def summarize_state_welfare(
    result: dict,
    model_key: str,
    beta: float,
    sigma: float,
    shock_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate welfare summaries and state-level welfare samples."""
    vf = np.asarray(result["value_function"], dtype=float)

    if vf.ndim == 1:
        vf = vf[np.newaxis, :]
        grid = np.array([0.0])
    elif model_key == "model3":
        grid = np.asarray(result.get("grid", {}).get("a_grid", np.arange(vf.shape[0])), dtype=float)
    else:
        grid = np.asarray(result.get("grid", np.arange(vf.shape[0])), dtype=float)

    summary_rows = []
    detail_rows = []
    for shock_idx, label in enumerate(shock_labels):
        column = vf[:, shock_idx]
        summary_rows.append(
            {
                "shock_state": label,
                "mean_value": float(np.mean(column)),
                "min_value": float(np.min(column)),
                "max_value": float(np.max(column)),
            }
        )
        if model_key in ("model1", "model2"):
            ce = certainty_equivalent_consumption(column, beta=beta, sigma=sigma)
            summary_rows[-1]["mean_ce_consumption"] = float(np.mean(ce))
        for point in np.linspace(grid[0], grid[-1], min(5, len(grid))):
            detail_rows.append(
                {
                    "shock_state": label,
                    "state": float(point),
                    "value": float(interp_policy(grid, column, np.array([point]))[0]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def certainty_equivalent_consumption(value_array: np.ndarray, beta: float, sigma: float) -> np.ndarray:
    """Convert lifetime value into a constant-consumption equivalent for CRRA utility."""
    v = np.asarray(value_array, dtype=float)
    flow_value = (1.0 - beta) * v
    if abs(sigma - 1.0) < 1e-10:
        return np.exp(flow_value)
    inner = 1.0 + (1.0 - sigma) * flow_value
    inner = np.maximum(inner, 1e-12)
    return inner ** (1.0 / (1.0 - sigma))


def compare_counterfactual_welfare(
    base_result: dict,
    alt_result: dict,
    model_key: str,
    shock_labels: list[str],
) -> pd.DataFrame:
    """Compare value functions on a common grid."""
    if model_key == "model3":
        base_grid = np.asarray(base_result.get("grid", {}).get("a_grid", [0.0]), dtype=float)
        alt_grid = np.asarray(alt_result.get("grid", {}).get("a_grid", [0.0]), dtype=float)
    else:
        base_grid = np.asarray(base_result.get("grid", [0.0]), dtype=float)
        alt_grid = np.asarray(alt_result.get("grid", [0.0]), dtype=float)

    base_v = np.asarray(base_result["value_function"], dtype=float)
    alt_v = np.asarray(alt_result["value_function"], dtype=float)
    if base_v.ndim == 1:
        diff = alt_v - base_v
        return pd.DataFrame(
            {
                "shock_state": shock_labels,
                "mean_delta_value": diff,
                "max_delta_value": diff,
                "min_delta_value": diff,
            }
        )

    eval_points = np.linspace(
        max(float(base_grid[0]), float(alt_grid[0])),
        min(float(base_grid[-1]), float(alt_grid[-1])),
        min(80, len(base_grid), len(alt_grid)),
    )
    rows = []
    for shock_idx, label in enumerate(shock_labels):
        base_eval = interp_policy(base_grid, base_v[:, shock_idx], eval_points)
        alt_eval = interp_policy(alt_grid, alt_v[:, shock_idx], eval_points)
        diff = alt_eval - base_eval
        rows.append(
            {
                "shock_state": label,
                "mean_delta_value": float(np.mean(diff)),
                "max_delta_value": float(np.max(diff)),
                "min_delta_value": float(np.min(diff)),
            }
        )
    return pd.DataFrame(rows)
