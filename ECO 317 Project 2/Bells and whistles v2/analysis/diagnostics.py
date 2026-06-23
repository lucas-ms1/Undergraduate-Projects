"""Advanced numerical diagnostics for solved models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.interpolation import interp_policy


def get_primary_grid(result: dict, model_key: str) -> np.ndarray | None:
    """Return the main state grid if the model has one."""
    if model_key in ("model1", "model2"):
        return np.asarray(result["grid"], dtype=float)
    if model_key == "model3":
        grid = result.get("grid", {})
        if isinstance(grid, dict) and "a_grid" in grid:
            return np.asarray(grid["a_grid"], dtype=float)
    return None


def solver_diagnostics_summary(result: dict) -> pd.DataFrame:
    """Tabular summary of the built-in VFI diagnostics."""
    diag = result.get("diagnostics", {})
    rows = [
        {"metric": "Converged", "value": bool(diag.get("converged", False))},
        {"metric": "Iterations", "value": int(diag.get("iterations", 0))},
        {"metric": "Final sup-norm error", "value": float(diag.get("final_error", np.nan))},
    ]
    return pd.DataFrame(rows)


def _check_1d_policy(name: str, values: np.ndarray, lower=None, upper=None) -> dict:
    arr = np.asarray(values, dtype=float)
    diffs = np.diff(arr) if arr.size > 1 else np.array([])
    monotone = bool(np.all(diffs >= -1e-10)) if diffs.size else True
    finite = bool(np.isfinite(arr).all())
    feasible = True
    if lower is not None:
        feasible = feasible and bool(np.all(arr >= lower - 1e-10))
    if upper is not None:
        feasible = feasible and bool(np.all(arr <= upper + 1e-10))
    return {
        "object": name,
        "finite": finite,
        "monotone_non_decreasing": monotone,
        "minimum": float(np.nanmin(arr)) if arr.size else np.nan,
        "maximum": float(np.nanmax(arr)) if arr.size else np.nan,
        "feasible_range": feasible,
    }


def policy_shape_checks(result: dict, model_key: str) -> pd.DataFrame:
    """Low-risk post-solve checks that should not require re-solving."""
    rows: list[dict] = []

    value_function = np.asarray(result["value_function"], dtype=float)
    rows.append(
        {
            "object": "value_function",
            "finite": bool(np.isfinite(value_function).all()),
            "monotone_non_decreasing": np.nan,
            "minimum": float(np.nanmin(value_function)),
            "maximum": float(np.nanmax(value_function)),
            "feasible_range": True,
        }
    )

    c_policy = np.asarray(result["c_policy"], dtype=float)
    rows.append(
        {
            "object": "consumption_policy",
            "finite": bool(np.isfinite(c_policy).all()),
            "monotone_non_decreasing": np.nan,
            "minimum": float(np.nanmin(c_policy)),
            "maximum": float(np.nanmax(c_policy)),
            "feasible_range": bool(np.all(c_policy > 0.0)),
        }
    )

    if model_key in ("model1", "model2"):
        policy_levels = np.asarray(result["policy_levels"], dtype=float)
        for shock_idx in range(policy_levels.shape[1]):
            rows.append(
                _check_1d_policy(
                    f"policy_shock_{shock_idx}",
                    policy_levels[:, shock_idx],
                    lower=np.min(result["grid"]),
                    upper=np.max(result["grid"]),
                )
            )
    elif model_key == "model3":
        policy_levels = result["policy_levels"]
        labor = np.asarray(policy_levels["labor"], dtype=float)
        if labor.ndim == 1:
            rows.append(_check_1d_policy("labor_policy", labor, lower=0.0, upper=1.0))
        else:
            for shock_idx in range(labor.shape[1]):
                rows.append(
                    _check_1d_policy(
                        f"labor_policy_shock_{shock_idx}",
                        labor[:, shock_idx],
                        lower=0.0,
                        upper=1.0,
                    )
                )
        if "savings" in policy_levels:
            savings = np.asarray(policy_levels["savings"], dtype=float)
            a_grid = np.asarray(result["grid"]["a_grid"], dtype=float)
            for shock_idx in range(savings.shape[1]):
                rows.append(
                    _check_1d_policy(
                        f"savings_policy_shock_{shock_idx}",
                        savings[:, shock_idx],
                        lower=float(a_grid[0]),
                        upper=float(a_grid[-1]),
                    )
                )

    return pd.DataFrame(rows)


def edge_usage_summary(result: dict, model_key: str) -> pd.DataFrame:
    """Report how often policies sit on the lower or upper grid boundary."""
    rows: list[dict] = []
    if model_key in ("model1", "model2"):
        grid = np.asarray(result["grid"], dtype=float)
        policies = {"policy_levels": np.asarray(result["policy_levels"], dtype=float)}
    elif model_key == "model3" and "a_grid" in result.get("grid", {}):
        grid = np.asarray(result["grid"]["a_grid"], dtype=float)
        policies = {
            "savings_policy": np.asarray(result["policy_levels"]["savings"], dtype=float),
        }
    else:
        grid = None
        policies = {}

    if grid is not None:
        for name, arr in policies.items():
            rows.append(
                {
                    "object": name,
                    "share_at_lower_edge": float(np.mean(np.isclose(arr, grid[0]))),
                    "share_at_upper_edge": float(np.mean(np.isclose(arr, grid[-1]))),
                }
            )

    if model_key == "model3":
        labor = np.asarray(result["policy_levels"]["labor"], dtype=float)
        rows.append(
            {
                "object": "labor_policy",
                "share_at_lower_edge": float(np.mean(np.isclose(labor, 0.0))),
                "share_at_upper_edge": float(np.mean(np.isclose(labor, 1.0))),
            }
        )

    return pd.DataFrame(rows)


def _named_surfaces(result: dict, model_key: str) -> dict[str, np.ndarray]:
    if model_key in ("model1", "model2"):
        return {
            "value_function": np.asarray(result["value_function"], dtype=float),
            "consumption_policy": np.asarray(result["c_policy"], dtype=float),
            "policy_levels": np.asarray(result["policy_levels"], dtype=float),
        }
    if model_key == "model3":
        surfaces = {
            "value_function": np.asarray(result["value_function"], dtype=float),
            "consumption_policy": np.asarray(result["c_policy"], dtype=float),
            "labor_policy": np.asarray(result["policy_levels"]["labor"], dtype=float),
        }
        if "savings" in result["policy_levels"]:
            surfaces["savings_policy"] = np.asarray(
                result["policy_levels"]["savings"], dtype=float
            )
        return surfaces
    return {}


def grid_convergence_check(
    base_result: dict,
    compare_results: dict[str, dict],
    model_key: str,
    n_eval: int = 80,
) -> pd.DataFrame:
    """Compare solved objects across grid sizes on a common evaluation grid."""
    base_grid = get_primary_grid(base_result, model_key)
    surfaces = _named_surfaces(base_result, model_key)

    if model_key == "model3" and base_grid is None:
        rows = []
        base_surfaces = _named_surfaces(base_result, model_key)
        for label, alt_result in compare_results.items():
            alt_surfaces = _named_surfaces(alt_result, model_key)
            for name, base_arr in base_surfaces.items():
                alt_arr = alt_surfaces.get(name)
                if alt_arr is None:
                    continue
                diff = np.asarray(base_arr, dtype=float) - np.asarray(alt_arr, dtype=float)
                rows.append(
                    {
                        "comparison": label,
                        "object": name,
                        "max_abs_diff": float(np.max(np.abs(diff))),
                        "mean_abs_diff": float(np.mean(np.abs(diff))),
                    }
                )
        return pd.DataFrame(rows)

    if base_grid is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for label, alt_result in compare_results.items():
        alt_grid = get_primary_grid(alt_result, model_key)
        if alt_grid is None:
            continue
        lower = max(float(base_grid[0]), float(alt_grid[0]))
        upper = min(float(base_grid[-1]), float(alt_grid[-1]))
        eval_points = np.linspace(lower, upper, n_eval)
        alt_surfaces = _named_surfaces(alt_result, model_key)

        for name, base_arr in surfaces.items():
            alt_arr = alt_surfaces.get(name)
            if alt_arr is None:
                continue
            base_arr = np.asarray(base_arr, dtype=float)
            alt_arr = np.asarray(alt_arr, dtype=float)
            n_shocks = 1 if base_arr.ndim == 1 else base_arr.shape[1]
            diffs = []
            for shock_idx in range(n_shocks):
                base_slice = base_arr if base_arr.ndim == 1 else base_arr[:, shock_idx]
                alt_slice = alt_arr if alt_arr.ndim == 1 else alt_arr[:, shock_idx]
                base_eval = interp_policy(base_grid, base_slice, eval_points)
                alt_eval = interp_policy(alt_grid, alt_slice, eval_points)
                diffs.append(np.abs(base_eval - alt_eval))
            stacked = np.vstack(diffs)
            rows.append(
                {
                    "comparison": label,
                    "object": name,
                    "max_abs_diff": float(np.max(stacked)),
                    "mean_abs_diff": float(np.mean(stacked)),
                }
            )

    return pd.DataFrame(rows)


def iteration_path_table(result: dict) -> pd.DataFrame:
    """Iteration-by-iteration sup-norm errors from solver diagnostics."""
    diag = result.get("diagnostics", {})
    path = np.asarray(diag.get("error_path", []), dtype=float)
    if path.size == 0:
        return pd.DataFrame()
    return pd.DataFrame({"iteration": np.arange(1, path.size + 1), "error": path})


def runtime_comparison_table(results_by_method: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for method, result in results_by_method.items():
        diag = result.get("diagnostics", {})
        rows.append(
            {
                "method": method,
                "runtime_seconds": float(diag.get("runtime_seconds", np.nan)),
                "iterations": int(diag.get("iterations", 0)),
                "final_error": float(diag.get("final_error", np.nan)),
                "converged": bool(diag.get("converged", False)),
            }
        )
    return pd.DataFrame(rows)


def bellman_residuals(result: dict, model_key: str, params: dict) -> dict:
    """Compute Bellman residual arrays and summary stats."""
    V = np.asarray(result["value_function"], dtype=float)
    P = np.asarray(params["P"], dtype=float)
    beta = float(params["beta"])
    if model_key == "model1":
        a = np.asarray(result["grid"], dtype=float)
        y = np.asarray(params["y_vals"], dtype=float)
        r = float(params["r"])
        sigma = float(params["sigma"])
        ap = np.asarray(result["policy_levels"], dtype=float)
        c = (1.0 + r) * a[:, None] + y[None, :] - ap
        if sigma == 1.0:
            u = np.log(np.maximum(c, 1e-12))
        else:
            u = (np.maximum(c, 1e-12) ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
        EV = V @ P.T
        idx = np.searchsorted(a, ap)
        idx = np.clip(idx, 0, len(a) - 1)
        cont = np.take_along_axis(EV, idx, axis=0)
        rhs = u + beta * cont
        resid = V - rhs
    elif model_key == "model2":
        k = np.asarray(result["grid"], dtype=float)
        z = np.asarray(params["z_vals"], dtype=float)
        alpha = float(params["alpha"])
        delta = float(params["delta"])
        A = float(params.get("A", 1.0))
        sigma = float(params["sigma"])
        kp = np.asarray(result["policy_levels"], dtype=float)
        y = z[None, :] * A * (k[:, None] ** alpha)
        c = y + (1.0 - delta) * k[:, None] - kp
        if sigma == 1.0:
            u = np.log(np.maximum(c, 1e-12))
        else:
            u = (np.maximum(c, 1e-12) ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
        EV = V @ P.T
        idx = np.searchsorted(k, kp)
        idx = np.clip(idx, 0, len(k) - 1)
        cont = np.take_along_axis(EV, idx, axis=0)
        rhs = u + beta * cont
        resid = V - rhs
    else:
        resid = np.zeros_like(V, dtype=float)
    abs_resid = np.abs(resid)
    return {
        "residuals": resid,
        "abs_residuals": abs_resid,
        "summary": pd.DataFrame(
            [
                {"metric": "max_abs_bellman_residual", "value": float(np.nanmax(abs_resid))},
                {"metric": "mean_abs_bellman_residual", "value": float(np.nanmean(abs_resid))},
                {"metric": "p95_abs_bellman_residual", "value": float(np.nanpercentile(abs_resid, 95))},
            ]
        ),
    }


def euler_error_summary(result: dict, model_key: str, params: dict) -> pd.DataFrame:
    """Approximate log10 Euler-error panel where available."""
    if model_key not in ("model1", "model2"):
        return pd.DataFrame([{"metric": "euler_errors_available", "value": False}])
    P = np.asarray(params["P"], dtype=float)
    beta = float(params["beta"])
    sigma = float(params["sigma"])
    c = np.asarray(result["c_policy"], dtype=float)
    mu = np.maximum(c, 1e-12) ** (-sigma)
    if model_key == "model1":
        r = float(params["r"])
        rhs = beta * (1.0 + r) * (mu @ P.T)
    else:
        alpha = float(params["alpha"])
        delta = float(params["delta"])
        A = float(params.get("A", 1.0))
        k = np.asarray(result["policy_levels"], dtype=float)
        z = np.asarray(params["z_vals"], dtype=float)
        mpk_next = alpha * z[None, :] * A * np.maximum(k, 1e-12) ** (alpha - 1.0)
        rhs = beta * ((mu * (mpk_next + 1.0 - delta)) @ P.T)
    lhs = mu
    ratio = np.maximum(lhs / np.maximum(rhs, 1e-16), 1e-16)
    log10_err = np.log10(np.abs(ratio - 1.0) + 1e-16)
    abs_err = np.abs(log10_err)
    return pd.DataFrame(
        [
            {"metric": "euler_log10_abs_error_max", "value": float(np.nanmax(abs_err))},
            {"metric": "euler_log10_abs_error_mean", "value": float(np.nanmean(abs_err))},
            {"metric": "euler_log10_abs_error_p95", "value": float(np.nanpercentile(abs_err, 95))},
        ]
    )
