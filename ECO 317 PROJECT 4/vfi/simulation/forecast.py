"""
Exact non-linear forecasting from solved policy functions.
==========================================================
User selects a sequence of future shocks (High/Low); the forecast
applies the already-solved policy rules deterministically to
generate a projected path.

See models/consumption_savings.py solve() for the solver return format.
"""

import numpy as np
from vfi.utils.interpolation import interp_policy


def forecast_model(
    solver_result: dict,
    shock_vals: np.ndarray,
    shock_path: list[int],
    current_state: float,
    model_name: str = "model1",
    model_params: dict | None = None,
) -> dict:
    """Generate a deterministic forecast from solved policies.

    Parameters
    ----------
    solver_result : dict
        Output of a model's solve() function.
    shock_vals : np.ndarray
        e.g. y_vals, z_vals, or w_vals.
    shock_path : list[int]
        User-chosen sequence of shock indices (0 or 1) for each
        future period.  Length determines forecast horizon.
    current_state : float
        Starting value on the state grid.  Ignored for model3_labor_only.
    model_name : str
        One of 'model1', 'model2', 'model3', 'model3_labor_only'.
    model_params : dict, optional
        Original model parameters (needed for Model 2 derived variables).

    Returns
    -------
    dict of np.ndarray
        Forecast paths keyed by variable name.
    """
    T = len(shock_path)
    if T <= 0:
        raise ValueError("shock_path must contain at least one period.")

    name = model_name.lower()
    if name == "model1":
        return _forecast_model1(solver_result, shock_vals, shock_path, current_state, T)
    if name == "model2":
        return _forecast_model2(solver_result, shock_vals, shock_path, current_state, T, model_params or {})
    if name in ("model3", "model3_assets"):
        return _forecast_model3_assets(solver_result, shock_vals, shock_path, current_state, T)
    if name == "model3_labor_only":
        return _forecast_model3_labor_only(solver_result, shock_vals, shock_path, T)
    raise ValueError(f"Unknown model_name: {model_name!r}")


# ── Model 1 ──────────────────────────────────────────────────────────────

def _forecast_model1(solver_result, shock_vals, shock_path, current_state, T):
    a_grid = solver_result["grid"]
    policy_levels = solver_result["policy_levels"]
    c_policy = solver_result["c_policy"]

    a_path = np.empty(T + 1)
    c_path = np.empty(T)
    y_path = np.empty(T)
    shocks = np.empty(T, dtype=int)

    a_min, a_max = float(a_grid[0]), float(a_grid[-1])
    a_path[0] = np.clip(float(current_state), a_min, a_max)

    for t in range(T):
        s = int(shock_path[t])
        shocks[t] = s
        y_path[t] = float(shock_vals[s])
        a_t = a_path[t]
        c_path[t] = float(interp_policy(a_grid, c_policy[:, s], a_t))
        a_path[t + 1] = float(interp_policy(a_grid, policy_levels[:, s], a_t))

    return {"a": a_path[:-1], "c": c_path, "y": y_path, "shock_idx": shocks}


# ── Model 2 ──────────────────────────────────────────────────────────────

def _forecast_model2(solver_result, shock_vals, shock_path, current_state, T, params):
    from vfi.models.robinson_crusoe import production

    k_grid = solver_result["grid"]
    policy_levels = solver_result["policy_levels"]
    c_policy = solver_result["c_policy"]
    alpha = params.get("alpha", 0.36)
    delta = params.get("delta", 0.10)
    A = params.get("A", 1.0)

    k_path = np.empty(T + 1)
    c_path = np.empty(T)
    y_path = np.empty(T)
    inv_path = np.empty(T)
    shocks = np.empty(T, dtype=int)

    k_min, k_max = float(k_grid[0]), float(k_grid[-1])
    k_path[0] = np.clip(float(current_state), k_min, k_max)

    for t in range(T):
        s = int(shock_path[t])
        shocks[t] = s
        z = float(shock_vals[s])
        k_t = k_path[t]
        y_path[t] = float(production(np.array([k_t]), z, alpha, A)[0])
        c_path[t] = float(interp_policy(k_grid, c_policy[:, s], k_t))
        kp = float(interp_policy(k_grid, policy_levels[:, s], k_t))
        k_path[t + 1] = kp
        inv_path[t] = kp - (1 - delta) * k_t

    return {
        "k": k_path[:-1], "c": c_path, "y": y_path,
        "investment": inv_path, "shock_idx": shocks,
    }


# ── Model 3 (asset-inclusive) ─────────────────────────────────────────────

def _forecast_model3_assets(solver_result, shock_vals, shock_path, current_state, T):
    a_grid = solver_result["grid"]["a_grid"]
    savings_policy = solver_result["policy_levels"]["savings"]
    labor_policy = solver_result["policy_levels"]["labor"]
    c_policy = solver_result["c_policy"]

    a_path = np.empty(T + 1)
    c_path = np.empty(T)
    labor_path = np.empty(T)
    earnings_path = np.empty(T)
    shocks = np.empty(T, dtype=int)

    a_min, a_max = float(a_grid[0]), float(a_grid[-1])
    a_path[0] = np.clip(float(current_state), a_min, a_max)

    for t in range(T):
        s = int(shock_path[t])
        shocks[t] = s
        w = float(shock_vals[s])
        a_t = a_path[t]
        c_path[t] = float(interp_policy(a_grid, c_policy[:, s], a_t))
        a_path[t + 1] = float(interp_policy(a_grid, savings_policy[:, s], a_t))
        labor_path[t] = float(interp_policy(a_grid, labor_policy[:, s], a_t))
        earnings_path[t] = w * labor_path[t]

    return {
        "a": a_path[:-1], "c": c_path, "labor": labor_path,
        "earnings": earnings_path, "shock_idx": shocks,
    }


# ── Model 3 (labor-only) ─────────────────────────────────────────────────

def _forecast_model3_labor_only(solver_result, shock_vals, shock_path, T):
    labor_policy = solver_result["policy_levels"]["labor"]
    c_policy = solver_result["c_policy"]

    c_path = np.empty(T)
    labor_path = np.empty(T)
    earnings_path = np.empty(T)
    shocks = np.empty(T, dtype=int)

    for t in range(T):
        s = int(shock_path[t])
        shocks[t] = s
        w = float(shock_vals[s])
        labor_path[t] = float(labor_policy[s])
        c_path[t] = float(c_policy[s])
        earnings_path[t] = w * labor_path[t]

    return {
        "c": c_path, "labor": labor_path,
        "earnings": earnings_path, "shock_idx": shocks,
    }
