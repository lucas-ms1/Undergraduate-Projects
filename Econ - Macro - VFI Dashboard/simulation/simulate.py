"""
Stochastic simulation using solved policy functions.
=====================================================
Given a solved model's policy arrays and a Markov shock path,
produce time-series paths for all relevant variables.

Minimum 100 periods (assignment requirement).
Uses utils.interpolation.interp_policy for off-grid evaluation.

See models/consumption_savings.py solve() for the solver return format.
"""

import numpy as np
from config import SIM_DEFAULTS
from utils.markov import simulate_shock_path, validate_transition_matrix
from utils.interpolation import interp_policy


def simulate_model(
    solver_result: dict,
    shock_vals: np.ndarray,
    P: np.ndarray,
    initial_state: float,
    initial_shock_idx: int = 0,
    T_sim: int = SIM_DEFAULTS["T_sim"],
    seed: int = SIM_DEFAULTS["seed"],
    model_name: str = "model1",
    model_params: dict | None = None,
) -> dict:
    """Simulate a solved model for T_sim periods.

    Parameters
    ----------
    solver_result : dict
        Output of a model's solve() function.
    shock_vals : np.ndarray
        e.g. y_vals, z_vals, or w_vals.
    P : np.ndarray, shape (2, 2)
        Markov transition matrix.
    initial_state : float
        Starting value on the state grid (e.g. initial assets or capital).
        Ignored for model3_labor_only.
    initial_shock_idx : int
        0 or 1 -- starting shock state.
    T_sim : int
    seed : int
    model_name : str
        One of 'model1', 'model2', 'model3', 'model3_labor_only'.
    model_params : dict, optional
        Original model parameters (needed for Model 2 derived variables).

    Returns
    -------
    dict of np.ndarray
        Time-series paths keyed by variable name.
    """
    validate_transition_matrix(P)
    T = int(T_sim)
    if T <= 0:
        raise ValueError("T_sim must be a positive integer.")

    shock_idx = simulate_shock_path(
        P=P, T=T, initial_idx=int(initial_shock_idx), seed=int(seed),
    )

    name = model_name.lower()
    if name == "model1":
        return _simulate_model1(solver_result, shock_vals, shock_idx, initial_state, T)
    if name == "model2":
        return _simulate_model2(solver_result, shock_vals, shock_idx, initial_state, T, model_params or {})
    if name in ("model3", "model3_assets"):
        return _simulate_model3_assets(solver_result, shock_vals, shock_idx, initial_state, T)
    if name == "model3_labor_only":
        return _simulate_model3_labor_only(solver_result, shock_vals, shock_idx, T)
    raise ValueError(f"Unknown model_name: {model_name!r}")


# ── Model 1: Consumption-Savings ──────────────────────────────────────────

def _simulate_model1(solver_result, shock_vals, shock_idx, initial_state, T):
    a_grid = solver_result["grid"]
    policy_levels = solver_result["policy_levels"]
    c_policy = solver_result["c_policy"]

    a_path = np.empty(T + 1)
    c_path = np.empty(T)
    y_path = np.empty(T)

    a_min, a_max = float(a_grid[0]), float(a_grid[-1])
    a_path[0] = np.clip(float(initial_state), a_min, a_max)

    for t in range(T):
        s = int(shock_idx[t])
        y_path[t] = float(shock_vals[s])
        a_t = a_path[t]
        c_path[t] = float(interp_policy(a_grid, c_policy[:, s], a_t))
        a_path[t + 1] = float(interp_policy(a_grid, policy_levels[:, s], a_t))

    return {"a": a_path[:-1], "c": c_path, "y": y_path, "shock_idx": shock_idx}


# ── Model 2: Robinson Crusoe ─────────────────────────────────────────────

def _simulate_model2(solver_result, shock_vals, shock_idx, initial_state, T, params):
    from models.robinson_crusoe import production

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

    k_min, k_max = float(k_grid[0]), float(k_grid[-1])
    k_path[0] = np.clip(float(initial_state), k_min, k_max)

    for t in range(T):
        s = int(shock_idx[t])
        z = float(shock_vals[s])
        k_t = k_path[t]
        y_path[t] = float(production(np.array([k_t]), z, alpha, A)[0])
        c_path[t] = float(interp_policy(k_grid, c_policy[:, s], k_t))
        kp = float(interp_policy(k_grid, policy_levels[:, s], k_t))
        k_path[t + 1] = kp
        inv_path[t] = kp - (1 - delta) * k_t

    return {
        "k": k_path[:-1], "c": c_path, "y": y_path,
        "investment": inv_path, "shock_idx": shock_idx,
    }


# ── Model 3: Labor Supply (asset-inclusive) ───────────────────────────────

def _simulate_model3_assets(solver_result, shock_vals, shock_idx, initial_state, T):
    a_grid = solver_result["grid"]["a_grid"]
    savings_policy = solver_result["policy_levels"]["savings"]
    labor_policy = solver_result["policy_levels"]["labor"]
    c_policy = solver_result["c_policy"]

    a_path = np.empty(T + 1)
    c_path = np.empty(T)
    labor_path = np.empty(T)
    earnings_path = np.empty(T)

    a_min, a_max = float(a_grid[0]), float(a_grid[-1])
    a_path[0] = np.clip(float(initial_state), a_min, a_max)

    for t in range(T):
        s = int(shock_idx[t])
        w = float(shock_vals[s])
        a_t = a_path[t]
        c_path[t] = float(interp_policy(a_grid, c_policy[:, s], a_t))
        a_path[t + 1] = float(interp_policy(a_grid, savings_policy[:, s], a_t))
        labor_path[t] = float(interp_policy(a_grid, labor_policy[:, s], a_t))
        earnings_path[t] = w * labor_path[t]

    return {
        "a": a_path[:-1], "c": c_path, "labor": labor_path,
        "earnings": earnings_path, "shock_idx": shock_idx,
    }


# ── Model 3: Labor Supply (labor-only) ───────────────────────────────────

def _simulate_model3_labor_only(solver_result, shock_vals, shock_idx, T):
    labor_policy = solver_result["policy_levels"]["labor"]   # (n_w,)
    c_policy = solver_result["c_policy"]                     # (n_w,)

    c_path = np.empty(T)
    labor_path = np.empty(T)
    earnings_path = np.empty(T)

    for t in range(T):
        s = int(shock_idx[t])
        w = float(shock_vals[s])
        labor_path[t] = float(labor_policy[s])
        c_path[t] = float(c_policy[s])
        earnings_path[t] = w * labor_path[t]

    return {
        "c": c_path, "labor": labor_path,
        "earnings": earnings_path, "shock_idx": shock_idx,
    }
