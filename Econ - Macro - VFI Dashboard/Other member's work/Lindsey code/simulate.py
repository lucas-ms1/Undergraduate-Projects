"""
Stochastic simulation using solved policy functions.
=====================================================
Given a solved model's policy arrays and a Markov shock path,
produce time-series paths for all relevant variables.

Minimum 100 periods (assignment requirement).
Uses numpy.interp for off-grid evaluation during simulation.

Notation follows model_lockin_sheet.tex cross-model conventions:
    shock_idx, T_sim, seed, policy_level, c_policy, etc.
"""

import numpy as np
from config import SIM_DEFAULTS
from utils.interpolation import interp_policy
from utils.markov import simulate_shock_path, validate_transition_matrix


def simulate_model(
    solver_result: dict,
    shock_vals: np.ndarray,
    P: np.ndarray,
    initial_state: float,
    initial_shock_idx: int = 0,
    T_sim: int = SIM_DEFAULTS["T_sim"],
    seed: int = SIM_DEFAULTS["seed"],
    model_name: str = "model1",
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
    initial_shock_idx : int
        0 or 1 – starting shock state.
    T_sim : int
    seed : int
    model_name : str
        Used to determine which variables to track.

    Returns
    -------
    dict of np.ndarray
        Time-series paths keyed by variable name.
    """
    if model_name.lower() != "model1":
        raise NotImplementedError(
            "simulate_model is currently implemented only for Model 1."
        )

    # Validate transition matrix (basic checks)
    if not validate_transition_matrix(P):
        raise ValueError("Invalid transition matrix P passed to simulate_model.")

    # Unpack solver outputs for Model 1
    try:
        a_grid = solver_result["grid"]  # shape (n_a,)
        policy_levels = solver_result["policy_levels"]  # shape (n_a, n_shocks)
        c_policy = solver_result["c_policy"]  # shape (n_a, n_shocks)
    except KeyError as exc:
        raise KeyError(f"Missing key in solver_result: {exc}") from exc

    T = int(T_sim)
    if T <= 0:
        raise ValueError("T_sim must be a positive integer.")

    # Draw Markov shock path
    shock_idx = simulate_shock_path(
        P=P,
        T=T,
        initial_idx=int(initial_shock_idx),
        seed=int(seed),
    )

    # Preallocate arrays
    a_path = np.empty(T + 1, dtype=float)
    c_path = np.empty(T, dtype=float)
    y_path = np.empty(T, dtype=float)

    # Initialise state, clipped to the asset grid
    a_min, a_max = float(a_grid[0]), float(a_grid[-1])
    a0 = float(initial_state)
    a_path[0] = np.clip(a0, a_min, a_max)

    # Simulate forward using interpolated policy rules
    for t in range(T):
        s = int(shock_idx[t])

        # Income for current shock state
        y_path[t] = float(shock_vals[s])

        # Consumption and next-period assets from policy arrays
        a_t = a_path[t]
        c_path[t] = float(interp_policy(a_grid, c_policy[:, s], a_t))
        a_path[t + 1] = float(interp_policy(a_grid, policy_levels[:, s], a_t))

    # Drop terminal asset level so all series share the same length T
    a_series = a_path[:-1]

    return {
        "a": a_series,
        "c": c_path,
        "y": y_path,
        "shock_idx": shock_idx,
    }
