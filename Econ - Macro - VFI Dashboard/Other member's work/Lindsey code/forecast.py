"""
Exact non-linear forecasting from solved policy functions.
==========================================================
User selects a sequence of future shocks (High/Low); the forecast
applies the already-solved policy rules deterministically to
generate a projected path.

Assignment requirement:
    "Use the exact policy functions from the VFI solver to generate
     a 10-period forecast of consumption and capital based on that
     specific shock path."
"""

import numpy as np
from utils.interpolation import interp_policy


def forecast_model(
    solver_result: dict,
    shock_vals: np.ndarray,
    shock_path: list[int],
    current_state: float,
    model_name: str = "model1",
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
        Starting value on the state grid.
    model_name : str

    Returns
    -------
    dict of np.ndarray
        Forecast paths keyed by variable name.
    """
    if model_name.lower() != "model1":
        raise NotImplementedError(
            "forecast_model is currently implemented only for Model 1."
        )

    # Unpack solver outputs for Model 1
    try:
        a_grid = solver_result["grid"]  # shape (n_a,)
        policy_levels = solver_result["policy_levels"]  # shape (n_a, n_shocks)
        c_policy = solver_result["c_policy"]  # shape (n_a, n_shocks)
    except KeyError as exc:
        raise KeyError(f"Missing key in solver_result: {exc}") from exc

    # Forecast horizon is implied by the user-specified shock path (configurable)
    T = len(shock_path)
    if T <= 0:
        raise ValueError("shock_path must contain at least one period.")

    # Preallocate arrays
    a_path = np.empty(T + 1, dtype=float)
    c_path = np.empty(T, dtype=float)
    y_path = np.empty(T, dtype=float)
    shocks = np.empty(T, dtype=int)

    # Initialise state, clipped to the asset grid
    a_min, a_max = float(a_grid[0]), float(a_grid[-1])
    a0 = float(current_state)
    a_path[0] = np.clip(a0, a_min, a_max)

    # Deterministic forward iteration using the chosen shock sequence
    for t in range(T):
        s = int(shock_path[t])
        shocks[t] = s

        # Income for current shock state
        y_path[t] = float(shock_vals[s])

        # Consumption and next-period assets from policy arrays
        a_t = a_path[t]
        c_path[t] = float(interp_policy(a_grid, c_policy[:, s], a_t))
        a_path[t + 1] = float(interp_policy(a_grid, policy_levels[:, s], a_t))

    # Align asset series length with other variables
    a_series = a_path[:-1]

    return {
        "a": a_series,
        "c": c_path,
        "y": y_path,
        "shock_idx": shocks,
    }
