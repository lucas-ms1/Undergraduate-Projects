"""Alternative EGM-style solver path for Model 1."""

from __future__ import annotations

import numpy as np
from config import MODEL1_DEFAULTS
from utils.grids import linear_grid
from utils.markov import validate_transition_matrix


def _u_prime(c: np.ndarray, sigma: float) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if sigma == 1.0:
        return 1.0 / np.maximum(c, 1e-10)
    return np.maximum(c, 1e-10) ** (-sigma)


def _u_prime_inv(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sigma == 1.0:
        return 1.0 / np.maximum(x, 1e-10)
    return np.maximum(x, 1e-10) ** (-1.0 / sigma)


def solve_egm_model1(params: dict | None = None, max_iter: int = 500, tol: float = 1e-6) -> dict:
    """Solve model1 with an endogenous-grid style update.

    This is intentionally lightweight and serves as an advanced-mode method comparison path.
    """
    p = {**MODEL1_DEFAULTS, **(params or {})}
    beta = float(p["beta"])
    sigma = float(p["sigma"])
    r = float(p["r"])
    a_grid = linear_grid(p["a_min"], p["a_max"], p["n_a"])
    y_vals = np.asarray(p["y_vals"], dtype=float)
    P = np.asarray(p["P"], dtype=float)
    validate_transition_matrix(P)

    n_a = a_grid.size
    n_y = y_vals.size

    # Initial guess: consume all cash-on-hand.
    c_policy = np.maximum((1.0 + r) * a_grid[:, None] + y_vals[None, :] - a_grid[0], 1e-8)
    error_path = []

    for _ in range(max_iter):
        c_new = np.empty_like(c_policy)
        a_prime_policy = np.empty_like(c_policy)

        for j in range(n_y):
            Emu_next = np.zeros(n_a)
            for jp in range(n_y):
                Emu_next += P[j, jp] * _u_prime(c_policy[:, jp], sigma)
            mu_now = beta * (1.0 + r) * Emu_next
            c_now = _u_prime_inv(mu_now, sigma)

            # Endogenous current-asset grid implied by each a' node
            a_endo = (c_now + a_grid - y_vals[j]) / (1.0 + r)
            a_endo = np.maximum.accumulate(a_endo)

            # Interpolate back to exogenous grid.
            a_prime = np.interp(a_grid, a_endo, a_grid, left=a_grid[0], right=a_grid[-1])
            c_interp = (1.0 + r) * a_grid + y_vals[j] - a_prime
            c_new[:, j] = np.maximum(c_interp, 1e-8)
            a_prime_policy[:, j] = a_prime

        err = float(np.max(np.abs(c_new - c_policy)))
        error_path.append(err)
        c_policy = c_new
        if err < tol:
            break

    # Build an approximate value function by one-step utility + discounted continuation utility.
    util = np.zeros_like(c_policy)
    if sigma == 1.0:
        util = np.log(np.maximum(c_policy, 1e-12))
    else:
        util = (np.maximum(c_policy, 1e-12) ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    V = util / max(1e-8, 1.0 - beta)

    pol_idx = np.searchsorted(a_grid, a_prime_policy)
    pol_idx = np.clip(pol_idx, 0, n_a - 1)

    return {
        "value_function": V,
        "policy_indices": pol_idx.astype(int),
        "policy_levels": a_prime_policy,
        "c_policy": c_policy,
        "grid": a_grid,
        "diagnostics": {
            "iterations": len(error_path),
            "final_error": float(error_path[-1]) if error_path else np.nan,
            "converged": bool(error_path and error_path[-1] < tol),
            "error_path": error_path,
            "method": "egm_model1",
        },
    }
