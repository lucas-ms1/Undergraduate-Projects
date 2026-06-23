"""
Model 2: Stochastic Robinson Crusoe Economy
============================================
Household accumulates productive capital instead of a risk-free asset.
Output depends on capital and a two-state TFP shock.

*** TIMING (course notes): Tomorrow's output is Y_{t+1} = F(K_{t+1}).
    So today: enter with k_t, observe z_t; choose k_{t+1}; consumption
    is c_t = z_t*F(k_t) + (1-delta)*k_t - k_{t+1}. ***

Bellman:
    V(k, z) = max_{k'} { u(c) + beta * sum_{z'} P(z'|z) V(k', z') }

Resource constraint:
    c = z*F(k) + (1 - delta)*k - k',   c > 0

State:   (k, z)
Control: k'
Residual: c (from resource constraint)
Derived: Y = z*F(k),  I = k' - (1-delta)*k

Notation follows model_lockin_sheet.tex  Section 4 and the
cross-model coding conventions table.
"""

import numpy as np
from config import MODEL2_DEFAULTS
from utils.grids import linear_grid
from solvers.vfi import iterate


def production(k: np.ndarray, z: float, alpha: float,
               A: float = 1.0) -> np.ndarray:
    """Cobb-Douglas production: Y = z * A * k^alpha."""
    return z * A * (k ** alpha)


def utility(c: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility. Returns -inf where c <= 0. Log case when sigma == 1."""
    out = np.empty_like(c)
    out[:] = -np.inf
    pos = c > 0
    if np.any(pos):
        if abs(sigma - 1.0) < 1e-10:
            out[pos] = np.log(c[pos])
        else:
            out[pos] = (c[pos] ** (1 - sigma) - 1) / (1 - sigma)
    return out


def build_grids(params: dict) -> dict:
    """Return dict with keys 'k_grid' and 'z_vals'."""
    p = params
    k_grid = linear_grid(p["k_min"], p["k_max"], p["n_k"])
    z_vals = np.atleast_1d(p["z_vals"])
    return {"k_grid": k_grid, "z_vals": z_vals}


def feasibility_mask(k_grid: np.ndarray, z_vals: np.ndarray,
                     alpha: float, delta: float,
                     A: float = 1.0) -> np.ndarray:
    """Boolean array of shape (n_k, 2, n_k): (i, j, ip) feasible iff c > 0.
    c = z*F(k_i) + (1-delta)*k_i - k' with k' = k_grid[ip], z = z_vals[j].
    """
    n_k = len(k_grid)
    n_z = len(z_vals)
    mask = np.zeros((n_k, n_z, n_k), dtype=bool)
    for i in range(n_k):
        k = k_grid[i]
        for j in range(n_z):
            z = z_vals[j]
            y = production(k, z, alpha, A)
            max_kp = y + (1 - delta) * k
            # Strictly positive consumption: k' < max_kp (use small epsilon)
            eps = 1e-12
            feasible = (k_grid >= 0) & (k_grid < max_kp - eps)
            mask[i, j, :] = feasible
    return mask


def solve(params: dict | None = None) -> dict:
    """Solve Model 2 via VFI.

    Parameters
    ----------
    params : dict, optional
        Overrides for MODEL2_DEFAULTS.

    Returns
    -------
    dict with keys (standardised across all models):
        value_function : np.ndarray, shape (n_k, 2)
        policy_indices : np.ndarray, shape (n_k, 2)
        policy_levels  : np.ndarray, shape (n_k, 2)   – optimal k'
        c_policy       : np.ndarray, shape (n_k, 2)
        grid           : np.ndarray, shape (n_k,)      – the capital grid
        diagnostics    : dict with 'iterations', 'final_error', 'converged'
    """
    p = {**MODEL2_DEFAULTS, **(params or {})}
    beta = p["beta"]
    sigma = p["sigma"]
    alpha = p["alpha"]
    delta = p["delta"]
    A = p["A"]
    P = np.asarray(p["P"])
    z_vals = np.atleast_1d(p["z_vals"])

    grids = build_grids(p)
    k_grid = grids["k_grid"]
    n_k = len(k_grid)
    n_z = len(z_vals)

    mask = feasibility_mask(k_grid, z_vals, alpha, delta, A)

    def bellman_operator(V: np.ndarray):
        V_new = np.empty_like(V)
        policy_idx = np.empty((n_k, n_z), dtype=int)
        for i in range(n_k):
            k = k_grid[i]
            for j in range(n_z):
                z = z_vals[j]
                c_vals = z * A * (k ** alpha) + (1 - delta) * k - k_grid
                u_vals = utility(c_vals, sigma)
                # Continuation at grid points: E[V(k', z')] = sum_{z'} P(z'|z) V(k', z')
                ev = V @ P[j, :]  # shape (n_k,)
                total = u_vals + beta * ev
                total[~mask[i, j, :]] = -np.inf
                best = np.argmax(total)
                policy_idx[i, j] = best
                V_new[i, j] = total[best]
        return V_new, policy_idx

    V_init = np.zeros((n_k, n_z))
    result = iterate(V_init, bellman_operator)

    policy_indices = result["policy_indices"]
    policy_levels = np.empty((n_k, n_z))
    c_policy = np.empty((n_k, n_z))
    for i in range(n_k):
        for j in range(n_z):
            ip = policy_indices[i, j]
            kp = k_grid[ip]
            policy_levels[i, j] = kp
            k = k_grid[i]
            z = z_vals[j]
            c_policy[i, j] = z * A * (k ** alpha) + (1 - delta) * k - kp

    return {
        "value_function": result["value_function"],
        "policy_indices": policy_indices,
        "policy_levels": policy_levels,
        "c_policy": c_policy,
        "grid": k_grid,
        "diagnostics": result["diagnostics"],
    }
