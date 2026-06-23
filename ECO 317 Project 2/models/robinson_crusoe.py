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
from utils.markov import validate_transition_matrix
from solvers.vfi import iterate


def production(k: np.ndarray, z: float, alpha: float,
               A: float = 1.0) -> np.ndarray:
    """Cobb-Douglas production: Y = z * A * k^alpha."""
    return z * A * (k ** alpha)


def utility(c: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility. Returns -inf where c <= 0. Log case when sigma == 1."""
    c = np.asarray(c, dtype=float)
    out = np.full_like(c, -np.inf)
    pos = c > 0
    if np.any(pos):
        if abs(sigma - 1.0) < 1e-10:
            out[pos] = np.log(c[pos])
        else:
            out[pos] = (c[pos] ** (1 - sigma) - 1) / (1 - sigma)
    return out


def build_grids(params: dict) -> dict:
    """Return dict with keys 'k_grid' and 'z_vals'."""
    k_grid = linear_grid(params["k_min"], params["k_max"], params["n_k"])
    z_vals = np.atleast_1d(params["z_vals"])
    return {"k_grid": k_grid, "z_vals": z_vals}


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
        policy_levels  : np.ndarray, shape (n_k, 2)   -- optimal k'
        c_policy       : np.ndarray, shape (n_k, 2)
        grid           : np.ndarray, shape (n_k,)      -- the capital grid
        diagnostics    : dict with 'iterations', 'final_error', 'converged'
    """
    p = {**MODEL2_DEFAULTS, **(params or {})}
    beta = p["beta"]
    sigma = p["sigma"]
    alpha = p["alpha"]
    delta = p["delta"]
    A = p["A"]
    P = np.asarray(p["P"], dtype=float)
    validate_transition_matrix(P)
    z_vals = np.atleast_1d(p["z_vals"])

    grids = build_grids(p)
    k_grid = grids["k_grid"]
    n_k = len(k_grid)
    n_z = len(z_vals)

    # Pre-compute cash-on-hand for every (k_i, z_j)
    y_all = production(k_grid[:, None], z_vals[None, :], alpha, A)  # (n_k, n_z)
    coh_all = y_all + (1 - delta) * k_grid[:, None]                # (n_k, n_z)

    # Pre-compute consumption and utility for every (k_i, z_j, k'_ip)
    c_all = coh_all[:, :, None] - k_grid[None, None, :]  # (n_k, n_z, n_k)
    u_all = utility(c_all, sigma)                         # -inf where c <= 0

    def bellman(V: np.ndarray):
        # EV[ip, j] = sum_{j'} P[j, j'] * V[ip, j']
        EV = V @ P.T                                     # (n_k, n_z)
        # rhs[i, j, ip] = u_all[i, j, ip] + beta * EV[ip, j]
        rhs = u_all + beta * EV.T[np.newaxis, :, :]      # (n_k, n_z, n_k)
        policy_idx = np.argmax(rhs, axis=2)               # (n_k, n_z)
        V_new = np.max(rhs, axis=2)                       # (n_k, n_z)
        return V_new, policy_idx

    V_init = np.zeros((n_k, n_z))
    result = iterate(V_init, bellman)

    policy_indices = result["policy_indices"]
    policy_levels = k_grid[policy_indices]
    c_policy = coh_all - policy_levels

    result["policy_levels"] = policy_levels
    result["c_policy"] = c_policy
    result["grid"] = k_grid

    return result
