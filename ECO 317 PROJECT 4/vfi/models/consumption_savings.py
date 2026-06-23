"""
Model 1: Stochastic Consumption-Savings
========================================
Household chooses next-period assets a' to maximise expected discounted
CRRA utility subject to a two-state Markov income shock.

Bellman:
    V(a, y) = max_{a'} { u(c) + beta * sum_{y'} P(y'|y) V(a', y') }

Budget constraint:
    c = (1+r)*a + y - a',   c > 0

State:   (a, y)
Control: a'
Residual: c (from budget constraint)

Notation follows model_lockin_sheet.tex  Section 3 and the
cross-model coding conventions table.
"""

import numpy as np
from vfi.config import MODEL1_DEFAULTS
from vfi.utils.grids import linear_grid
from vfi.utils.markov import validate_transition_matrix
from vfi.solvers.vfi import iterate


# ── Utility ────────────────────────────────────────────────────────────────

def utility(c: np.ndarray, sigma: float) -> np.ndarray:
    """CRRA utility.  Returns -inf where c <= 0."""
    c = np.asarray(c, dtype=float)
    out = np.full_like(c, -np.inf)
    pos = c > 0
    if sigma == 1.0:
        out[pos] = np.log(c[pos])
    else:
        out[pos] = (c[pos] ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    return out


# ── Grid / shock helpers ──────────────────────────────────────────────────

def build_grids(params: dict) -> dict:
    """Return dict with keys 'a_grid' and 'y_vals'."""
    a_grid = linear_grid(params["a_min"], params["a_max"], params["n_a"])
    y_vals = np.asarray(params["y_vals"], dtype=float)
    return {"a_grid": a_grid, "y_vals": y_vals}


def feasibility_mask(a_grid: np.ndarray, y_vals: np.ndarray,
                     r: float) -> np.ndarray:
    """Boolean array marking (a, y, a') triples where c > 0.

    Returns shape (n_a, n_a, n_shocks):
        axis 0 = current asset index,
        axis 1 = choice (next-period asset) index,
        axis 2 = current income-shock index.
    """
    n_a = len(a_grid)
    n_y = len(y_vals)
    # cash-on-hand for each (a_i, y_j): shape (n_a, 1, n_y)
    coh = ((1.0 + r) * a_grid[:, None, None]
           + y_vals[None, None, :])
    # consumption: c = coh - a'
    c = coh - a_grid[None, :, None]
    return c > 0


# ── Solve ─────────────────────────────────────────────────────────────────

def solve(params: dict | None = None) -> dict:
    """Solve Model 1 via VFI.

    Parameters
    ----------
    params : dict, optional
        Overrides for MODEL1_DEFAULTS.

    Returns
    -------
    dict with keys (standardised across all models):
        value_function : np.ndarray, shape (n_a, 2)
        policy_indices : np.ndarray, shape (n_a, 2)
        policy_levels  : np.ndarray, shape (n_a, 2)   -- optimal a'
        c_policy       : np.ndarray, shape (n_a, 2)
        grid           : np.ndarray, shape (n_a,)      -- the asset grid
        diagnostics    : dict with 'iterations', 'final_error', 'converged'
    """
    p = {**MODEL1_DEFAULTS, **(params or {})}

    beta  = p["beta"]
    sigma = p["sigma"]
    r     = p["r"]

    grids  = build_grids(p)
    a_grid = grids["a_grid"]
    y_vals = grids["y_vals"]
    P      = np.asarray(p["P"], dtype=float)
    validate_transition_matrix(P)

    n_a = len(a_grid)
    n_y = len(y_vals)

    # Pre-compute consumption for every (a_i, a'_k, y_j) triple
    # c[i, k, j] = (1+r)*a_i + y_j - a'_k
    c_all = ((1.0 + r) * a_grid[:, None, None]
             + y_vals[None, None, :]
             - a_grid[None, :, None])

    # Pre-compute utility (infeasible entries -> -inf)
    u_all = utility(c_all, sigma)          # (n_a, n_a, n_y)

    # --- Bellman operator (closure over pre-computed matrices) -----------
    def bellman(V: np.ndarray):
        # Expected continuation value conditioned on current shock j:
        # EV[k, j] = sum_{j'} P[j, j'] * V[k, j']
        EV = V @ P.T                       # (n_a, n_y)

        # RHS of the Bellman equation
        rhs = u_all + beta * EV[np.newaxis, :, :]   # (n_a, n_a, n_y)

        # Maximise over the choice dimension (axis 1)
        policy_idx = np.argmax(rhs, axis=1)          # (n_a, n_y)
        V_new = np.max(rhs, axis=1)                  # (n_a, n_y)
        return V_new, policy_idx

    # --- Run VFI -------------------------------------------------------
    V_init = np.zeros((n_a, n_y))
    result = iterate(V_init, bellman)

    # --- Extract level policies ----------------------------------------
    pol_idx = result["policy_indices"]                # (n_a, n_y)
    policy_levels = a_grid[pol_idx]                   # optimal a'

    # Consumption policy: c = (1+r)*a + y - a'
    c_policy = ((1.0 + r) * a_grid[:, None]
                + y_vals[None, :]
                - policy_levels)

    result["policy_levels"] = policy_levels
    result["c_policy"]      = c_policy
    result["grid"]          = a_grid

    return result
