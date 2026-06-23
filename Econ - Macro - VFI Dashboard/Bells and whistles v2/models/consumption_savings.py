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
from config import MODEL1_DEFAULTS
from utils.grids import linear_grid
from utils.markov import validate_transition_matrix
from solvers.vfi import iterate
from solvers.advanced_vfi import iterate_with_howard


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
    method = str(p.get("method", "plain_vfi"))
    howard_steps = int(p.get("howard_steps", 10))

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

    def evaluate_policy(V: np.ndarray, policy_idx: np.ndarray):
        EV = V @ P.T
        cont = np.take_along_axis(EV, policy_idx, axis=0)
        u_pick = np.take_along_axis(u_all, policy_idx[:, None, :], axis=1)[:, 0, :]
        return u_pick + beta * cont

    # --- Run VFI -------------------------------------------------------
    V_init = np.zeros((n_a, n_y))
    if method == "howard_vfi":
        result = iterate_with_howard(
            V_init,
            bellman_operator=bellman,
            evaluate_policy_operator=evaluate_policy,
            howard_steps=howard_steps,
            max_iter=int(p.get("max_iter", 2000)),
            tol=float(p.get("tol", 1e-6)),
        )
    else:
        result = iterate(V_init, bellman, max_iter=int(p.get("max_iter", 2000)), tol=float(p.get("tol", 1e-6)))

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
