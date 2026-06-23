"""
Model 3: Endogenous Labor Supply
=================================
Household receives time endowment T = 1 and chooses labor L_t and
(optionally) next-period assets a', with a two-state Markov wage shock.

Asset-inclusive version (include_assets=True):
    V(a, w) = max_{a', L in [0,1]} { u(c) + v(1-L)
              + beta * sum_{w'} P(w'|w) V(a', w') }
    c = (1+r)*a + w*L - a',   c > 0

Labor-only version (include_assets=False):
    V(w) = max_{L in [0,1]} { u(c) + v(1-L)
            + beta * sum_{w'} P(w'|w) V(w') }
    c = w*L

State:   (a, w) or (w) depending on include_assets
Control: (a', L) or (L)
Residual: c
Derived: leisure = 1 - L

Notation follows model_lockin_sheet.tex  Section 5 and the
cross-model coding conventions table.
"""

import numpy as np
from vfi.config import MODEL3_DEFAULTS
from vfi.utils.grids import linear_grid
from vfi.utils.markov import validate_transition_matrix
from vfi.solvers.vfi import iterate


def utility(c: np.ndarray, leisure: np.ndarray,
            sigma: float, psi: float, nu: float) -> np.ndarray:
    """Separable consumption-leisure utility.

    u(c, 1-L) = c^{1-sigma}/(1-sigma) + psi * (1-L)^{1-nu}/(1-nu)
    Returns -inf where c <= 0 or leisure <= 0.  Log cases for sigma==1 or nu==1.
    """
    c = np.asarray(c, dtype=float)
    leisure = np.asarray(leisure, dtype=float)
    out = np.full_like(c, -np.inf)
    valid = (c > 0) & (leisure > 0)
    if not np.any(valid):
        return out
    if abs(sigma - 1.0) < 1e-10:
        out[valid] = np.log(c[valid])
    else:
        out[valid] = (c[valid] ** (1 - sigma) - 1) / (1 - sigma)
    if abs(nu - 1.0) < 1e-10:
        out[valid] += psi * np.log(leisure[valid])
    else:
        out[valid] += psi * (leisure[valid] ** (1 - nu) - 1) / (1 - nu)
    return out


def build_grids(params: dict) -> dict:
    """Return dict with keys 'a_grid' (if applicable), 'labor_grid', 'w_vals'."""
    labor_grid = linear_grid(0.0, 1.0, params["n_L"])
    w_vals = np.atleast_1d(params["w_vals"])
    out = {"labor_grid": labor_grid, "w_vals": w_vals}
    if params.get("include_assets", True):
        out["a_grid"] = linear_grid(params["a_min"], params["a_max"], params["n_a"])
    return out


# ── Asset-inclusive solver ────────────────────────────────────────────────

def _solve_with_assets(p: dict) -> dict:
    beta = p["beta"]
    sigma = p["sigma"]
    psi = p["psi"]
    nu = p["nu"]
    r = p["r"]
    P = np.asarray(p["P"], dtype=float)
    validate_transition_matrix(P)
    w_vals = np.atleast_1d(p["w_vals"])

    grids = build_grids(p)
    a_grid = grids["a_grid"]
    labor_grid = grids["labor_grid"]
    n_a = len(a_grid)
    n_w = len(w_vals)
    n_L = len(labor_grid)
    n_controls = n_a * n_L

    leisure_grid = 1.0 - labor_grid  # (n_L,)

    # Pre-compute flow utility for each current wage state. This keeps the
    # economics unchanged while removing the slow Python loop over current
    # asset states inside the Bellman step.
    #
    # For a fixed wage state j:
    #   c[i, ip, il] = (1+r)a_i + w_j L_il - a'_ip
    # where:
    #   i   = current asset index
    #   ip  = next-period asset choice index
    #   il  = labor choice index
    resources = ((1.0 + r) * a_grid[:, np.newaxis, np.newaxis]
                 - a_grid[np.newaxis, :, np.newaxis])  # (n_a, n_a, 1)
    leisure_all = np.broadcast_to(
        leisure_grid[np.newaxis, np.newaxis, :], (n_a, n_a, n_L)
    )
    flow_utility_by_wage = []
    for w in w_vals:
        c = resources + w * labor_grid[np.newaxis, np.newaxis, :]
        flow_utility_by_wage.append(
            utility(c, leisure_all, sigma, psi, nu)
        )  # each entry has shape (n_a, n_a, n_L)

    def bellman(V: np.ndarray):
        EV = V @ P.T  # (n_a, n_w)
        V_new = np.empty_like(V)
        policy_idx = np.empty((n_a, n_w), dtype=int)

        for j in range(n_w):
            # continuation[ip] = E[V(a'_ip, w') | current wage state j]
            continuation = beta * EV[:, j][np.newaxis, :, np.newaxis]
            total = flow_utility_by_wage[j] + continuation  # (n_a, n_a, n_L)
            flat = total.reshape(n_a, n_controls)           # choices = (a', L)
            policy_idx[:, j] = np.argmax(flat, axis=1)
            V_new[:, j] = np.max(flat, axis=1)

        return V_new, policy_idx

    V_init = np.zeros((n_a, n_w))
    result = iterate(V_init, bellman)

    policy_flat = result["policy_indices"]  # (n_a, n_w)
    savings_idx = policy_flat // n_L
    labor_idx = policy_flat % n_L

    policy_levels_savings = a_grid[savings_idx]
    policy_levels_labor = labor_grid[labor_idx]
    c_policy = ((1.0 + r) * a_grid[:, np.newaxis]
                + w_vals[np.newaxis, :] * policy_levels_labor
                - policy_levels_savings)

    return {
        "value_function": result["value_function"],
        "policy_indices": {"savings": savings_idx, "labor": labor_idx},
        "policy_levels": {"savings": policy_levels_savings,
                          "labor": policy_levels_labor},
        "c_policy": c_policy,
        "grid": {"a_grid": a_grid, "labor_grid": labor_grid},
        "diagnostics": result["diagnostics"],
    }


# ── Labor-only solver ────────────────────────────────────────────────────

def _solve_labor_only(p: dict) -> dict:
    """Solve the static-asset variant: V(w) = max_L {u(wL,1-L)} + beta E[V]."""
    beta = p["beta"]
    sigma = p["sigma"]
    psi = p["psi"]
    nu = p["nu"]
    P = np.asarray(p["P"], dtype=float)
    validate_transition_matrix(P)
    w_vals = np.atleast_1d(p["w_vals"])

    labor_grid = linear_grid(0.0, 1.0, p["n_L"])
    n_w = len(w_vals)
    n_L = len(labor_grid)
    leisure_grid = 1.0 - labor_grid

    # Pre-compute utility for every (w_j, L_il)
    # c[j, il] = w_j * L_il
    c_all = w_vals[:, np.newaxis] * labor_grid[np.newaxis, :]      # (n_w, n_L)
    leis_all = np.broadcast_to(leisure_grid[np.newaxis, :], c_all.shape)
    u_all = utility(c_all, leis_all, sigma, psi, nu)               # (n_w, n_L)

    def bellman(V: np.ndarray):
        EV = P @ V                              # (n_w,)
        # rhs[j, il] = u_all[j, il] + beta * EV[j]
        rhs = u_all + beta * EV[:, np.newaxis]  # (n_w, n_L)
        policy_idx = np.argmax(rhs, axis=1)     # (n_w,)
        V_new = np.max(rhs, axis=1)             # (n_w,)
        return V_new, policy_idx

    V_init = np.zeros(n_w)
    result = iterate(V_init, bellman)

    policy_idx = result["policy_indices"]
    policy_levels_labor = labor_grid[policy_idx]
    c_policy = w_vals * policy_levels_labor

    return {
        "value_function": result["value_function"],
        "policy_indices": {"labor": policy_idx},
        "policy_levels": {"labor": policy_levels_labor},
        "c_policy": c_policy,
        "grid": {"labor_grid": labor_grid},
        "diagnostics": result["diagnostics"],
    }


# ── Public entry point ───────────────────────────────────────────────────

def solve(params: dict | None = None) -> dict:
    """Solve Model 3 via VFI.

    Parameters
    ----------
    params : dict, optional
        Overrides for MODEL3_DEFAULTS.  Set ``include_assets=False``
        for the labour-only variant (no savings decision).

    Returns
    -------
    dict with keys (standardised across all models):
        value_function : np.ndarray
            shape (n_a, 2) if assets included, else (2,)
        policy_indices : dict
        policy_levels  : dict
            'labor': np.ndarray -- optimal L
            'savings': np.ndarray -- optimal a' (asset-inclusive only)
        c_policy       : np.ndarray
        grid           : dict
            'a_grid': np.ndarray (asset-inclusive only)
            'labor_grid': np.ndarray
        diagnostics    : dict with 'iterations', 'final_error', 'converged'
    """
    p = {**MODEL3_DEFAULTS, **(params or {})}
    if p.get("include_assets", True):
        return _solve_with_assets(p)
    return _solve_labor_only(p)


def labor_vs_wage_for_plot(solver_result: dict, w_vals: np.ndarray,
                           a_idx: int | None = 0):
    """Return (w_vals, labor_at_each_wage) for a static intuition graph.

    For the asset-inclusive case, slices at a fixed asset level ``a_idx``.
    For the labor-only case ``a_idx`` is ignored.
    """
    labor = solver_result["policy_levels"]["labor"]
    if labor.ndim == 1:
        return w_vals, labor
    return w_vals, labor[a_idx, :]
