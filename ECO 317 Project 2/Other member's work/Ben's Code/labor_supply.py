"""
Model 3: Endogenous Labor Supply
=================================
Household receives time endowment T = 1 and chooses labor L_t and
next-period assets a', with a two-state Markov wage shock (asset-inclusive
version per course notes).

Bellman:
    V(a, w) = max_{a', L in [0,1]} { u(c) + v(1-L) + beta * sum_{w'} P(w'|w) V(a', w') }

Budget constraint:
    c = (1+r)*a + w*L - a',   c > 0

State:   (a, w)
Control: (a', L),  L in [0, 1],  leisure = 1 - L
Residual: c

Notation follows model_lockin_sheet.tex  Section 5 and the
cross-model coding conventions table.
"""

import numpy as np
from config import MODEL3_DEFAULTS
from utils.grids import linear_grid
from solvers.vfi import iterate


def utility(c: np.ndarray, leisure: np.ndarray,
            sigma: float, psi: float, nu: float) -> np.ndarray:
    """Separable consumption-leisure utility.
    u(c, 1-L) = c^{1-sigma}/(1-sigma) + psi * (1-L)^{1-nu}/(1-nu)
    Returns -inf where c <= 0 or leisure <= 0. Log cases for sigma==1 or nu==1.
    """
    out = np.full_like(c, -np.inf, dtype=float)
    pos_c = c > 0
    pos_l = leisure > 0
    valid = pos_c & pos_l
    if not np.any(valid):
        return out
    # Consumption part (CRRA / log)
    if abs(sigma - 1.0) < 1e-10:
        out[valid] = np.log(c[valid])
    else:
        out[valid] = (c[valid] ** (1 - sigma) - 1) / (1 - sigma)
    # Leisure part
    if abs(nu - 1.0) < 1e-10:
        out[valid] += psi * np.log(leisure[valid])
    else:
        out[valid] += psi * (leisure[valid] ** (1 - nu) - 1) / (1 - nu)
    return out


def build_grids(params: dict) -> dict:
    """Return dict with keys 'a_grid', 'labor_grid', 'w_vals' (asset-inclusive)."""
    p = params
    a_grid = linear_grid(p["a_min"], p["a_max"], p["n_a"])
    labor_grid = linear_grid(0.0, 1.0, p["n_L"])
    w_vals = np.atleast_1d(p["w_vals"])
    return {"a_grid": a_grid, "labor_grid": labor_grid, "w_vals": w_vals}


def feasibility_mask(a_grid: np.ndarray, w_vals: np.ndarray,
                     labor_grid: np.ndarray, r: float) -> np.ndarray:
    """Boolean array (n_a, 2, n_a, n_L): (i, j, ip, il) feasible iff c > 0.
    c = (1+r)*a_i + w_j*L_il - a'_ip. Also L in [0,1] and a' >= a_min are
    satisfied by grid construction.
    """
    n_a = len(a_grid)
    n_w = len(w_vals)
    n_L = len(labor_grid)
    mask = np.zeros((n_a, n_w, n_a, n_L), dtype=bool)
    for i in range(n_a):
        a = a_grid[i]
        for j in range(n_w):
            w = w_vals[j]
            # Cash on hand for each L: (1+r)*a + w*L
            coh = (1 + r) * a + w * labor_grid  # (n_L,)
            # c = coh[il] - a'_ip > 0  =>  a'_ip < coh[il]
            for il in range(n_L):
                max_ap = coh[il]
                feasible_ap = (a_grid >= 0) & (a_grid < max_ap - 1e-12)
                mask[i, j, :, il] = feasible_ap
    return mask


def solve(params: dict | None = None) -> dict:
    """Solve Model 3 via VFI (joint search over a' and L).

    Parameters
    ----------
    params : dict, optional
        Overrides for MODEL3_DEFAULTS.

    Returns
    -------
    dict with keys (standardised across all models):
        value_function, policy_indices (dict), policy_levels (dict),
        c_policy, grid (dict), diagnostics
    """
    p = {**MODEL3_DEFAULTS, **(params or {})}
    if not p.get("include_assets", True):
        raise NotImplementedError("Model 3 implemented only for asset-inclusive version (include_assets=True).")

    beta = p["beta"]
    sigma = p["sigma"]
    psi = p["psi"]
    nu = p["nu"]
    r = p["r"]
    P = np.asarray(p["P"])
    w_vals = np.atleast_1d(p["w_vals"])

    grids = build_grids(p)
    a_grid = grids["a_grid"]
    labor_grid = grids["labor_grid"]
    n_a = len(a_grid)
    n_w = len(w_vals)
    n_L = len(labor_grid)
    n_controls = n_a * n_L

    mask = feasibility_mask(a_grid, w_vals, labor_grid, r)

    def bellman_operator(V: np.ndarray):
        V_new = np.empty_like(V)
        # policy_idx: flat index g = a'_idx * n_L + L_idx
        policy_idx = np.empty((n_a, n_w), dtype=int)
        for i in range(n_a):
            a = a_grid[i]
            for j in range(n_w):
                w = w_vals[j]
                total = np.full(n_controls, -np.inf)
                for ip in range(n_a):
                    ap = a_grid[ip]
                    for il in range(n_L):
                        L = labor_grid[il]
                        g = ip * n_L + il
                        if not mask[i, j, ip, il]:
                            continue
                        c = (1 + r) * a + w * L - ap
                        leisure = 1.0 - L
                        u_val = utility(
                            np.array([c]), np.array([leisure]),
                            sigma, psi, nu
                        )[0]
                        ev = np.dot(P[j, :], V[ip, :])
                        total[g] = u_val + beta * ev
                best = np.argmax(total)
                policy_idx[i, j] = best
                V_new[i, j] = total[best]
        return V_new, policy_idx

    V_init = np.zeros((n_a, n_w))
    result = iterate(V_init, bellman_operator)

    policy_indices_flat = result["policy_indices"]
    # Decode flat index to (savings_idx, labor_idx)
    savings_idx = np.empty((n_a, n_w), dtype=int)
    labor_idx = np.empty((n_a, n_w), dtype=int)
    for i in range(n_a):
        for j in range(n_w):
            g = policy_indices_flat[i, j]
            savings_idx[i, j] = g // n_L
            labor_idx[i, j] = g % n_L

    policy_levels_savings = a_grid[savings_idx]
    policy_levels_labor = labor_grid[labor_idx]
    c_policy = np.empty((n_a, n_w))
    for i in range(n_a):
        for j in range(n_w):
            ap = policy_levels_savings[i, j]
            L = policy_levels_labor[i, j]
            a = a_grid[i]
            w = w_vals[j]
            c_policy[i, j] = (1 + r) * a + w * L - ap

    return {
        "value_function": result["value_function"],
        "policy_indices": {"savings": savings_idx, "labor": labor_idx},
        "policy_levels": {"savings": policy_levels_savings, "labor": policy_levels_labor},
        "c_policy": c_policy,
        "grid": {"a_grid": a_grid, "labor_grid": labor_grid},
        "diagnostics": result["diagnostics"],
    }


def labor_vs_wage_for_plot(solver_result: dict, w_vals: np.ndarray,
                           a_idx: int = 0):
    """Return (w_vals, labor_at_each_wage) for a static intuition graph: labor vs wage.
    Use at a fixed asset level (e.g. a_idx=0 for lowest assets). App can plot these.
    """
    labor = solver_result["policy_levels"]["labor"]
    return w_vals, labor[a_idx, :]
