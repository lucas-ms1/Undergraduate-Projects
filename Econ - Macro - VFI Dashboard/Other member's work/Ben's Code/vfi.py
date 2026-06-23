"""
Generic Value Function Iteration (VFI) engine.
===============================================
Shared loop logic that each model's `solve()` can call.

Algorithm outline (to be implemented in Step 5):
    1. Initialise V to zeros (or a warm start).
    2. For each state (grid point x shock state):
       a. Enumerate feasible controls.
       b. Compute current utility + beta * E[V'] for each control.
       c. Take the max; store argmax index and value.
    3. Check convergence: max|V_new - V_old| < tol.
    4. Return V, policy_indices, diagnostics.

Convergence criterion from model_lockin_sheet.tex:
    max|V_{n+1} - V_n| < epsilon,  e.g. epsilon = 1e-6

Solver return format (all models):
    value_function, policy_indices, policy_levels,
    c_policy, grid, diagnostics
"""

import numpy as np
from config import VFI_DEFAULTS


def iterate(
    V_init: np.ndarray,
    bellman_operator,
    max_iter: int = VFI_DEFAULTS["max_iter"],
    tol: float = VFI_DEFAULTS["tol"],
) -> dict:
    """Run VFI until convergence or max_iter.

    Parameters
    ----------
    V_init : np.ndarray
        Initial guess for the value function.
    bellman_operator : callable
        Function(V) -> (V_new, policy_indices).
        Each model supplies its own operator.
    max_iter : int
    tol : float

    Returns
    -------
    dict
        'value_function': np.ndarray
        'policy_indices': np.ndarray
        'diagnostics': dict with 'iterations', 'final_error', 'converged'
    """
    V = np.asarray(V_init, dtype=float)
    for it in range(max_iter):
        V_new, policy_indices = bellman_operator(V)
        err = np.max(np.abs(V_new - V))
        V = V_new
        if err < tol:
            return {
                "value_function": V,
                "policy_indices": policy_indices,
                "diagnostics": {
                    "iterations": it + 1,
                    "final_error": float(err),
                    "converged": True,
                },
            }
    return {
        "value_function": V,
        "policy_indices": policy_indices,
        "diagnostics": {
            "iterations": max_iter,
            "final_error": float(err),
            "converged": False,
        },
    }
