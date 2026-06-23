"""
Generic Value Function Iteration (VFI) engine.
===============================================
Shared loop logic that each model's `solve()` can call.

Algorithm:
    1. Initialise V (caller provides V_init).
    2. Apply bellman_operator(V) -> (V_new, policy_indices).
    3. Check convergence: max|V_new - V_old| < tol.
    4. Return V, policy_indices, diagnostics.

Convergence criterion:
    max|V_{n+1} - V_n| < epsilon,  e.g. epsilon = 1e-6

Solver return format (all models):
    value_function, policy_indices, diagnostics
"""

import numpy as np
from time import perf_counter
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
    V = V_init.copy()
    policy_indices = None
    error_path = []
    t0 = perf_counter()

    for it in range(1, max_iter + 1):
        V_new, policy_indices = bellman_operator(V)
        error = float(np.max(np.abs(V_new - V)))
        error_path.append(error)
        V = V_new

        if error < tol:
            runtime = float(perf_counter() - t0)
            return {
                "value_function": V,
                "policy_indices": policy_indices,
                "diagnostics": {
                    "iterations": it,
                    "final_error": error,
                    "converged": True,
                    "error_path": error_path,
                    "runtime_seconds": runtime,
                },
            }

    runtime = float(perf_counter() - t0)
    return {
        "value_function": V,
        "policy_indices": policy_indices,
        "diagnostics": {
            "iterations": max_iter,
            "final_error": error,
            "converged": False,
            "error_path": error_path,
            "runtime_seconds": runtime,
        },
    }
