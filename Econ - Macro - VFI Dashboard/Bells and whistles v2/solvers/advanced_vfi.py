"""Advanced VFI utilities (Howard-style variants and runtime diagnostics)."""

from __future__ import annotations

from time import perf_counter
import numpy as np


def iterate_with_howard(
    V_init: np.ndarray,
    bellman_operator,
    evaluate_policy_operator,
    howard_steps: int = 10,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> dict:
    """Run VFI with optional Howard policy improvement steps.

    The model must provide:
    - bellman_operator(V) -> (V_new, policy_indices)
    - evaluate_policy_operator(V, policy_indices) -> V_eval
    """
    V = V_init.copy()
    error_path = []
    policy_indices = None
    t0 = perf_counter()

    for it in range(1, max_iter + 1):
        V_new, policy_indices = bellman_operator(V)
        for _ in range(max(0, int(howard_steps))):
            V_new = evaluate_policy_operator(V_new, policy_indices)
        error = float(np.max(np.abs(V_new - V)))
        error_path.append(error)
        V = V_new
        if error < tol:
            return {
                "value_function": V,
                "policy_indices": policy_indices,
                "diagnostics": {
                    "iterations": it,
                    "final_error": error,
                    "converged": True,
                    "error_path": error_path,
                    "runtime_seconds": float(perf_counter() - t0),
                    "method": "howard_vfi",
                    "howard_steps": int(howard_steps),
                },
            }

    return {
        "value_function": V,
        "policy_indices": policy_indices,
        "diagnostics": {
            "iterations": max_iter,
            "final_error": float(error_path[-1]) if error_path else np.nan,
            "converged": False,
            "error_path": error_path,
            "runtime_seconds": float(perf_counter() - t0),
            "method": "howard_vfi",
            "howard_steps": int(howard_steps),
        },
    }
