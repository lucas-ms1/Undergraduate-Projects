"""Assemble model objects used by the dashboard."""

import numpy as np

from dsge.steady_state import compute_steady_state
from solvers.state_space import build_state_space

EQUATION_ROW_MAP = {
    "resource_constraint": 0,
    "consumption_euler_reduced": 1,
    "investment_q_reduced": 2,
    "government_budget": 3,
    "new_keynesian_phillips_curve": 4,
    "labor_supply_reduced": 5,
    "monetary_policy_rule": 6,
}


def validate_equation_coverage(A, B, equation_row_map=None):
    """Ensure each expected equation appears exactly once in the system."""
    mapping = EQUATION_ROW_MAP if equation_row_map is None else equation_row_map
    n_eq = A.shape[0]
    rows = list(mapping.values())
    expected_rows = set(range(n_eq))
    seen_rows = set(rows)
    duplicates = len(rows) != len(seen_rows)
    missing_rows = sorted(expected_rows - seen_rows)
    extra_rows = sorted(seen_rows - expected_rows)

    empty_rows = []
    for row in sorted(seen_rows.intersection(expected_rows)):
        if np.allclose(A[row], 0.0) and np.allclose(B[row], 0.0):
            empty_rows.append(row)

    ok = (not duplicates) and (not missing_rows) and (not extra_rows) and (not empty_rows)
    return {
        "ok": ok,
        "duplicates": duplicates,
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
        "empty_rows": empty_rows,
    }


def budget_residuals(paths, shocks, A, B, variable_index):
    """Compute government budget residual path from the state transition equation."""
    b_idx = variable_index["b_hat"]
    residual = paths[1:, b_idx] - (paths[:-1] @ A[b_idx, :]) - (shocks[:-1] @ B[b_idx, :])
    return residual


def check_budget_residuals(paths, shocks, A, B, variable_index, tol=1e-8):
    residual = budget_residuals(paths, shocks, A, B, variable_index)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {"ok": bool(max_abs <= tol), "max_abs_residual": max_abs, "tol": tol}


def solve_model_objects(params):
    steady_state = compute_steady_state(params)
    A, B, C, D, variable_index, shock_index = build_state_space(params)
    eq_check = validate_equation_coverage(A, B)
    diagnostics = {
        "solver_success": bool(eq_check["ok"]),
        "equation_coverage_ok": bool(eq_check["ok"]),
        "equation_coverage": eq_check,
        "message": (
            "State-space assembled successfully."
            if eq_check["ok"]
            else "Equation coverage check failed: each equation must appear exactly once."
        ),
    }
    return {
        "steady_state": steady_state,
        "A_matrix": A,
        "B_matrix": B,
        "C_matrix": C,
        "D_matrix": D,
        "variable_index": variable_index,
        "shock_index": shock_index,
        "diagnostics": diagnostics,
    }
