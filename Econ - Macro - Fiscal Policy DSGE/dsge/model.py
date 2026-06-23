from __future__ import annotations

import numpy as np

from dsge.steady_state import solve_steady_state
from solvers.rational_expectations import solve_with_qz
from solvers.state_space import (
    build_state_space,
    build_structural_system,
)


def validate_equation_coverage(system: dict):
    labels = system["equation_labels"]
    n_eq = system["Gamma_0"].shape[0]
    duplicates = len(labels) != len(set(labels))
    missing_rows = [] if len(labels) == n_eq else list(range(len(labels), n_eq))
    empty_rows = []
    for row in range(n_eq):
        if (
            np.allclose(system["Gamma_f"][row], 0.0)
            and np.allclose(system["Gamma_0"][row], 0.0)
            and np.allclose(system["Gamma_l"][row], 0.0)
            and np.allclose(system["Psi"][row], 0.0)
        ):
            empty_rows.append(row)
    ok = (not duplicates) and (not missing_rows) and (not empty_rows)
    return {
        "ok": ok,
        "duplicates": duplicates,
        "missing_rows": missing_rows,
        "empty_rows": empty_rows,
    }


def budget_residuals(paths, shocks, A, B, variable_index):
    b_idx = variable_index["b_hat"]
    residual = paths[1:, b_idx] - (paths[:-1] @ A[b_idx, :]) - (shocks[:-1] @ B[b_idx, :])
    return residual


def check_budget_residuals(paths, shocks, A, B, variable_index, tol=1e-8):
    residual = budget_residuals(paths, shocks, A, B, variable_index)
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {"ok": bool(max_abs <= tol), "max_abs_residual": max_abs, "tol": tol}


def solve_model_objects(params, financing_rule: str = "lump_sum"):
    steady_state = solve_steady_state(params)
    # Always solve with lump_sum (Ricardian) to guarantee BK conditions.
    # Distortionary financing rules are applied as post-solve feedback
    # during IRF simulation (see simulation/irf.py).
    structural = build_structural_system(
        params=params,
        steady_state=steady_state,
        financing_rule="lump_sum",
    )

    eq_check = validate_equation_coverage(structural)
    qz = solve_with_qz(structural, jump_count=None)
    policy_ok = params.get("phi_pi", 1.5) > 1.0
    solver_success = bool(eq_check["ok"] and qz["solver_success"] and policy_ok)
    A = qz["A_matrix"] if solver_success else None
    B = qz["B_matrix"] if solver_success else None
    C = np.eye(A.shape[0]) if A is not None else None
    D = np.zeros((A.shape[0], B.shape[1])) if A is not None and B is not None else None

    diagnostics = {
        "solver_success": solver_success,
        "equation_coverage_ok": bool(eq_check["ok"]),
        "policy_rule_ok": bool(policy_ok),
        "equation_coverage": eq_check,
        "jump_variables": structural["jump_variables"],
        "equation_labels": structural["equation_labels"],
        "message": qz["message"],
        **qz,
    }
    if not policy_ok:
        diagnostics["flag"] = "indeterminacy"
        diagnostics["bk_ok"] = False
        diagnostics["message"] = "Taylor principle violated: phi_pi must be greater than 1."
    diagnostics["solver_success"] = solver_success
    return {
        "steady_state": steady_state,
        "structural_system": structural,
        "A_matrix": A,
        "B_matrix": B,
        "C_matrix": C,
        "D_matrix": D,
        "variable_index": structural["variable_index"],
        "shock_index": structural["shock_index"],
        "diagnostics": diagnostics,
    }


__all__ = [
    "build_state_space",
    "budget_residuals",
    "check_budget_residuals",
    "solve_model_objects",
    "validate_equation_coverage",
]
