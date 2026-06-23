"""Assemble model objects used by the dashboard.

Builds the structural state-space, solves it via Blanchard-Kahn /
Schur decomposition, and returns the reduced-form matrices for
simulation together with steady-state values and diagnostics.
"""

from dsge.steady_state import compute_steady_state
from solvers.rational_expectations import solve_with_qz
from solvers.state_space import build_state_space


def solve_model_objects(params):
    steady_state = compute_steady_state(params)

    # 1. Build STRUCTURAL state-space (may have explosive eigenvalues)
    A_struct, B_struct, C, D, variable_index, shock_index = build_state_space(params)

    # 2. Solve rational-expectations model via Schur / BK
    qz = solve_with_qz(A_struct, B_struct)

    # 3. Use the SOLVED reduced-form matrices (all eigenvalues < 1)
    return {
        "steady_state": steady_state,
        "A_matrix": qz["A_matrix"],
        "B_matrix": qz["B_matrix"],
        "A_structural": A_struct,
        "B_structural": B_struct,
        "C_matrix": C,
        "D_matrix": D,
        "variable_index": variable_index,
        "shock_index": shock_index,
        "diagnostics": qz,
    }
