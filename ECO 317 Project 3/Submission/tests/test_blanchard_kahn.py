import numpy as np

from dsge.calibration import override_parameters
from dsge.model import solve_model_objects
from solvers.rational_expectations import solve_with_qz


def test_qz_infers_five_jump_variables():
    model = solve_model_objects(override_parameters())
    out = solve_with_qz(model["structural_system"])
    assert out["jump_count"] == 5


def test_baseline_model_satisfies_bk_and_returns_matrices():
    model = solve_model_objects(override_parameters())
    diag = model["diagnostics"]
    assert diag["solver_success"]
    assert diag["flag"] == "converged"
    assert diag["explosive_roots"] == diag["jump_count"] == 5
    assert np.all(np.isfinite(model["A_matrix"]))
    assert np.all(np.isfinite(model["B_matrix"]))


def test_failed_bk_does_not_return_transition_matrices():
    params = override_parameters({"phi_pi": 0.5})
    model = solve_model_objects(params)
    assert not model["diagnostics"]["solver_success"]
    assert model["A_matrix"] is None
    assert model["B_matrix"] is None


def test_complex_pair_counts_as_two_explosive_roots():
    gamma_f = np.eye(4)
    roots = np.array([1.2 + 0.4j, 1.2 - 0.4j, 0.8, 0.7], dtype=complex)
    gamma_0 = -np.diag(roots)
    system = {
        "Gamma_f": gamma_f,
        "Gamma_0": gamma_0,
        "Gamma_l": np.zeros((4, 4)),
        "Psi": np.zeros((4, 1)),
        "jump_variables": ["j1", "j2"],
        "variable_index": {f"x{i}": i for i in range(4)},
    }
    out = solve_with_qz(system)
    assert out["explosive_roots"] == 2
    assert out["flag"] == "converged"


def test_indeterminacy_when_explosive_roots_are_too_few():
    gamma_f = np.diag([1.2, 0.9, 0.8])
    gamma_0 = -np.eye(3)
    system = {
        "Gamma_f": gamma_f,
        "Gamma_0": gamma_0,
        "Gamma_l": np.zeros((3, 3)),
        "Psi": np.zeros((3, 1)),
        "jump_variables": ["j1", "j2"],
        "variable_index": {f"x{i}": i for i in range(3)},
    }
    out = solve_with_qz(system)
    assert out["flag"] == "indeterminacy"


def test_no_solution_when_explosive_roots_are_too_many():
    gamma_f = np.diag([1.2, 1.1, 0.8])
    gamma_0 = -np.eye(3)
    system = {
        "Gamma_f": gamma_f,
        "Gamma_0": gamma_0,
        "Gamma_l": np.zeros((3, 3)),
        "Psi": np.zeros((3, 1)),
        "jump_variables": ["j1"],
        "variable_index": {f"x{i}": i for i in range(3)},
    }
    out = solve_with_qz(system)
    assert out["flag"] == "no_solution"


def test_model_returns_structural_system():
    model = solve_model_objects(override_parameters())
    assert "structural_system" in model
    assert model["A_matrix"].shape[0] == len(model["variable_index"])
