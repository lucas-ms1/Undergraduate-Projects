from config import BASELINE_PARAMS
from dsge.model import solve_model_objects
from solvers.rational_expectations import solve_with_qz


def test_qz_returns_diagnostics():
    model = solve_model_objects(BASELINE_PARAMS)
    out = solve_with_qz(model["A_matrix"], model["B_matrix"])
    assert "eigenvalues" in out
    assert "flag" in out
