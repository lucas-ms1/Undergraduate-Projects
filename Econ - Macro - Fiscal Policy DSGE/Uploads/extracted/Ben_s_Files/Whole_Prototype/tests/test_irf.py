from config import BASELINE_PARAMS
from dsge.model import solve_model_objects
from simulation.irf import compute_irf


def test_irf_shape():
    model = solve_model_objects(BASELINE_PARAMS)
    irf = compute_irf(
        model["A_matrix"],
        model["B_matrix"],
        model["variable_index"],
        model["shock_index"],
        "g",
        "Lump-Sum transfers",
        horizon=40,
    )
    assert irf.shape[0] == 40
