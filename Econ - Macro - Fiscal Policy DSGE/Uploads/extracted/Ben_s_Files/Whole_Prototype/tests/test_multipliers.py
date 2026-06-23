from config import BASELINE_PARAMS
from dsge.model import solve_model_objects
from policy.multipliers import compute_multipliers
from simulation.irf import compute_irf


def test_multipliers_return_keys():
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
    out = compute_multipliers(irf, model["variable_index"])
    assert "impact_multiplier" in out
    assert "cumulative_multiplier" in out
