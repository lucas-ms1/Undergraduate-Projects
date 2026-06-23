from dsge.steady_state import compute_steady_state
from config import BASELINE_PARAMS


def test_resource_constraint_holds():
    ss = compute_steady_state(BASELINE_PARAMS)
    assert abs(ss["resource_gap"]) < 1e-8
