from config import BASELINE_PARAMS
from dsge.model import solve_model_objects
from simulation.moments import compute_moments
from simulation.simulate import simulate_paths


def test_moment_count():
    model = solve_model_objects(BASELINE_PARAMS)
    sim = simulate_paths(model["A_matrix"], model["B_matrix"], horizon=200, seed=1)
    moments, _ = compute_moments(sim, model["variable_index"])
    assert len(moments) >= 10
