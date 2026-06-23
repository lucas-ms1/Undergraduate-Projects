"""
Tests for shared utility modules (grids, markov, interpolation).

Acceptance criteria derived from the feasibility-check script
contributed by the Step 6-7 owner (converted from Bash to pytest).
"""

import numpy as np
import pytest

from utils.grids import (
    linear_grid,
    curved_grid,
    clip_to_grid,
    is_consumption_positive,
    is_labor_feasible,
)
from utils.markov import validate_transition_matrix, simulate_shock_path
from utils.interpolation import interp_policy


# ── Grid tests ────────────────────────────────────────────────────────────

class TestGrids:
    def test_linear_grid_endpoints(self):
        g = linear_grid(0.0, 10.0, 5)
        assert g[0] == 0.0
        assert g[-1] == 10.0
        assert len(g) == 5

    def test_curved_grid_denser_at_bottom(self):
        g = curved_grid(0.0, 10.0, 100)
        midpoint_idx = len(g) // 2
        assert g[midpoint_idx] < 5.0  # denser below midpoint

    def test_clip_to_grid(self):
        assert clip_to_grid(-1.0, 0.0, 10.0) == 0.0
        assert clip_to_grid(15.0, 0.0, 10.0) == 10.0
        assert clip_to_grid(5.0, 0.0, 10.0) == 5.0

    def test_is_consumption_positive_true(self):
        assert is_consumption_positive(1)
        assert is_consumption_positive(np.array([0.1, 0.2]))

    def test_is_consumption_positive_false(self):
        assert not is_consumption_positive(0)
        assert not is_consumption_positive(np.array([1.0, -0.1]))

    def test_is_labor_feasible(self):
        assert is_labor_feasible(0.5, strict=True)
        assert is_labor_feasible(0.0, strict=False)
        assert not is_labor_feasible(0.0, strict=True)
        assert not is_labor_feasible(1.5, strict=False)


# ── Markov tests ──────────────────────────────────────────────────────────

class TestMarkov:
    def test_valid_matrix(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        assert validate_transition_matrix(P) is True

    def test_invalid_row_sum(self):
        P = np.array([[0.5, 0.3], [0.1, 0.9]])
        with pytest.raises(ValueError, match="sum to 1"):
            validate_transition_matrix(P)

    def test_negative_probability(self):
        P = np.array([[1.1, -0.1], [0.1, 0.9]])
        with pytest.raises(ValueError, match="negative"):
            validate_transition_matrix(P)

    def test_simulate_length(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        path = simulate_shock_path(P, T=100)
        assert len(path) == 100

    def test_simulate_values_in_range(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        path = simulate_shock_path(P, T=200)
        assert set(path).issubset({0, 1})

    def test_simulate_deterministic_seed(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        p1 = simulate_shock_path(P, T=50, seed=123)
        p2 = simulate_shock_path(P, T=50, seed=123)
        np.testing.assert_array_equal(p1, p2)


# ── Interpolation tests ──────────────────────────────────────────────────

class TestInterpolation:
    def test_exact_grid_points(self):
        g = np.array([0.0, 1.0, 2.0, 3.0])
        p = np.array([0.0, 2.0, 4.0, 6.0])
        result = interp_policy(g, p, np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_midpoint_interpolation(self):
        g = np.array([0.0, 1.0, 2.0])
        p = np.array([0.0, 2.0, 4.0])
        result = interp_policy(g, p, np.array([0.5]))
        np.testing.assert_allclose(result, [1.0])

    def test_clamp_outside_grid(self):
        g = np.array([0.0, 1.0, 2.0])
        p = np.array([10.0, 20.0, 30.0])
        result = interp_policy(g, p, np.array([-1.0, 5.0]))
        np.testing.assert_allclose(result, [10.0, 30.0])
