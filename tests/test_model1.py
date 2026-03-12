"""
Tests for Model 1: Stochastic Consumption-Savings.

Acceptance criteria derived from the feasibility-check script
contributed by the Step 6-7 owner (converted from Bash to pytest).
"""

import numpy as np
import pytest

from models.consumption_savings import solve, utility, feasibility_mask
from utils.grids import linear_grid


# Use a smaller grid so tests stay fast
FAST_PARAMS = dict(n_a=80)


@pytest.fixture(scope="module")
def model1_result():
    return solve(FAST_PARAMS)


class TestUtility:
    def test_positive_consumption(self):
        c = np.array([0.5, 1.0, 2.0])
        u = utility(c, sigma=2.0)
        assert np.all(np.isfinite(u))

    def test_zero_consumption_gives_neg_inf(self):
        u = utility(np.array([0.0]), sigma=2.0)
        assert u[0] == -np.inf

    def test_negative_consumption_gives_neg_inf(self):
        u = utility(np.array([-1.0]), sigma=2.0)
        assert u[0] == -np.inf

    def test_log_case_sigma_one(self):
        c = np.array([1.0, np.e])
        u = utility(c, sigma=1.0)
        np.testing.assert_allclose(u, [0.0, 1.0], atol=1e-12)


class TestFeasibilityMask:
    def test_shape(self):
        a_grid = linear_grid(0.0, 20.0, 50)
        y_vals = np.array([0.5, 1.5])
        mask = feasibility_mask(a_grid, y_vals, r=0.03)
        assert mask.shape == (50, 50, 2)

    def test_not_all_true_or_false(self):
        a_grid = linear_grid(0.0, 20.0, 50)
        y_vals = np.array([0.5, 1.5])
        mask = feasibility_mask(a_grid, y_vals, r=0.03)
        assert 0 < mask.mean() < 1


class TestSolve:
    def test_convergence(self, model1_result):
        assert model1_result["diagnostics"]["converged"] is True

    def test_consumption_positive(self, model1_result):
        assert (model1_result["c_policy"] > 0).all()

    def test_no_nans(self, model1_result):
        assert not np.any(np.isnan(model1_result["c_policy"]))
        assert not np.any(np.isnan(model1_result["value_function"]))

    def test_policy_within_grid(self, model1_result):
        g = model1_result["grid"]
        pl = model1_result["policy_levels"]
        assert pl.min() >= g[0]
        assert pl.max() <= g[-1]

    def test_return_keys(self, model1_result):
        for key in ("value_function", "policy_indices", "policy_levels",
                     "c_policy", "grid", "diagnostics"):
            assert key in model1_result

    def test_shapes_consistent(self, model1_result):
        n_a = len(model1_result["grid"])
        assert model1_result["value_function"].shape == (n_a, 2)
        assert model1_result["policy_indices"].shape == (n_a, 2)
        assert model1_result["policy_levels"].shape == (n_a, 2)
        assert model1_result["c_policy"].shape == (n_a, 2)
