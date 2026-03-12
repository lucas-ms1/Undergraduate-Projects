"""
Tests for Model 2: Stochastic Robinson Crusoe.

Acceptance criteria from plan.tex Step 10 "Done when":
  - Produces reasonable paths for capital, output, consumption, investment
  - Upper grid edge is not binding all the time
"""

import numpy as np
import pytest

from models.robinson_crusoe import solve, production, utility
from utils.grids import linear_grid


FAST_PARAMS = dict(n_k=80)


@pytest.fixture(scope="module")
def model2_result():
    return solve(FAST_PARAMS)


class TestProduction:
    def test_positive(self):
        k = np.array([1.0, 5.0, 10.0])
        y = production(k, z=1.0, alpha=0.36)
        assert np.all(y > 0)

    def test_cobb_douglas(self):
        k = np.array([1.0])
        y = production(k, z=1.1, alpha=0.36, A=1.0)
        np.testing.assert_allclose(y, [1.1 * 1.0 ** 0.36])


class TestUtility:
    def test_positive_consumption(self):
        c = np.array([0.5, 1.0, 2.0])
        u = utility(c, sigma=2.0)
        assert np.all(np.isfinite(u))

    def test_zero_gives_neg_inf(self):
        assert utility(np.array([0.0]), sigma=2.0)[0] == -np.inf

    def test_negative_gives_neg_inf(self):
        assert utility(np.array([-1.0]), sigma=2.0)[0] == -np.inf

    def test_log_case(self):
        c = np.array([1.0, np.e])
        u = utility(c, sigma=1.0)
        np.testing.assert_allclose(u, [0.0, 1.0], atol=1e-12)


class TestSolve:
    def test_convergence(self, model2_result):
        assert model2_result["diagnostics"]["converged"] is True

    def test_consumption_positive(self, model2_result):
        assert (model2_result["c_policy"] > 0).all()

    def test_no_nans(self, model2_result):
        assert not np.any(np.isnan(model2_result["c_policy"]))
        assert not np.any(np.isnan(model2_result["value_function"]))

    def test_policy_within_grid(self, model2_result):
        g = model2_result["grid"]
        pl = model2_result["policy_levels"]
        assert pl.min() >= g[0]
        assert pl.max() <= g[-1]

    def test_upper_edge_not_always_binding(self, model2_result):
        """Step 10 acceptance: the upper grid edge is not binding all the time."""
        g = model2_result["grid"]
        pl = model2_result["policy_levels"]
        frac_at_max = np.mean(pl == g[-1])
        assert frac_at_max < 0.5

    def test_return_keys(self, model2_result):
        for key in ("value_function", "policy_indices", "policy_levels",
                     "c_policy", "grid", "diagnostics"):
            assert key in model2_result

    def test_shapes_consistent(self, model2_result):
        n_k = len(model2_result["grid"])
        assert model2_result["value_function"].shape == (n_k, 2)
        assert model2_result["policy_indices"].shape == (n_k, 2)
        assert model2_result["policy_levels"].shape == (n_k, 2)
        assert model2_result["c_policy"].shape == (n_k, 2)
