"""
Tests for Model 3: Endogenous Labor Supply.

Acceptance criteria from plan.tex Step 11 "Done when":
  - Labor stays in [0, 1]
  - Leisure remains valid
  - Policy behaviour is economically interpretable

Both asset-inclusive and labor-only variants are tested.
"""

import numpy as np
import pytest

from models.labor_supply import solve, utility, labor_vs_wage_for_plot


# Use small grids for fast tests
FAST_ASSETS = dict(n_a=50, n_L=30, include_assets=True)
FAST_LABOR_ONLY = dict(n_L=40, include_assets=False)


@pytest.fixture(scope="module")
def model3_assets_result():
    return solve(FAST_ASSETS)


@pytest.fixture(scope="module")
def model3_labor_only_result():
    return solve(FAST_LABOR_ONLY)


# ── Utility tests ─────────────────────────────────────────────────────────

class TestUtility:
    def test_positive_inputs(self):
        c = np.array([0.5, 1.0])
        leisure = np.array([0.5, 0.8])
        u = utility(c, leisure, sigma=2.0, psi=1.0, nu=2.0)
        assert np.all(np.isfinite(u))

    def test_zero_consumption_gives_neg_inf(self):
        u = utility(np.array([0.0]), np.array([0.5]), 2.0, 1.0, 2.0)
        assert u[0] == -np.inf

    def test_zero_leisure_gives_neg_inf(self):
        u = utility(np.array([1.0]), np.array([0.0]), 2.0, 1.0, 2.0)
        assert u[0] == -np.inf

    def test_log_cases(self):
        c = np.array([np.e])
        leisure = np.array([np.e])
        u = utility(c, leisure, sigma=1.0, psi=1.0, nu=1.0)
        np.testing.assert_allclose(u, [2.0], atol=1e-12)


# ── Asset-inclusive variant ───────────────────────────────────────────────

class TestSolveWithAssets:
    def test_convergence(self, model3_assets_result):
        assert model3_assets_result["diagnostics"]["converged"] is True

    def test_consumption_positive(self, model3_assets_result):
        assert (model3_assets_result["c_policy"] > 0).all()

    def test_labor_in_unit_interval(self, model3_assets_result):
        labor = model3_assets_result["policy_levels"]["labor"]
        assert labor.min() >= 0.0
        assert labor.max() <= 1.0

    def test_leisure_valid(self, model3_assets_result):
        labor = model3_assets_result["policy_levels"]["labor"]
        leisure = 1.0 - labor
        assert (leisure >= 0).all()
        assert (leisure <= 1).all()

    def test_no_nans(self, model3_assets_result):
        assert not np.any(np.isnan(model3_assets_result["c_policy"]))
        assert not np.any(np.isnan(model3_assets_result["value_function"]))

    def test_return_keys(self, model3_assets_result):
        for key in ("value_function", "policy_indices", "policy_levels",
                     "c_policy", "grid", "diagnostics"):
            assert key in model3_assets_result

    def test_policy_levels_keys(self, model3_assets_result):
        pl = model3_assets_result["policy_levels"]
        assert "labor" in pl
        assert "savings" in pl

    def test_grid_keys(self, model3_assets_result):
        g = model3_assets_result["grid"]
        assert "a_grid" in g
        assert "labor_grid" in g

    def test_shapes_consistent(self, model3_assets_result):
        n_a = len(model3_assets_result["grid"]["a_grid"])
        assert model3_assets_result["value_function"].shape == (n_a, 2)
        assert model3_assets_result["c_policy"].shape == (n_a, 2)
        assert model3_assets_result["policy_levels"]["labor"].shape == (n_a, 2)
        assert model3_assets_result["policy_levels"]["savings"].shape == (n_a, 2)


# ── Labor-only variant ───────────────────────────────────────────────────

class TestSolveLaborOnly:
    def test_convergence(self, model3_labor_only_result):
        assert model3_labor_only_result["diagnostics"]["converged"] is True

    def test_consumption_positive(self, model3_labor_only_result):
        assert (model3_labor_only_result["c_policy"] > 0).all()

    def test_labor_in_unit_interval(self, model3_labor_only_result):
        labor = model3_labor_only_result["policy_levels"]["labor"]
        assert labor.min() >= 0.0
        assert labor.max() <= 1.0

    def test_value_function_shape(self, model3_labor_only_result):
        assert model3_labor_only_result["value_function"].shape == (2,)

    def test_no_asset_grid(self, model3_labor_only_result):
        assert "a_grid" not in model3_labor_only_result["grid"]

    def test_return_keys(self, model3_labor_only_result):
        for key in ("value_function", "policy_indices", "policy_levels",
                     "c_policy", "grid", "diagnostics"):
            assert key in model3_labor_only_result

    def test_labor_responds_to_wage(self, model3_labor_only_result):
        """Economically interpretable: labor should differ across wage states.

        Direction depends on income vs substitution effect — a backward-bending
        supply curve is valid when risk aversion is high enough.
        """
        labor = model3_labor_only_result["policy_levels"]["labor"]
        assert labor[0] != labor[1]


# ── Intuition plot helper ─────────────────────────────────────────────────

class TestLaborVsWage:
    def test_assets_variant(self, model3_assets_result):
        w_vals = np.array([0.8, 1.2])
        ws, ls = labor_vs_wage_for_plot(model3_assets_result, w_vals, a_idx=0)
        assert len(ws) == 2
        assert len(ls) == 2

    def test_labor_only_variant(self, model3_labor_only_result):
        w_vals = np.array([0.8, 1.2])
        ws, ls = labor_vs_wage_for_plot(model3_labor_only_result, w_vals)
        assert len(ws) == 2
        assert len(ls) == 2
