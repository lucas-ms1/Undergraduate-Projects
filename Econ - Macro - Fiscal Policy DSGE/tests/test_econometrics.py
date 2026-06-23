"""
tests/test_econometrics.py
Step 11b – Econometric regression tests
------------------------------------------
Tests that:
  1. Taylor-rule recovery produces estimates within 15% of structural values
  2. Consumption-smoothness regression returns valid output
  3. Variance ratios have bootstrapped SEs
  4. OLS helper returns correct shapes and finite values
"""

import numpy as np
import pytest

from dsge.calibration import override_parameters
from dsge.model import solve_model_objects
from simulation.econometrics import (
    taylor_rule_regression,
    consumption_smoothness_regression,
    variance_ratios_bootstrap,
    run_all_econometrics,
    OLSResult,
)
from simulation.simulate import run_simulation


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – synthetic Taylor-rule data
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def taylor_data():
    """Generate synthetic data from structural Taylor-rule targets."""
    rng = np.random.default_rng(42)
    T = 4000
    rho_i = 0.80
    phi_pi = 1.50
    phi_y = 0.125
    pi = rng.normal(0, 0.01, T)
    y = rng.normal(0, 0.01, T)
    r = np.zeros(T)
    for t in range(1, T):
        r[t] = (
            rho_i * r[t - 1]
            + (1.0 - rho_i) * (phi_pi * pi[t] + phi_y * y[t])
            + rng.normal(0, 0.0005)
        )
    return r, pi, y


@pytest.fixture
def consumption_data():
    """
    Synthetic consumption with habit:
      c_t = 0.1 + 0.3 * y_t + 0.1 * y_{t-1} + 0.5 * c_{t-1} + eps_t
    """
    rng = np.random.default_rng(99)
    T = 1500
    y = np.cumsum(rng.normal(0, 0.01, T))
    c = np.zeros(T)
    for t in range(1, T):
        c[t] = 0.3 * y[t] + 0.1 * y[t-1] + 0.5 * c[t-1] + rng.normal(0, 0.005)
    return c, y


# ──────────────────────────────────────────────────────────────────────────────
# Taylor-rule recovery
# ──────────────────────────────────────────────────────────────────────────────

class TestTaylorRuleRecovery:
    def test_returns_ols_result(self, taylor_data):
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        assert isinstance(result, OLSResult)

    def test_correct_regressors(self, taylor_data):
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        assert len(result.regressors) == 4
        assert result.n_obs == len(r) - 1

    def test_rho_i_recovery(self, taylor_data):
        """Recovered ρ_i should be within 15% of 0.80."""
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        coefs = result.as_dict()
        rho_hat = coefs["rho_i_hat"]
        assert abs(rho_hat - 0.80) / 0.80 < 0.15, f"ρ̂_i = {rho_hat:.4f}, expected ~0.80"

    def test_phi_pi_recovery(self, taylor_data):
        """Recovered φ_π should be within 15% of 0.30."""
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        coefs = result.as_dict()
        phi_hat = coefs["phi_pi_hat"]
        assert abs(phi_hat - 1.50) / 1.50 < 0.15, f"φ̂_π = {phi_hat:.4f}, expected ~1.50"

    def test_phi_y_recovery(self, taylor_data):
        """Recovered structural phi_y should be near 0.125."""
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        coefs = result.as_dict()
        phi_hat = coefs["phi_y_hat"]
        assert abs(phi_hat - 0.125) / 0.125 < 0.25, f"phi_y_hat = {phi_hat:.4f}, expected ~0.125"

    def test_r_squared_high(self, taylor_data):
        """R² should be high for data generated from the same DGP."""
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        assert result.r_squared > 0.90

    def test_std_errors_positive(self, taylor_data):
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        assert np.all(result.std_errors > 0)

    def test_summary_rows(self, taylor_data):
        r, pi, y = taylor_data
        result = taylor_rule_regression(r, pi, y)
        rows = result.summary_rows()
        assert len(rows) == 6  # 4 regressors + R² + N


# ──────────────────────────────────────────────────────────────────────────────
# Consumption smoothness
# ──────────────────────────────────────────────────────────────────────────────

class TestConsumptionSmoothness:
    def test_returns_ols_result(self, consumption_data):
        c, y = consumption_data
        result = consumption_smoothness_regression(c, y)
        assert isinstance(result, OLSResult)

    def test_habit_coefficient_positive(self, consumption_data):
        """c_{t-1} coefficient should be positive (habit persistence)."""
        c, y = consumption_data
        result = consumption_smoothness_regression(c, y)
        coefs = result.as_dict()
        assert coefs["c_lag1"] > 0

    def test_income_coefficient_positive(self, consumption_data):
        """y_t coefficient should be positive (income → consumption)."""
        c, y = consumption_data
        result = consumption_smoothness_regression(c, y)
        coefs = result.as_dict()
        assert coefs["y_t"] > 0

    def test_r_squared_reasonable(self, consumption_data):
        c, y = consumption_data
        result = consumption_smoothness_regression(c, y)
        assert result.r_squared > 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Variance ratios
# ──────────────────────────────────────────────────────────────────────────────

class TestVarianceRatios:
    @pytest.fixture
    def toy_series(self):
        rng = np.random.default_rng(0)
        T = 1000
        return {
            "y": rng.normal(0, 1.0, T),
            "c": rng.normal(0, 0.5, T),
            "i": rng.normal(0, 3.0, T),
        }

    def test_returns_list(self, toy_series):
        results = variance_ratios_bootstrap(toy_series)
        assert isinstance(results, list)
        assert len(results) == 2  # c and i (not y)

    def test_consumption_ratio_below_one(self, toy_series):
        results = variance_ratios_bootstrap(toy_series)
        c_result = [r for r in results if "c" in r.name][0]
        assert c_result.ratio < 1.0

    def test_investment_ratio_above_one(self, toy_series):
        results = variance_ratios_bootstrap(toy_series)
        i_result = [r for r in results if "i" in r.name][0]
        assert i_result.ratio > 1.0

    def test_bootstrap_se_positive(self, toy_series):
        results = variance_ratios_bootstrap(toy_series)
        for r in results:
            assert r.se > 0

    def test_ci_contains_point_estimate(self, toy_series):
        results = variance_ratios_bootstrap(toy_series)
        for r in results:
            assert r.ci_low <= r.ratio <= r.ci_high


# ──────────────────────────────────────────────────────────────────────────────
# run_all_econometrics convenience
# ──────────────────────────────────────────────────────────────────────────────

class TestRunAll:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        T = 500
        series = {
            "y": rng.normal(0, 1, T),
            "c": rng.normal(0, 0.5, T),
            "r": rng.normal(0, 0.3, T),
            "pi": rng.normal(0, 0.2, T),
        }
        result = run_all_econometrics(series)
        assert "taylor_rule" in result
        assert "consumption_smoothness" in result
        assert "variance_ratios" in result

    def test_default_observables_include_taylor_rule_inputs(self):
        model = solve_model_objects(override_parameters())
        series = run_simulation(model["A_matrix"], model["B_matrix"], T=200, seed=123)
        result = run_all_econometrics(series)
        assert "r" in series
        assert "taylor_rule" in result

    def test_actual_model_taylor_rule_recovery(self):
        params = override_parameters()
        model = solve_model_objects(params)
        idx = model["variable_index"]
        sidx = model["shock_index"]
        obs = {
            "y": idx["y_hat"],
            "c": idx["c_hat"],
            "i": idx["i_hat"],
            "l": idx["l_hat"],
            "pi": idx["pi_hat"],
            "r": idx["i_nom_hat"],
            "m_policy": idx["m_policy_hat"],
        }
        shock_stds = np.zeros(len(sidx))
        for shock, param in {
            "tfp": "sigma_a",
            "g_c": "sigma_g",
            "g_i": "sigma_gi",
            "monetary": "sigma_m",
            "risk": "sigma_rp",
        }.items():
            shock_stds[sidx[shock]] = params[param]

        series = run_simulation(
            model["A_matrix"], model["B_matrix"], T=1000, seed=42,
            obs_index=obs, shock_stds=shock_stds,
        )
        coefs = run_all_econometrics(series, structural_params=params)["taylor_rule"].as_dict()
        assert abs(coefs["rho_i_hat"] - params["rho_i"]) < 1e-8
        assert abs(coefs["phi_pi_hat"] - params["phi_pi"]) < 1e-8
        assert abs(coefs["phi_y_hat"] - params["phi_y"]) < 1e-8

    def test_shock_scales_change_simulated_volatility_not_solution(self):
        low = override_parameters({"sigma_a": 0.001})
        high = override_parameters({"sigma_a": 0.05})
        model_low = solve_model_objects(low)
        model_high = solve_model_objects(high)
        assert np.allclose(model_low["A_matrix"], model_high["A_matrix"])
        assert np.allclose(model_low["B_matrix"], model_high["B_matrix"])

        sidx = model_low["shock_index"]
        idx = model_low["variable_index"]
        obs = {"y": idx["y_hat"]}
        low_stds = np.zeros(len(sidx))
        high_stds = np.zeros(len(sidx))
        low_stds[sidx["tfp"]] = low["sigma_a"]
        high_stds[sidx["tfp"]] = high["sigma_a"]
        low_series = run_simulation(
            model_low["A_matrix"], model_low["B_matrix"], T=500, seed=1,
            obs_index=obs, shock_stds=low_stds,
        )
        high_series = run_simulation(
            model_high["A_matrix"], model_high["B_matrix"], T=500, seed=1,
            obs_index=obs, shock_stds=high_stds,
        )
        assert np.std(high_series["y"]) > 10.0 * np.std(low_series["y"])
