"""
Tests for simulation.simulate, simulation.forecast, and simulation.moments.

Covers all three models plus the Model 3 labor-only variant.
"""

import numpy as np
import pytest

from models.consumption_savings import solve as solve_m1
from models.robinson_crusoe import solve as solve_m2
from models.labor_supply import solve as solve_m3
from simulation.simulate import simulate_model
from simulation.forecast import forecast_model
from simulation.moments import compute_moments

P = np.array([[0.9, 0.1],
              [0.1, 0.9]])

# Solve with small grids for speed
FAST_M1 = dict(n_a=80)
FAST_M2 = dict(n_k=80)
FAST_M3_ASSETS = dict(n_a=50, n_L=30, include_assets=True)
FAST_M3_LABOR = dict(n_L=40, include_assets=False)

Y_VALS = np.array([0.5, 1.5])
Z_VALS = np.array([0.9, 1.1])
W_VALS = np.array([0.8, 1.2])

M2_PARAMS = dict(alpha=0.36, delta=0.10, A=1.0)


@pytest.fixture(scope="module")
def m1():
    return solve_m1(FAST_M1)


@pytest.fixture(scope="module")
def m2():
    return solve_m2(FAST_M2)


@pytest.fixture(scope="module")
def m3_assets():
    return solve_m3(FAST_M3_ASSETS)


@pytest.fixture(scope="module")
def m3_labor():
    return solve_m3(FAST_M3_LABOR)


# ═══════════════════════════════════════════════════════════════════════════
# simulate_model
# ═══════════════════════════════════════════════════════════════════════════

class TestSimulateModel1:
    def test_returns_expected_keys(self, m1):
        sim = simulate_model(m1, Y_VALS, P, initial_state=5.0,
                             T_sim=100, model_name="model1")
        assert set(sim.keys()) == {"a", "c", "y", "shock_idx"}

    def test_series_lengths(self, m1):
        T = 150
        sim = simulate_model(m1, Y_VALS, P, initial_state=5.0,
                             T_sim=T, model_name="model1")
        for key in ("a", "c", "y", "shock_idx"):
            assert len(sim[key]) == T

    def test_deterministic_seed(self, m1):
        kw = dict(shock_vals=Y_VALS, P=P, initial_state=5.0,
                  T_sim=100, seed=99, model_name="model1")
        s1 = simulate_model(m1, **kw)
        s2 = simulate_model(m1, **kw)
        np.testing.assert_array_equal(s1["a"], s2["a"])

    def test_consumption_positive(self, m1):
        sim = simulate_model(m1, Y_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model1")
        assert (sim["c"] > 0).all()


class TestSimulateModel2:
    def test_returns_expected_keys(self, m2):
        sim = simulate_model(m2, Z_VALS, P, initial_state=5.0,
                             T_sim=100, model_name="model2",
                             model_params=M2_PARAMS)
        assert set(sim.keys()) == {"k", "c", "y", "investment", "shock_idx"}

    def test_series_lengths(self, m2):
        T = 150
        sim = simulate_model(m2, Z_VALS, P, initial_state=5.0,
                             T_sim=T, model_name="model2",
                             model_params=M2_PARAMS)
        for key in ("k", "c", "y", "investment"):
            assert len(sim[key]) == T

    def test_consumption_positive(self, m2):
        sim = simulate_model(m2, Z_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model2",
                             model_params=M2_PARAMS)
        assert (sim["c"] > 0).all()

    def test_output_positive(self, m2):
        sim = simulate_model(m2, Z_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model2",
                             model_params=M2_PARAMS)
        assert (sim["y"] > 0).all()

    def test_deterministic_seed(self, m2):
        kw = dict(shock_vals=Z_VALS, P=P, initial_state=5.0,
                  T_sim=100, seed=99, model_name="model2",
                  model_params=M2_PARAMS)
        s1 = simulate_model(m2, **kw)
        s2 = simulate_model(m2, **kw)
        np.testing.assert_array_equal(s1["k"], s2["k"])


class TestSimulateModel3Assets:
    def test_returns_expected_keys(self, m3_assets):
        sim = simulate_model(m3_assets, W_VALS, P, initial_state=5.0,
                             T_sim=100, model_name="model3")
        assert set(sim.keys()) == {"a", "c", "labor", "earnings", "shock_idx"}

    def test_labor_in_unit_interval(self, m3_assets):
        sim = simulate_model(m3_assets, W_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model3")
        assert sim["labor"].min() >= 0.0
        assert sim["labor"].max() <= 1.0

    def test_consumption_positive(self, m3_assets):
        sim = simulate_model(m3_assets, W_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model3")
        assert (sim["c"] > 0).all()


class TestSimulateModel3LaborOnly:
    def test_returns_expected_keys(self, m3_labor):
        sim = simulate_model(m3_labor, W_VALS, P, initial_state=0.0,
                             T_sim=100, model_name="model3_labor_only")
        assert set(sim.keys()) == {"c", "labor", "earnings", "shock_idx"}

    def test_no_assets_key(self, m3_labor):
        sim = simulate_model(m3_labor, W_VALS, P, initial_state=0.0,
                             T_sim=100, model_name="model3_labor_only")
        assert "a" not in sim

    def test_consumption_positive(self, m3_labor):
        sim = simulate_model(m3_labor, W_VALS, P, initial_state=0.0,
                             T_sim=200, model_name="model3_labor_only")
        assert (sim["c"] > 0).all()

    def test_labor_in_unit_interval(self, m3_labor):
        sim = simulate_model(m3_labor, W_VALS, P, initial_state=0.0,
                             T_sim=200, model_name="model3_labor_only")
        assert sim["labor"].min() >= 0.0
        assert sim["labor"].max() <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# forecast_model
# ═══════════════════════════════════════════════════════════════════════════

class TestForecastModel1:
    def test_returns_expected_keys(self, m1):
        fc = forecast_model(m1, Y_VALS, shock_path=[1, 1, 0],
                            current_state=5.0, model_name="model1")
        assert set(fc.keys()) == {"a", "c", "y", "shock_idx"}

    def test_horizon_matches_shock_path(self, m1):
        path = [1, 0, 1, 0, 0]
        fc = forecast_model(m1, Y_VALS, shock_path=path,
                            current_state=5.0, model_name="model1")
        for key in ("a", "c", "y", "shock_idx"):
            assert len(fc[key]) == len(path)

    def test_consumption_positive(self, m1):
        fc = forecast_model(m1, Y_VALS, shock_path=[1] * 10,
                            current_state=5.0, model_name="model1")
        assert (fc["c"] > 0).all()

    def test_different_paths_change_income(self, m1):
        fc_high = forecast_model(m1, Y_VALS, shock_path=[1, 1, 1],
                                 current_state=5.0, model_name="model1")
        fc_low = forecast_model(m1, Y_VALS, shock_path=[0, 0, 0],
                                current_state=5.0, model_name="model1")
        assert not np.array_equal(fc_high["y"], fc_low["y"])

    def test_shock_idx_matches_input(self, m1):
        path = [0, 1, 0, 1]
        fc = forecast_model(m1, Y_VALS, shock_path=path,
                            current_state=5.0, model_name="model1")
        np.testing.assert_array_equal(fc["shock_idx"], path)


class TestForecastModel2:
    def test_returns_expected_keys(self, m2):
        fc = forecast_model(m2, Z_VALS, shock_path=[1, 0, 1],
                            current_state=5.0, model_name="model2",
                            model_params=M2_PARAMS)
        assert set(fc.keys()) == {"k", "c", "y", "investment", "shock_idx"}

    def test_consumption_positive(self, m2):
        fc = forecast_model(m2, Z_VALS, shock_path=[1] * 10,
                            current_state=5.0, model_name="model2",
                            model_params=M2_PARAMS)
        assert (fc["c"] > 0).all()

    def test_different_paths_change_output(self, m2):
        fc_high = forecast_model(m2, Z_VALS, shock_path=[1, 1, 1],
                                 current_state=5.0, model_name="model2",
                                 model_params=M2_PARAMS)
        fc_low = forecast_model(m2, Z_VALS, shock_path=[0, 0, 0],
                                current_state=5.0, model_name="model2",
                                model_params=M2_PARAMS)
        assert not np.array_equal(fc_high["y"], fc_low["y"])


class TestForecastModel3:
    def test_assets_returns_expected_keys(self, m3_assets):
        fc = forecast_model(m3_assets, W_VALS, shock_path=[1, 0],
                            current_state=5.0, model_name="model3")
        assert set(fc.keys()) == {"a", "c", "labor", "earnings", "shock_idx"}

    def test_labor_only_returns_expected_keys(self, m3_labor):
        fc = forecast_model(m3_labor, W_VALS, shock_path=[1, 0],
                            current_state=0.0, model_name="model3_labor_only")
        assert set(fc.keys()) == {"c", "labor", "earnings", "shock_idx"}

    def test_labor_only_consumption_positive(self, m3_labor):
        fc = forecast_model(m3_labor, W_VALS, shock_path=[1] * 5,
                            current_state=0.0, model_name="model3_labor_only")
        assert (fc["c"] > 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# moments compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestMomentsCompat:
    def test_model1_moments(self, m1):
        sim = simulate_model(m1, Y_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model1")
        m = compute_moments({k: v for k, v in sim.items() if k != "shock_idx"},
                            income_key="y")
        assert "c" in m
        assert np.isfinite(m["c"]["mean"])

    def test_model2_moments(self, m2):
        sim = simulate_model(m2, Z_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model2",
                             model_params=M2_PARAMS)
        m = compute_moments({k: v for k, v in sim.items() if k != "shock_idx"},
                            income_key="y")
        assert "c" in m
        assert "k" in m
        assert np.isfinite(m["c"]["mean"])

    def test_model3_assets_moments(self, m3_assets):
        sim = simulate_model(m3_assets, W_VALS, P, initial_state=5.0,
                             T_sim=200, model_name="model3")
        m = compute_moments({k: v for k, v in sim.items() if k != "shock_idx"},
                            income_key="earnings")
        assert "labor" in m
        assert np.isfinite(m["labor"]["mean"])

    def test_model3_labor_only_moments(self, m3_labor):
        sim = simulate_model(m3_labor, W_VALS, P, initial_state=0.0,
                             T_sim=200, model_name="model3_labor_only")
        m = compute_moments({k: v for k, v in sim.items() if k != "shock_idx"},
                            income_key="earnings")
        assert "labor" in m
        assert np.isfinite(m["c"]["mean"])
