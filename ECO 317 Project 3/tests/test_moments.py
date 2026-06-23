"""
tests/test_moments.py
Step 8 – Simulation and moment computation tests
---------------------------------------------------
Owner: Steps 8-9 contributor

Tests for simulation/simulate.py and simulation/moments.py.
"""

import numpy as np
import pytest

from simulation.simulate import simulate, extract_observables, run_simulation
from simulation.moments import (
    compute_moments,
    compute_all_moments,
    hp_cycle,
    consumption_smoother_than_output,
    investment_more_volatile_than_output,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – simple 5-variable AR(1) toy model
# ──────────────────────────────────────────────────────────────────────────────

N_VARS = 7   # matches our 7-variable state vector

@pytest.fixture
def toy_AB():
    """
    Toy A and B for a 7-variable AR(1) system.
    Consumption (col 1) has lower shock variance → smoother than output.
    Investment (col 2) has higher shock variance → more volatile.
    """
    rho = 0.8
    A = rho * np.eye(N_VARS)
    sigma = np.array([1.0, 0.5, 3.0, 0.4, 0.6, 0.8, 0.3])
    B = np.diag(sigma)
    return A, B


@pytest.fixture
def toy_obs_index():
    return {"y": 0, "c": 1, "i": 2, "l": 5, "pi": 4}


@pytest.fixture
def simulated_series(toy_AB, toy_obs_index):
    A, B = toy_AB
    return run_simulation(A, B, T=1000, seed=42, obs_index=toy_obs_index)


@pytest.fixture
def raw_moments(simulated_series):
    return compute_moments(simulated_series, output_key="y", apply_hp=False)


# ──────────────────────────────────────────────────────────────────────────────
# simulate.py tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulate:
    def test_output_shape(self, toy_AB):
        A, B = toy_AB
        X = simulate(A, B, T=1000, seed=42)
        assert X.shape == (1000, N_VARS)

    def test_reproducibility(self, toy_AB):
        A, B = toy_AB
        X1 = simulate(A, B, T=1000, seed=42)
        X2 = simulate(A, B, T=1000, seed=42)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds(self, toy_AB):
        A, B = toy_AB
        X1 = simulate(A, B, T=1000, seed=42)
        X2 = simulate(A, B, T=1000, seed=99)
        assert not np.allclose(X1, X2)

    def test_extract_observables_keys(self, simulated_series):
        assert set(simulated_series.keys()) == {"y", "c", "i", "l", "pi"}

    def test_extract_observables_lengths(self, simulated_series):
        for arr in simulated_series.values():
            assert len(arr) == 1000


# ──────────────────────────────────────────────────────────────────────────────
# moments.py – structure tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMomentsStructure:
    EXPECTED_KEYS_PER_VAR = ["var", "corr", "ac1"]
    VARS = ["y", "c", "i", "l", "pi"]

    def test_all_moment_keys_present(self, raw_moments):
        for prefix in self.EXPECTED_KEYS_PER_VAR:
            for var in self.VARS:
                key = f"{prefix}_{var}"
                assert key in raw_moments, f"Missing key: {key}"

    def test_minimum_10_moments(self, raw_moments):
        assert len(raw_moments) >= 10

    def test_no_nan_in_raw_moments(self, raw_moments):
        for k, v in raw_moments.items():
            assert not np.isnan(v), f"NaN found in moment: {k}"

    def test_variances_positive(self, raw_moments):
        for var in self.VARS:
            assert raw_moments[f"var_{var}"] > 0

    def test_output_self_correlation_is_one(self, raw_moments):
        assert abs(raw_moments["corr_y"] - 1.0) < 1e-10

    def test_autocorrelations_in_range(self, raw_moments):
        for var in self.VARS:
            ac1 = raw_moments[f"ac1_{var}"]
            assert -1.0 <= ac1 <= 1.0

    def test_correlations_in_range(self, raw_moments):
        for var in self.VARS:
            corr = raw_moments[f"corr_{var}"]
            assert -1.0 <= corr <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# moments.py – reproducibility
# ──────────────────────────────────────────────────────────────────────────────

class TestMomentsReproducibility:
    def test_same_seed_identical_moments(self, toy_AB, toy_obs_index):
        A, B = toy_AB
        s1 = run_simulation(A, B, T=1000, seed=42, obs_index=toy_obs_index)
        s2 = run_simulation(A, B, T=1000, seed=42, obs_index=toy_obs_index)
        m1 = compute_moments(s1)
        m2 = compute_moments(s2)
        for k in m1:
            assert m1[k] == m2[k]


# ──────────────────────────────────────────────────────────────────────────────
# moments.py – economic sanity
# ──────────────────────────────────────────────────────────────────────────────

class TestEconomicSanity:
    def test_consumption_smoother_than_output(self, raw_moments):
        assert consumption_smoother_than_output(raw_moments)

    def test_investment_more_volatile_than_output(self, raw_moments):
        assert investment_more_volatile_than_output(raw_moments)


# ──────────────────────────────────────────────────────────────────────────────
# HP filter
# ──────────────────────────────────────────────────────────────────────────────

class TestHPFilter:
    def test_hp_cycle_length(self, simulated_series):
        y = simulated_series["y"]
        cycle = hp_cycle(y, lamb=1600)
        assert len(cycle) == len(y)

    def test_hp_moments_returned(self, simulated_series):
        raw, hp = compute_all_moments(simulated_series)
        assert set(raw.keys()) == set(hp.keys())

    def test_hp_and_raw_differ(self, simulated_series):
        raw, hp = compute_all_moments(simulated_series)
        assert raw["var_y"] != hp["var_y"]
