"""
tests/test_moments.py
Tests for simulation/simulate.py and simulation/moments.py (Step 8)
"""

import numpy as np
import pytest

# Make imports work when running from repo root
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from simulation.simulate import simulate, extract_observables, run_simulation
from simulation.moments import (
    compute_moments,
    compute_all_moments,
    hp_cycle,
    consumption_smoother_than_output,
    investment_more_volatile_than_output,
)


# ---------------------------------------------------------------------------
# Fixtures – simple 5-variable AR(1) toy model
# ---------------------------------------------------------------------------

N_VARS = 5   # y, c, i, l, pi

@pytest.fixture
def toy_AB():
    """
    Toy A and B matrices for a 5-variable AR(1) system.
    Each variable follows x_t = 0.8 x_{t-1} + sigma * eps_t  independently.
    Consumption (col 1) has lower shock variance → smoother than output.
    Investment (col 2) has higher shock variance → more volatile than output.
    """
    rho   = 0.8
    A     = rho * np.eye(N_VARS)
    sigma = np.array([1.0, 0.5, 3.0, 0.8, 0.6])   # y, c, i, l, pi
    B     = np.diag(sigma)
    return A, B


@pytest.fixture
def simulated_series(toy_AB):
    A, B = toy_AB
    return run_simulation(A, B, T=1000, seed=42)


@pytest.fixture
def raw_moments(simulated_series):
    return compute_moments(simulated_series, output_key="y", apply_hp=False)


# ---------------------------------------------------------------------------
# simulate.py tests
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_output_shape(self, toy_AB):
        A, B = toy_AB
        X = simulate(A, B, T=1000, seed=42)
        assert X.shape == (1000, N_VARS)

    def test_reproducibility(self, toy_AB):
        """Same seed → identical output."""
        A, B = toy_AB
        X1 = simulate(A, B, T=1000, seed=42)
        X2 = simulate(A, B, T=1000, seed=42)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds(self, toy_AB):
        """Different seeds → different output."""
        A, B = toy_AB
        X1 = simulate(A, B, T=1000, seed=42)
        X2 = simulate(A, B, T=1000, seed=99)
        assert not np.allclose(X1, X2)

    def test_extract_observables_keys(self, simulated_series):
        assert set(simulated_series.keys()) == {"y", "c", "i", "l", "pi"}

    def test_extract_observables_lengths(self, simulated_series):
        for arr in simulated_series.values():
            assert len(arr) == 1000

    def test_custom_obs_index(self, toy_AB):
        A, B = toy_AB
        X = simulate(A, B, T=500, seed=1)
        custom = {"output": 0, "cons": 1}
        obs = extract_observables(X, obs_index=custom)
        assert set(obs.keys()) == {"output", "cons"}
        np.testing.assert_array_equal(obs["output"], X[:, 0])


# ---------------------------------------------------------------------------
# moments.py – structure tests
# ---------------------------------------------------------------------------

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
            assert -1.0 <= ac1 <= 1.0, f"ac1_{var} = {ac1} out of [-1,1]"

    def test_correlations_in_range(self, raw_moments):
        for var in self.VARS:
            corr = raw_moments[f"corr_{var}"]
            assert -1.0 <= corr <= 1.0, f"corr_{var} = {corr} out of [-1,1]"


# ---------------------------------------------------------------------------
# moments.py – reproducibility
# ---------------------------------------------------------------------------

class TestMomentsReproducibility:
    def test_same_seed_identical_moments(self, toy_AB):
        A, B = toy_AB
        s1 = run_simulation(A, B, T=1000, seed=42)
        s2 = run_simulation(A, B, T=1000, seed=42)
        m1 = compute_moments(s1)
        m2 = compute_moments(s2)
        for k in m1:
            assert m1[k] == m2[k], f"Moment {k} differs across identical seeds"

    def test_different_seed_different_moments(self, toy_AB):
        A, B = toy_AB
        s1 = run_simulation(A, B, T=1000, seed=42)
        s2 = run_simulation(A, B, T=1000, seed=99)
        m1 = compute_moments(s1)
        m2 = compute_moments(s2)
        # At least one moment should differ
        diffs = [abs(m1[k] - m2[k]) for k in m1 if not np.isnan(m1[k])]
        assert any(d > 1e-10 for d in diffs)


# ---------------------------------------------------------------------------
# moments.py – economic sanity checks
# ---------------------------------------------------------------------------

class TestEconomicSanity:
    def test_consumption_smoother_than_output(self, raw_moments):
        """
        In our toy model, consumption has lower shock std (0.5 vs 1.0 for output)
        so Var(c) < Var(y) must hold.
        """
        assert consumption_smoother_than_output(raw_moments), (
            f"Expected Var(c) < Var(y), got var_c={raw_moments['var_c']:.6f}, "
            f"var_y={raw_moments['var_y']:.6f}"
        )

    def test_investment_more_volatile_than_output(self, raw_moments):
        """
        Investment shock std is 3.0 vs 1.0 for output → Var(i) > Var(y).
        """
        assert investment_more_volatile_than_output(raw_moments), (
            f"Expected Var(i) > Var(y), got var_i={raw_moments['var_i']:.6f}, "
            f"var_y={raw_moments['var_y']:.6f}"
        )

    def test_hours_procyclical(self, raw_moments):
        """Hours should be positively correlated with output."""
        assert raw_moments["corr_l"] > 0, (
            f"Expected corr_l > 0, got {raw_moments['corr_l']:.4f}"
        )


# ---------------------------------------------------------------------------
# HP filter
# ---------------------------------------------------------------------------

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
        # HP-filtered variances should generally be smaller
        assert raw["var_y"] != hp["var_y"]

    def test_hp_variances_smaller_or_equal(self, simulated_series):
        """HP filter removes trend → HP-filtered variance ≤ raw variance."""
        raw, hp = compute_all_moments(simulated_series)
        for var in ["y", "c", "i", "l", "pi"]:
            assert hp[f"var_{var}"] <= raw[f"var_{var}"] + 1e-10, (
                f"HP var_{var} ({hp[f'var_{var}']:.6f}) > "
                f"raw var_{var} ({raw[f'var_{var}']:.6f})"
            )
