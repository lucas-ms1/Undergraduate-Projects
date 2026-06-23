"""
Tests for simulation.moments – verifies computed statistics
match direct numpy calculations from the same arrays.
"""

import numpy as np
import pytest

from simulation.moments import compute_moments


@pytest.fixture
def sample_series():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    c = 0.5 * y + rng.standard_normal(200) * 0.1
    return {"y": y, "consumption": c}


class TestMoments:
    def test_mean_matches_numpy(self, sample_series):
        m = compute_moments(sample_series, income_key="y")
        np.testing.assert_allclose(
            m["y"]["mean"], np.mean(sample_series["y"]), atol=1e-10
        )

    def test_variance_matches_numpy(self, sample_series):
        m = compute_moments(sample_series, income_key="y")
        np.testing.assert_allclose(
            m["y"]["variance"], np.var(sample_series["y"]), atol=1e-10
        )

    def test_autocorrelation_finite(self, sample_series):
        m = compute_moments(sample_series, income_key="y")
        assert np.isfinite(m["y"]["autocorrelation"])

    def test_income_self_correlation_is_one(self, sample_series):
        m = compute_moments(sample_series, income_key="y")
        np.testing.assert_allclose(m["y"]["corr_with_income"], 1.0, atol=1e-10)

    def test_consumption_correlated_with_income(self, sample_series):
        m = compute_moments(sample_series, income_key="y")
        assert m["consumption"]["corr_with_income"] > 0.5

    def test_all_keys_present(self, sample_series):
        m = compute_moments(sample_series, income_key="y")
        for var in ("y", "consumption"):
            for key in ("mean", "variance", "autocorrelation", "corr_with_income"):
                assert key in m[var]
