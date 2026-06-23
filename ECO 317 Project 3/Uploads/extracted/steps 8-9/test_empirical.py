"""
tests/test_empirical.py
Tests for simulation/empirical.py (Step 8b)

FRED network calls are mocked so tests pass offline / in CI without an API key.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from simulation.empirical import (
    build_comparison_table,
    load_empirical_moments,
    _transform,
    _hp_and_moments,
    FRED_SERIES,
)


# ---------------------------------------------------------------------------
# Helpers – synthetic FRED-like DataFrame
# ---------------------------------------------------------------------------

def _make_fake_fred_df(n: int = 120) -> pd.DataFrame:
    """
    120 quarterly obs (≈30 years) of synthetic level data.
    All series are random walks in log-space to mimic real macro data.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("1993-01-01", periods=n, freq="QS")
    data = {}
    for ticker in FRED_SERIES.values():
        log_level = np.cumsum(rng.normal(0.005, 0.01, n))  # upward drift
        data[ticker] = np.exp(log_level) * 10_000           # levels
    return pd.DataFrame(data, index=dates)


FAKE_DF = _make_fake_fred_df()


# ---------------------------------------------------------------------------
# _transform
# ---------------------------------------------------------------------------

class TestTransform:
    def test_keys_match_model_names(self):
        result = _transform(FAKE_DF)
        assert set(result.keys()) == set(FRED_SERIES.keys())

    def test_pi_is_shorter_by_one(self):
        """CPI is first-differenced → one observation shorter."""
        result = _transform(FAKE_DF)
        n_y  = len(FAKE_DF[FRED_SERIES["y"]].dropna())
        n_pi = len(result["pi"])
        # After alignment to min length they should be equal, but pi starts 1 shorter
        assert n_pi <= n_y

    def test_output_arrays_are_finite(self):
        result = _transform(FAKE_DF)
        for k, v in result.items():
            assert np.all(np.isfinite(v)), f"Non-finite values in series '{k}'"

    def test_lengths_aligned(self):
        result = _transform(FAKE_DF)
        lengths = [len(v) for v in result.values()]
        assert len(set(lengths)) == 1, f"Lengths not aligned: {lengths}"


# ---------------------------------------------------------------------------
# _hp_and_moments
# ---------------------------------------------------------------------------

class TestHpAndMoments:
    @pytest.fixture
    def raw_dict(self):
        return _transform(FAKE_DF)

    def test_returns_dict(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        assert isinstance(m, dict)

    def test_has_15_moments(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        # 5 vars × 3 moment types = 15
        assert len(m) == 15

    def test_all_finite(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        for k, v in m.items():
            assert np.isfinite(v), f"Non-finite empirical moment: {k}={v}"

    def test_output_self_corr_is_one(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        assert abs(m["corr_y"] - 1.0) < 1e-10

    def test_variances_positive(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        for var in FRED_SERIES:
            assert m[f"var_{var}"] > 0


# ---------------------------------------------------------------------------
# load_empirical_moments  (mocked FRED pull)
# ---------------------------------------------------------------------------

class TestLoadEmpiricalMoments:
    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_returns_moments_and_df(self, mock_fetch, tmp_path, monkeypatch):
        # Redirect cache to tmp dir
        import simulation.empirical as emp_mod
        monkeypatch.setattr(emp_mod, "CACHE_PATH", tmp_path / "_fred_cache.parquet")

        moments, df = load_empirical_moments(force_refresh=True)
        assert isinstance(moments, dict)
        assert isinstance(df, pd.DataFrame)

    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_moment_keys_present(self, mock_fetch, tmp_path, monkeypatch):
        import simulation.empirical as emp_mod
        monkeypatch.setattr(emp_mod, "CACHE_PATH", tmp_path / "_fred_cache.parquet")

        moments, _ = load_empirical_moments(force_refresh=True)
        for var in FRED_SERIES:
            for prefix in ["var", "corr", "ac1"]:
                key = f"{prefix}_{var}"
                assert key in moments, f"Missing empirical moment: {key}"

    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_cache_used_on_second_call(self, mock_fetch, tmp_path, monkeypatch):
        import simulation.empirical as emp_mod
        monkeypatch.setattr(emp_mod, "CACHE_PATH", tmp_path / "_fred_cache.parquet")

        load_empirical_moments(force_refresh=True)   # first call: fetch
        load_empirical_moments(force_refresh=False)  # second call: from cache

        assert mock_fetch.call_count == 1            # FRED hit only once

    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_df_has_correct_columns(self, mock_fetch, tmp_path, monkeypatch):
        import simulation.empirical as emp_mod
        monkeypatch.setattr(emp_mod, "CACHE_PATH", tmp_path / "_fred_cache.parquet")

        _, df = load_empirical_moments(force_refresh=True)
        assert set(df.columns) == set(FRED_SERIES.keys())


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    @pytest.fixture
    def dummy_moments(self):
        return {
            "var_y": 0.002, "var_c": 0.001, "var_i": 0.008, "var_l": 0.003, "var_pi": 0.0005,
            "corr_y": 1.0,  "corr_c": 0.8,  "corr_i": 0.9,  "corr_l": 0.7,  "corr_pi": 0.4,
            "ac1_y": 0.7,   "ac1_c": 0.6,   "ac1_i": 0.5,   "ac1_l": 0.65,  "ac1_pi": 0.4,
        }

    def test_returns_dataframe(self, dummy_moments):
        df = build_comparison_table(dummy_moments, dummy_moments)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, dummy_moments):
        df = build_comparison_table(dummy_moments, dummy_moments)
        for col in ["Model", "Empirical", "Difference", "Ratio"]:
            assert col in df.columns

    def test_same_moments_ratio_is_one(self, dummy_moments):
        df = build_comparison_table(dummy_moments, dummy_moments)
        # When model == empirical, all ratios should be 1.0
        assert all(abs(r - 1.0) < 1e-6 for r in df["Ratio"].dropna())

    def test_difference_column_is_zero_when_equal(self, dummy_moments):
        df = build_comparison_table(dummy_moments, dummy_moments)
        assert all(abs(d) < 1e-10 for d in df["Difference"].dropna())

    def test_handles_mismatched_keys(self, dummy_moments):
        partial = {k: v for k, v in dummy_moments.items() if "y" in k}
        df = build_comparison_table(dummy_moments, partial)
        # Missing empirical keys → NaN in Empirical column but no crash
        assert df is not None

    def test_row_count(self, dummy_moments):
        df = build_comparison_table(dummy_moments, dummy_moments)
        assert len(df) == len(dummy_moments)
