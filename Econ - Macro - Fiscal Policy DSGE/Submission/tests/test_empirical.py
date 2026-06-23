"""
tests/test_empirical.py
Step 8b - Empirical moments tests
------------------------------------
Owner: Steps 8-9 contributor

FRED network calls are mocked so tests pass offline / in CI without an API key.
"""

import numpy as np
import pandas as pd
import pytest
import time
from unittest.mock import patch

from simulation.empirical import (
    build_comparison_table,
    load_empirical_moments,
    _transform,
    CACHE_PATH,
    FRED_SERIES,
)
from simulation.moments import compute_moments, hp_cycle


def _make_fake_fred_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("1993-01-01", periods=n, freq="QS")
    data = {}
    for ticker in FRED_SERIES.values():
        log_level = np.cumsum(rng.normal(0.005, 0.01, n))
        data[ticker] = np.exp(log_level) * 10_000
    return pd.DataFrame(data, index=dates)


FAKE_DF = _make_fake_fred_df()


def _hp_and_moments(raw_dict):
    series_hp = {k: hp_cycle(v, lamb=1600.0) for k, v in raw_dict.items()}
    return compute_moments(series_hp, output_key="y", apply_hp=False)


class TestTransform:
    def test_keys_match_model_names(self):
        result = _transform(FAKE_DF)
        assert set(result.keys()) == set(FRED_SERIES.keys())

    def test_output_arrays_are_finite(self):
        result = _transform(FAKE_DF)
        for k, v in result.items():
            assert np.all(np.isfinite(v)), f"Non-finite values in '{k}'"

    def test_lengths_aligned(self):
        result = _transform(FAKE_DF)
        lengths = [len(v) for v in result.values()]
        assert len(set(lengths)) == 1


class TestHpAndMoments:
    @pytest.fixture
    def raw_dict(self):
        return _transform(FAKE_DF)

    def test_returns_dict(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        assert isinstance(m, dict)

    def test_has_15_moments(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        assert len(m) == 15

    def test_all_finite(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        for k, v in m.items():
            assert np.isfinite(v), f"Non-finite moment: {k}={v}"

    def test_variances_positive(self, raw_dict):
        m = _hp_and_moments(raw_dict)
        for var in FRED_SERIES:
            assert m[f"var_{var}"] > 0


class TestLoadEmpiricalMoments:
    def test_cache_path_is_module_relative(self):
        import simulation.empirical as emp_mod
        assert CACHE_PATH == emp_mod.Path(emp_mod.__file__).parent / "_fred_cache.csv"

    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_returns_moments_and_df(self, mock_fetch, tmp_path, monkeypatch):
        import simulation.empirical as emp_mod
        monkeypatch.setattr(emp_mod, "CACHE_PATH", tmp_path / "_fred_cache.csv")
        moments, df = load_empirical_moments(force_refresh=True)
        assert isinstance(moments, dict)
        assert isinstance(df, pd.DataFrame)

    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_moment_keys_present(self, mock_fetch, tmp_path, monkeypatch):
        import simulation.empirical as emp_mod
        monkeypatch.setattr(emp_mod, "CACHE_PATH", tmp_path / "_fred_cache.csv")
        moments, _ = load_empirical_moments(force_refresh=True)
        for var in FRED_SERIES:
            for prefix in ["var", "corr", "ac1"]:
                key = f"{prefix}_{var}"
                assert key in moments, f"Missing: {key}"

    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_cache_used_on_second_call(self, mock_fetch, tmp_path, monkeypatch):
        import simulation.empirical as emp_mod
        monkeypatch.setattr(emp_mod, "CACHE_PATH", tmp_path / "_fred_cache.csv")
        load_empirical_moments(force_refresh=True)
        load_empirical_moments(force_refresh=False)
        assert mock_fetch.call_count == 1

    @patch("simulation.empirical._fetch_from_fred", return_value=FAKE_DF)
    def test_stale_cache_refreshes(self, mock_fetch, tmp_path, monkeypatch):
        import simulation.empirical as emp_mod
        cache_path = tmp_path / "_fred_cache.csv"
        monkeypatch.setattr(emp_mod, "CACHE_PATH", cache_path)
        load_empirical_moments(force_refresh=True)
        old_time = time.time() - 3 * 24 * 60 * 60
        cache_path.touch()
        import os
        os.utime(cache_path, (old_time, old_time))
        load_empirical_moments(force_refresh=False, cache_ttl_seconds=24 * 60 * 60)
        assert mock_fetch.call_count == 2


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
        assert all(abs(r - 1.0) < 1e-6 for r in df["Ratio"].dropna())
