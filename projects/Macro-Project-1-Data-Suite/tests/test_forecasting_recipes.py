from __future__ import annotations

import pytest
import pandas as pd

from finrec.recipes.forecast.naive import DriftForecastRecipe, NaiveForecastRecipe


class _Ctx:
    def log(self, level: str, message: str) -> None:
        pass


def _make_daily_df(n: int = 120) -> pd.DataFrame:
    d0 = pd.Timestamp("2020-01-01")
    dates = [(d0 + pd.Timedelta(days=i)).date().isoformat() for i in range(n)]
    # smooth-ish increasing series
    close = [100.0 + 0.5 * i + 2.0 * (i % 7) for i in range(n)]
    return pd.DataFrame({"date": dates, "close": close})


def _assert_forecast_schema(df: pd.DataFrame) -> None:
    for c in ["date", "y", "yhat", "segment", "method", "target"]:
        assert c in df.columns
    assert set(df["segment"].unique()).issubset({"train", "test", "forecast"})


def test_naive_forecast_shapes():
    df = _make_daily_df(120)
    out = NaiveForecastRecipe().run(
        df,
        params={"date_col": "date", "y_col": "close", "transform": "none", "horizon": 10, "test_size": 0.25},
        ctx=_Ctx(),
    )
    _assert_forecast_schema(out)
    assert (out["segment"] == "train").sum() == 90
    assert (out["segment"] == "test").sum() == 30
    assert (out["segment"] == "forecast").sum() == 10

    test_rows = out[out["segment"] == "test"]
    assert test_rows["y"].notna().all()
    assert test_rows["yhat"].notna().all()

    fc_rows = out[out["segment"] == "forecast"]
    assert fc_rows["y"].isna().all()
    assert fc_rows["yhat"].notna().all()


def test_drift_forecast_runs():
    df = _make_daily_df(120)
    out = DriftForecastRecipe().run(
        df,
        params={"date_col": "date", "y_col": "close", "transform": "none", "horizon": 5, "test_size": 0.2},
        ctx=_Ctx(),
    )
    _assert_forecast_schema(out)
    assert (out["segment"] == "forecast").sum() == 5


def test_naive_log_return_target():
    df = _make_daily_df(120)
    out = NaiveForecastRecipe().run(
        df,
        params={"date_col": "date", "y_col": "close", "transform": "log_return", "horizon": 3, "test_size": 0.2},
        ctx=_Ctx(),
    )
    _assert_forecast_schema(out)
    assert out["target"].astype(str).str.contains("log_return").any()


statsmodels = pytest.importorskip("statsmodels")


def test_ets_forecast_runs():
    from finrec.recipes.forecast.ets import ETSForecastRecipe

    df = _make_daily_df(180)
    out = ETSForecastRecipe().run(
        df,
        params={"date_col": "date", "y_col": "close", "transform": "none", "horizon": 7, "test_size": 0.2},
        ctx=_Ctx(),
    )
    _assert_forecast_schema(out)
    assert (out["segment"] == "forecast").sum() == 7


def test_arima_forecast_runs():
    from finrec.recipes.forecast.arima import ARIMAForecastRecipe

    df = _make_daily_df(220)
    out = ARIMAForecastRecipe().run(
        df,
        params={"date_col": "date", "y_col": "close", "transform": "none", "horizon": 5, "test_size": 0.2},
        ctx=_Ctx(),
    )
    _assert_forecast_schema(out)
    assert (out["segment"] == "forecast").sum() == 5
    assert out["method"].astype(str).str.contains("arima").any()


sklearn = pytest.importorskip("sklearn")


def test_ridge_lags_forecast_runs():
    from finrec.recipes.forecast.ml_lags import RidgeLagForecastRecipe

    df = _make_daily_df(180)
    out = RidgeLagForecastRecipe().run(
        df,
        params={
            "date_col": "date",
            "y_col": "close",
            "transform": "none",
            "horizon": 6,
            "test_size": 0.2,
            "lookback": 14,
        },
        ctx=_Ctx(),
    )
    _assert_forecast_schema(out)
    assert (out["segment"] == "forecast").sum() == 6

