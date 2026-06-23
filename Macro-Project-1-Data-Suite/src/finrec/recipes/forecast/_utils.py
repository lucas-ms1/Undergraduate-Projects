from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Tuple

import pandas as pd

TargetTransform = Literal["none", "log_return"]


@dataclass(frozen=True)
class ForecastFrame:
    df: pd.DataFrame
    date_index: pd.DatetimeIndex
    y: pd.Series
    target: str


def _parse_dates(df: pd.DataFrame, *, date_col: str) -> pd.Series:
    if date_col not in df.columns:
        raise ValueError(f"Expected date_col '{date_col}' not found. Columns: {list(df.columns)}")

    d = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if d.notna().sum() == 0:
        raise ValueError(f"Could not parse any valid datetimes from '{date_col}'.")

    # Drop timezone for simpler date arithmetic (we only display dates).
    return d.dt.tz_convert("UTC").dt.tz_localize(None)


def _infer_step(dates: pd.DatetimeIndex) -> pd.DateOffset | pd.Timedelta:
    """
    Infer a plausible step size for future date generation from median spacing.
    Returns a DateOffset for month/quarter/year-like cadence when detected.
    """
    if len(dates) < 2:
        return pd.Timedelta(days=1)

    ds = pd.Series(dates).sort_values().diff().dropna()
    if ds.empty:
        return pd.Timedelta(days=1)
    median_days = float(ds.median() / pd.Timedelta(days=1))
    if not math.isfinite(median_days) or median_days <= 0:
        return pd.Timedelta(days=1)

    # Heuristics for common macro cadences.
    if 27 <= median_days <= 31:
        return pd.DateOffset(months=1)
    if 85 <= median_days <= 95:
        return pd.DateOffset(months=3)
    if 360 <= median_days <= 370:
        return pd.DateOffset(years=1)
    if 6 <= median_days <= 8:
        return pd.Timedelta(days=7)

    return pd.Timedelta(days=max(1, int(round(median_days))))


def _make_future_dates(
    last_date: pd.Timestamp, *, horizon: int, step: pd.DateOffset | pd.Timedelta
) -> list[pd.Timestamp]:
    if horizon <= 0:
        return []
    out: list[pd.Timestamp] = []
    cur = pd.Timestamp(last_date)
    for _ in range(horizon):
        cur = cur + step
        out.append(pd.Timestamp(cur))
    return out


def _coerce_numeric(series: pd.Series, *, name: str) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    if y.isna().all():
        raise ValueError(f"Column '{name}' could not be parsed as numeric.")
    return y


def build_target_series(
    df: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    transform: TargetTransform = "none",
) -> ForecastFrame:
    """
    Build (date_index, y) from a dataframe, optionally transforming to log returns.
    """
    df2 = df.copy()
    df2["_dt"] = _parse_dates(df2, date_col=date_col)
    df2 = df2.dropna(subset=["_dt"]).sort_values("_dt").reset_index(drop=True)

    if y_col not in df2.columns:
        raise ValueError(f"Expected y_col '{y_col}' not found. Columns: {list(df2.columns)}")

    y_raw = _coerce_numeric(df2[y_col], name=y_col)

    if transform == "none":
        y = y_raw
        target = y_col
    elif transform == "log_return":
        # log return = log(p_t) - log(p_{t-1})
        if (y_raw <= 0).any():
            # Nonpositive values cannot be logged; they become NaN.
            y_log = y_raw.where(y_raw > 0).apply(lambda x: math.log(x) if pd.notna(x) else float("nan"))
        else:
            y_log = y_raw.apply(lambda x: math.log(x) if pd.notna(x) else float("nan"))
        y = y_log.diff()
        target = f"log_return({y_col})"
    else:  # pragma: no cover
        raise ValueError(f"Unknown transform: {transform}")

    out = pd.DataFrame({"date": df2["_dt"].dt.date.astype(str), "y": y}).reset_index(drop=True)
    date_index = pd.DatetimeIndex(df2["_dt"])
    y2 = pd.to_numeric(out["y"], errors="coerce")

    return ForecastFrame(df=out, date_index=date_index, y=y2, target=target)


def split_train_test(
    dates: pd.DatetimeIndex, y: pd.Series, *, test_size: int | float
) -> Tuple[pd.DatetimeIndex, pd.Series, pd.DatetimeIndex, pd.Series]:
    n = len(y)
    if n != len(dates):
        raise ValueError("Internal error: dates and y length mismatch.")
    if n < 10:
        raise ValueError(f"Need at least 10 observations; got n={n}")

    if isinstance(test_size, float):
        if not (0.0 <= test_size < 1.0):
            raise ValueError("test_size as float must be in [0.0, 1.0).")
        n_test = int(math.floor(n * test_size))
    else:
        n_test = int(test_size)

    n_test = max(0, min(n_test, n - 5))  # leave at least 5 in train
    if n_test == 0:
        train_dates, train_y = dates, y
        test_dates, test_y = dates[:0], y.iloc[:0]
    else:
        train_dates, train_y = dates[:-n_test], y.iloc[:-n_test]
        test_dates, test_y = dates[-n_test:], y.iloc[-n_test:]

    return train_dates, train_y, test_dates, test_y


def build_forecast_output(
    *,
    train_dates: Iterable[pd.Timestamp],
    train_y: Iterable[float | int | None],
    test_yhat: Iterable[float | int | None],
    test_dates: Iterable[pd.Timestamp],
    test_y: Iterable[float | int | None],
    forecast_dates: Iterable[pd.Timestamp],
    forecast_yhat: Iterable[float | int | None],
    method: str,
    target: str,
    test_lower: Iterable[float | int | None] | None = None,
    test_upper: Iterable[float | int | None] | None = None,
    fcst_lower: Iterable[float | int | None] | None = None,
    fcst_upper: Iterable[float | int | None] | None = None,
) -> pd.DataFrame:
    """
    Standardized output schema used by all forecast recipes.
    """
    train_dates = list(train_dates)
    train_y = list(train_y)
    test_dates = list(test_dates)
    test_y = list(test_y)
    test_yhat = list(test_yhat)
    forecast_dates = list(forecast_dates)
    forecast_yhat = list(forecast_yhat)

    if len(train_dates) != len(train_y):
        raise ValueError("Internal error: train_dates/train_y length mismatch.")
    if len(test_dates) != len(test_y):
        raise ValueError("Internal error: test_dates/test_y length mismatch.")
    if len(test_dates) != len(test_yhat):
        raise ValueError("Internal error: test_dates/test_yhat length mismatch.")
    if len(forecast_dates) != len(forecast_yhat):
        raise ValueError("Internal error: forecast_dates/forecast_yhat length mismatch.")

    rows: list[dict] = []
    for dt, y in zip(train_dates, train_y):
        rows.append(
            {
                "date": pd.Timestamp(dt).date().isoformat(),
                "y": y,
                "yhat": None,
                "segment": "train",
                "method": method,
                "target": target,
            }
        )

    test_lower_list = list(test_lower) if test_lower is not None else None
    test_upper_list = list(test_upper) if test_upper is not None else None
    fcst_lower_list = list(fcst_lower) if fcst_lower is not None else None
    fcst_upper_list = list(fcst_upper) if fcst_upper is not None else None

    for i, (dt, y_true, yhat) in enumerate(zip(test_dates, test_y, test_yhat)):
        row = {
            "date": pd.Timestamp(dt).date().isoformat(),
            "y": y_true,
            "yhat": yhat,
            "segment": "test",
            "method": method,
            "target": target,
        }
        if test_lower_list is not None and test_upper_list is not None:
            lo = test_lower_list[i]
            hi = test_upper_list[i]
            row["yhat_lower"] = lo
            row["yhat_upper"] = hi
        rows.append(row)

    for i, (dt, yhat) in enumerate(zip(forecast_dates, forecast_yhat)):
        row = {
            "date": pd.Timestamp(dt).date().isoformat(),
            "y": None,
            "yhat": yhat,
            "segment": "forecast",
            "method": method,
            "target": target,
        }
        if fcst_lower_list is not None and fcst_upper_list is not None:
            lo = fcst_lower_list[i]
            hi = fcst_upper_list[i]
            row["yhat_lower"] = lo
            row["yhat_upper"] = hi
        rows.append(row)

    out = pd.DataFrame(rows)
    # Ensure stable ordering by date, then segment order.
    seg_order = {"train": 0, "test": 1, "forecast": 2}
    out["_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out["_seg"] = out["segment"].map(seg_order).fillna(99)
    out = out.sort_values(["_dt", "_seg"]).drop(columns=["_dt", "_seg"]).reset_index(drop=True)
    return out

