from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta
from finrec.recipes.forecast._utils import (
    TargetTransform,
    _infer_step,
    _make_future_dates,
    build_forecast_output,
    build_target_series,
    split_train_test,
)


@dataclass
class NaiveForecastRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="forecast_naive",
        name="Forecast: Naive",
        description="Forecasts future values as the last observed training value (baseline).",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        date_col = str(params.get("date_col", "date"))
        y_col = str(params.get("y_col", "value"))
        transform: TargetTransform = str(params.get("transform", "none"))  # type: ignore[assignment]
        horizon = int(params.get("horizon", 12))
        test_size = params.get("test_size", 0.2)

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. date_col={date_col}, y_col={y_col}, transform={transform}, "
            f"horizon={horizon}, test_size={test_size}",
        )

        frame = build_target_series(df, date_col=date_col, y_col=y_col, transform=transform)
        dates, y, target = frame.date_index, frame.y, frame.target
        train_dates, train_y, test_dates, test_y = split_train_test(dates, y, test_size=test_size)

        if train_y.dropna().empty:
            raise ValueError("Training series is empty after preprocessing.")

        last = float(train_y.dropna().iloc[-1])
        n_test = len(test_y)
        test_yhat = [last] * n_test

        step = _infer_step(train_dates)
        future_dates = _make_future_dates(pd.Timestamp(train_dates[-1]), horizon=horizon, step=step)
        fcst_yhat = [last] * len(future_dates)

        out = build_forecast_output(
            train_dates=list(train_dates),
            train_y=[float(v) if pd.notna(v) else None for v in train_y.tolist()],
            test_dates=list(test_dates),
            test_y=[float(v) if pd.notna(v) else None for v in test_y.tolist()],
            test_yhat=test_yhat,
            forecast_dates=future_dates,
            forecast_yhat=fcst_yhat,
            method="naive",
            target=target,
        )

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out


@dataclass
class DriftForecastRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="forecast_drift",
        name="Forecast: Drift",
        description="Forecasts using a linear drift from first to last training observation (baseline).",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        date_col = str(params.get("date_col", "date"))
        y_col = str(params.get("y_col", "value"))
        transform: TargetTransform = str(params.get("transform", "none"))  # type: ignore[assignment]
        horizon = int(params.get("horizon", 12))
        test_size = params.get("test_size", 0.2)

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. date_col={date_col}, y_col={y_col}, transform={transform}, "
            f"horizon={horizon}, test_size={test_size}",
        )

        frame = build_target_series(df, date_col=date_col, y_col=y_col, transform=transform)
        dates, y, target = frame.date_index, frame.y, frame.target
        train_dates, train_y, test_dates, test_y = split_train_test(dates, y, test_size=test_size)

        train_y2 = train_y.dropna()
        if len(train_y2) < 2:
            raise ValueError("Need at least 2 training observations for drift forecast.")

        y0 = float(train_y2.iloc[0])
        yT = float(train_y2.iloc[-1])
        n_train = len(train_y2)
        drift = (yT - y0) / max(1, (n_train - 1))

        # Out-of-sample predictions from end of train.
        def _path(steps: int) -> list[float]:
            return [yT + drift * (k + 1) for k in range(steps)]

        test_yhat = _path(len(test_y))

        step = _infer_step(train_dates)
        future_dates = _make_future_dates(pd.Timestamp(train_dates[-1]), horizon=horizon, step=step)
        fcst_yhat = _path(len(future_dates))

        out = build_forecast_output(
            train_dates=list(train_dates),
            train_y=[float(v) if pd.notna(v) else None for v in train_y.tolist()],
            test_dates=list(test_dates),
            test_y=[float(v) if pd.notna(v) else None for v in test_y.tolist()],
            test_yhat=test_yhat,
            forecast_dates=future_dates,
            forecast_yhat=fcst_yhat,
            method="drift",
            target=target,
        )

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out

