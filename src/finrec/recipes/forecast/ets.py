from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.providers.utils.optional import require_optional
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
class ETSForecastRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="forecast_ets",
        name="Forecast: ETS",
        description="Exponential smoothing (Holt-Winters). Forecasts using level/trend/seasonality if configured.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        sm_hw = require_optional("statsmodels.tsa.holtwinters", extra_hint="forecast")

        date_col = str(params.get("date_col", "date"))
        y_col = str(params.get("y_col", "value"))
        transform: TargetTransform = str(params.get("transform", "none"))  # type: ignore[assignment]
        horizon = int(params.get("horizon", 12))
        test_size = params.get("test_size", 0.2)

        trend = params.get("trend", "add")
        seasonal = params.get("seasonal", None)
        seasonal_periods = params.get("seasonal_periods", None)
        if seasonal_periods is not None:
            seasonal_periods = int(seasonal_periods)

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. date_col={date_col}, y_col={y_col}, transform={transform}, "
            f"horizon={horizon}, test_size={test_size}, trend={trend}, seasonal={seasonal}, "
            f"seasonal_periods={seasonal_periods}",
        )

        frame = build_target_series(df, date_col=date_col, y_col=y_col, transform=transform)
        dates, y, target = frame.date_index, frame.y, frame.target
        train_dates, train_y, test_dates, test_y = split_train_test(dates, y, test_size=test_size)

        y_train = train_y.dropna().astype(float)
        if len(y_train) < 10:
            raise ValueError(f"Need at least 10 non-NaN training observations; got n={len(y_train)}")

        # ETS expects a 1d series. We fit on training and forecast len(test)+horizon steps.
        model = sm_hw.ExponentialSmoothing(
            y_train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        res = model.fit(optimized=True)

        steps = int(len(test_y) + max(0, horizon))
        preds = res.forecast(steps) if steps > 0 else pd.Series([], dtype=float)
        preds = pd.to_numeric(preds, errors="coerce")

        test_yhat = preds.iloc[: len(test_y)].tolist() if len(test_y) > 0 else []

        step = _infer_step(train_dates)
        future_dates = _make_future_dates(pd.Timestamp(train_dates[-1]), horizon=horizon, step=step)
        fcst_yhat = preds.iloc[len(test_y) : len(test_y) + len(future_dates)].tolist() if future_dates else []

        out = build_forecast_output(
            train_dates=list(train_dates),
            train_y=[float(v) if pd.notna(v) else None for v in train_y.tolist()],
            test_dates=list(test_dates),
            test_y=[float(v) if pd.notna(v) else None for v in test_y.tolist()],
            test_yhat=[float(v) if pd.notna(v) else None for v in test_yhat],
            forecast_dates=future_dates,
            forecast_yhat=[float(v) if pd.notna(v) else None for v in fcst_yhat],
            method="ets",
            target=target,
        )

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out

