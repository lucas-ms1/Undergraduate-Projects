from __future__ import annotations

from dataclasses import dataclass
import itertools

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
class ARIMAForecastRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="forecast_arima",
        name="Forecast: ARIMA",
        description="Fits an ARIMA(p,d,q) model (small AIC grid search) and forecasts forward.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        sm_sarimax = require_optional("statsmodels.tsa.statespace.sarimax", extra_hint="forecast")

        date_col = str(params.get("date_col", "date"))
        y_col = str(params.get("y_col", "value"))
        transform: TargetTransform = str(params.get("transform", "none"))  # type: ignore[assignment]
        horizon = int(params.get("horizon", 12))
        test_size = params.get("test_size", 0.2)

        p_vals = params.get("p_values", [0, 1, 2])
        d_vals = params.get("d_values", [0, 1])
        q_vals = params.get("q_values", [0, 1, 2])
        alpha = float(params.get("alpha", 0.05))  # 95% CI default

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. date_col={date_col}, y_col={y_col}, transform={transform}, "
            f"horizon={horizon}, test_size={test_size}, grid=({p_vals},{d_vals},{q_vals}), alpha={alpha}",
        )

        frame = build_target_series(df, date_col=date_col, y_col=y_col, transform=transform)
        dates, y, target = frame.date_index, frame.y, frame.target
        train_dates, train_y, test_dates, test_y = split_train_test(dates, y, test_size=test_size)

        y_train = train_y.dropna().astype(float)
        if len(y_train) < 20:
            raise ValueError(f"Need at least 20 non-NaN training observations for ARIMA; got n={len(y_train)}")

        best_res = None
        best_order = None
        best_aic = float("inf")

        orders = list(itertools.product(list(p_vals), list(d_vals), list(q_vals)))
        for (p, d, q) in orders:
            order = (int(p), int(d), int(q))
            try:
                model = sm_sarimax.SARIMAX(
                    y_train,
                    order=order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                aic = float(getattr(res, "aic", float("inf")))
                if aic < best_aic:
                    best_aic = aic
                    best_res = res
                    best_order = order
            except Exception as e:
                ctx.log("WARNING", f"[{self.meta.id}] ARIMA{order} failed: {e}")
                continue

        if best_res is None or best_order is None:
            raise ValueError("All ARIMA grid search fits failed.")

        ctx.log("INFO", f"[{self.meta.id}] Selected order={best_order} (AIC={best_aic:.3f})")

        steps = int(len(test_y) + max(0, horizon))
        if steps <= 0:
            raise ValueError("Nothing to forecast: both test_size and horizon resulted in 0 steps.")

        fcst = best_res.get_forecast(steps=steps)
        mean = pd.to_numeric(fcst.predicted_mean, errors="coerce")

        ci = None
        try:
            ci = fcst.conf_int(alpha=alpha)
        except Exception:
            ci = None

        test_mean = mean.iloc[: len(test_y)]
        test_yhat = [float(v) if pd.notna(v) else None for v in test_mean.tolist()]

        step = _infer_step(train_dates)
        future_dates = _make_future_dates(pd.Timestamp(train_dates[-1]), horizon=horizon, step=step)
        fcst_mean = mean.iloc[len(test_y) : len(test_y) + len(future_dates)]
        fcst_yhat = [float(v) if pd.notna(v) else None for v in fcst_mean.tolist()]

        test_lo: list[float | None] | None = None
        test_hi: list[float | None] | None = None
        fcst_lo: list[float | None] | None = None
        fcst_hi: list[float | None] | None = None
        if ci is not None and len(ci.columns) >= 2:
            lo = pd.to_numeric(ci.iloc[:, 0], errors="coerce")
            hi = pd.to_numeric(ci.iloc[:, 1], errors="coerce")
            test_lo = [float(v) if pd.notna(v) else None for v in lo.iloc[: len(test_y)].tolist()]
            test_hi = [float(v) if pd.notna(v) else None for v in hi.iloc[: len(test_y)].tolist()]
            fcst_lo = [float(v) if pd.notna(v) else None for v in lo.iloc[len(test_y) : len(test_y) + len(future_dates)].tolist()]
            fcst_hi = [float(v) if pd.notna(v) else None for v in hi.iloc[len(test_y) : len(test_y) + len(future_dates)].tolist()]

        out = build_forecast_output(
            train_dates=list(train_dates),
            train_y=[float(v) if pd.notna(v) else None for v in train_y.tolist()],
            test_dates=list(test_dates),
            test_y=[float(v) if pd.notna(v) else None for v in test_y.tolist()],
            test_yhat=test_yhat,
            forecast_dates=future_dates,
            forecast_yhat=fcst_yhat,
            test_lower=test_lo,
            test_upper=test_hi,
            fcst_lower=fcst_lo,
            fcst_upper=fcst_hi,
            method=f"arima{best_order}",
            target=target,
        )

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out

