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


def _make_features(window: list[float], *, include_stats: bool) -> list[float]:
    if not window:
        return []
    feats = list(window)
    if include_stats:
        s = pd.Series(window, dtype="float64")
        feats.append(float(s.mean()))
        feats.append(float(s.std(ddof=0)))
    return feats


def _build_supervised(
    y: pd.Series, *, lookback: int, include_stats: bool
) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(y) <= lookback + 5:
        raise ValueError(f"Need more data for supervised lags: n={len(y)} lookback={lookback}")

    rows: list[list[float]] = []
    targets: list[float] = []
    values = y.tolist()
    for t in range(lookback, len(values)):
        window = values[t - lookback : t]
        rows.append(_make_features(window, include_stats=include_stats))
        targets.append(float(values[t]))

    X = pd.DataFrame(rows)
    y_out = pd.Series(targets)
    return X, y_out


def _recursive_predict(
    model, *, init_window: list[float], steps: int, include_stats: bool
) -> list[float]:
    window = list(init_window)
    out: list[float] = []
    for _ in range(steps):
        feats = _make_features(window, include_stats=include_stats)
        x = pd.DataFrame([feats])
        yhat = float(model.predict(x)[0])
        out.append(yhat)
        window = window[1:] + [yhat]
    return out


@dataclass
class RidgeLagForecastRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="forecast_ridge_lags",
        name="Forecast: Ridge (lags)",
        description="Lag-feature regression (Ridge) with recursive multi-step forecasting.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        sk_linear = require_optional("sklearn.linear_model", extra_hint="forecast")
        sk_pipe = require_optional("sklearn.pipeline", extra_hint="forecast")
        sk_pre = require_optional("sklearn.preprocessing", extra_hint="forecast")

        date_col = str(params.get("date_col", "date"))
        y_col = str(params.get("y_col", "close"))
        transform: TargetTransform = str(params.get("transform", "none"))  # type: ignore[assignment]
        horizon = int(params.get("horizon", 30))
        test_size = params.get("test_size", 0.2)
        lookback = int(params.get("lookback", 20))
        alpha = float(params.get("alpha", 1.0))
        include_stats = bool(params.get("include_stats", True))

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. date_col={date_col}, y_col={y_col}, transform={transform}, horizon={horizon}, "
            f"test_size={test_size}, lookback={lookback}, alpha={alpha}, include_stats={include_stats}",
        )

        frame = build_target_series(df, date_col=date_col, y_col=y_col, transform=transform)
        dates, y, target = frame.date_index, frame.y, frame.target
        train_dates, train_y, test_dates, test_y = split_train_test(dates, y, test_size=test_size)

        X_train, y_train_sup = _build_supervised(train_y, lookback=lookback, include_stats=include_stats)
        model = sk_pipe.make_pipeline(sk_pre.StandardScaler(), sk_linear.Ridge(alpha=alpha))
        model.fit(X_train, y_train_sup)

        y_train_vals = pd.to_numeric(train_y, errors="coerce").dropna().astype(float).tolist()
        if len(y_train_vals) < lookback:
            raise ValueError(f"Not enough training history for lookback={lookback}.")
        init_window = y_train_vals[-lookback:]

        steps = int(len(test_y) + max(0, horizon))
        preds = _recursive_predict(model, init_window=init_window, steps=steps, include_stats=include_stats)

        test_yhat = preds[: len(test_y)]

        step = _infer_step(train_dates)
        future_dates = _make_future_dates(pd.Timestamp(train_dates[-1]), horizon=horizon, step=step)
        fcst_yhat = preds[len(test_y) : len(test_y) + len(future_dates)] if future_dates else []

        out = build_forecast_output(
            train_dates=list(train_dates),
            train_y=[float(v) if pd.notna(v) else None for v in train_y.tolist()],
            test_dates=list(test_dates),
            test_y=[float(v) if pd.notna(v) else None for v in test_y.tolist()],
            test_yhat=[float(v) if pd.notna(v) else None for v in test_yhat],
            forecast_dates=future_dates,
            forecast_yhat=[float(v) if pd.notna(v) else None for v in fcst_yhat],
            method="ridge_lags",
            target=target,
        )

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out


@dataclass
class RandomForestLagForecastRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="forecast_rf_lags",
        name="Forecast: RandomForest (lags)",
        description="Lag-feature regression (RandomForest) with recursive multi-step forecasting.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        sk_ens = require_optional("sklearn.ensemble", extra_hint="forecast")

        date_col = str(params.get("date_col", "date"))
        y_col = str(params.get("y_col", "close"))
        transform: TargetTransform = str(params.get("transform", "none"))  # type: ignore[assignment]
        horizon = int(params.get("horizon", 30))
        test_size = params.get("test_size", 0.2)
        lookback = int(params.get("lookback", 20))
        include_stats = bool(params.get("include_stats", True))
        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. date_col={date_col}, y_col={y_col}, transform={transform}, horizon={horizon}, "
            f"test_size={test_size}, lookback={lookback}, include_stats={include_stats}, "
            f"n_estimators={n_estimators}, max_depth={max_depth}",
        )

        frame = build_target_series(df, date_col=date_col, y_col=y_col, transform=transform)
        dates, y, target = frame.date_index, frame.y, frame.target
        train_dates, train_y, test_dates, test_y = split_train_test(dates, y, test_size=test_size)

        X_train, y_train_sup = _build_supervised(train_y, lookback=lookback, include_stats=include_stats)
        model = sk_ens.RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=0,
            max_depth=max_depth,
            n_jobs=1,
        )
        model.fit(X_train, y_train_sup)

        y_train_vals = pd.to_numeric(train_y, errors="coerce").dropna().astype(float).tolist()
        if len(y_train_vals) < lookback:
            raise ValueError(f"Not enough training history for lookback={lookback}.")
        init_window = y_train_vals[-lookback:]

        steps = int(len(test_y) + max(0, horizon))
        preds = _recursive_predict(model, init_window=init_window, steps=steps, include_stats=include_stats)

        test_yhat = preds[: len(test_y)]

        step = _infer_step(train_dates)
        future_dates = _make_future_dates(pd.Timestamp(train_dates[-1]), horizon=horizon, step=step)
        fcst_yhat = preds[len(test_y) : len(test_y) + len(future_dates)] if future_dates else []

        out = build_forecast_output(
            train_dates=list(train_dates),
            train_y=[float(v) if pd.notna(v) else None for v in train_y.tolist()],
            test_dates=list(test_dates),
            test_y=[float(v) if pd.notna(v) else None for v in test_y.tolist()],
            test_yhat=[float(v) if pd.notna(v) else None for v in test_yhat],
            forecast_dates=future_dates,
            forecast_yhat=[float(v) if pd.notna(v) else None for v in fcst_yhat],
            method="rf_lags",
            target=target,
        )

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out

