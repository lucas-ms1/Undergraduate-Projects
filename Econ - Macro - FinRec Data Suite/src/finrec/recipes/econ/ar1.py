from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.providers.utils.optional import require_optional
from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class AR1Recipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="ar1",
        name="AR(1)",
        description="Fits an AR(1) model to a series and returns parameter estimates.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        sm = require_optional("statsmodels.api", extra_hint="econ")

        y_col = str(params.get("y_col", "value"))
        if y_col not in df.columns:
            raise ValueError(f"Expected y_col '{y_col}' not found. Columns: {list(df.columns)}")

        ctx.log("INFO", f"[{self.meta.id}] Starting. y_col={y_col}")

        y = pd.to_numeric(df[y_col], errors="coerce").dropna()
        # Allow smaller samples (monthly macro series over 1y can be ~12 obs).
        # Stats will be noisy, but this keeps the UI from failing for common ranges.
        if len(y) < 10:
            raise ValueError(f"Need at least 10 non-NaN observations; got n={len(y)}")

        # AR(1) via OLS: y_t = c + phi*y_{t-1} + e_t
        y_lag = y.shift(1)
        data = pd.concat([y.rename("y"), y_lag.rename("y_lag1")], axis=1).dropna()
        Y = data["y"]
        X = sm.add_constant(data["y_lag1"], has_constant="add")

        res = sm.OLS(Y, X).fit()

        out = pd.DataFrame(
            {
                "term": res.params.index.astype(str),
                "coef": res.params.values,
                "std_err": res.bse.values,
                "t": res.tvalues.values,
                "p_value": res.pvalues.values,
            }
        )
        out["nobs"] = float(res.nobs)
        out["r2"] = float(getattr(res, "rsquared", float("nan")))
        out["adj_r2"] = float(getattr(res, "rsquared_adj", float("nan")))

        ctx.log("INFO", f"[{self.meta.id}] Done. nobs={res.nobs}")
        return out

