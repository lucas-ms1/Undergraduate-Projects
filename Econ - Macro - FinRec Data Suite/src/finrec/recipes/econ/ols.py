from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.providers.utils.optional import require_optional
from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class OLSRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="ols",
        name="OLS regression",
        description="Runs OLS: y ~ X and returns a coefficient table (with standard errors, t-stats, p-values).",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        sm = require_optional("statsmodels.api", extra_hint="econ")

        y_col = str(params.get("y_col", "value"))
        x_cols_raw = str(params.get("x_cols", "")).strip()
        add_constant = bool(params.get("add_constant", True))

        if y_col not in df.columns:
            raise ValueError(f"Expected y_col '{y_col}' not found. Columns: {list(df.columns)}")

        x_cols = [c.strip() for c in x_cols_raw.split(",") if c.strip()]
        if not x_cols:
            raise ValueError("x_cols must be a comma-separated list of one or more columns.")

        missing = [c for c in x_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing x columns: {missing}. Columns: {list(df.columns)}")

        ctx.log("INFO", f"[{self.meta.id}] Starting. y_col={y_col}, x_cols={x_cols}, add_constant={add_constant}")

        y = pd.to_numeric(df[y_col], errors="coerce")
        X = df[x_cols].apply(pd.to_numeric, errors="coerce")

        data = pd.concat([y.rename(y_col), X], axis=1).dropna()
        k = len(x_cols) + (1 if add_constant else 0)
        min_n = max(5, k + 2)  # a small, practical minimum
        if len(data) < min_n:
            raise ValueError(
                f"Not enough complete observations after dropping NaNs: n={len(data)} (need >= {min_n})"
            )

        y2 = data[y_col]
        X2 = data[x_cols]
        if add_constant:
            X2 = sm.add_constant(X2, has_constant="add")

        model = sm.OLS(y2, X2)
        res = model.fit()

        # Return a tidy coefficient table
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

        ctx.log("INFO", f"[{self.meta.id}] Done. nobs={res.nobs}, r2={getattr(res, 'rsquared', None)}")
        return out

