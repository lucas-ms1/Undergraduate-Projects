from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class ExponentialMovingAverageRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="ema",
        name="Exponential moving average (EMA)",
        description="Adds an EMA column for a numeric series column.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        value_col = str(params.get("value_col", "close"))
        span = int(params.get("span", 20))
        out_col = str(params.get("out_col", f"ema_{span}"))

        ctx.log("INFO", f"[{self.meta.id}] Starting. value_col={value_col}, span={span}, out_col={out_col}")

        if value_col not in df.columns:
            raise ValueError(f"Expected column '{value_col}' not found. Columns: {list(df.columns)}")

        x = pd.to_numeric(df[value_col], errors="coerce")
        if x.isna().all():
            raise ValueError(f"Column '{value_col}' could not be parsed as numeric.")

        df2 = df.copy()
        df2[out_col] = x.ewm(span=span, adjust=False, min_periods=max(2, span // 3)).mean()

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

