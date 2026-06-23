from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class InflationYoYRecipe(Recipe):
    """
    Compute year-over-year inflation (%) from a monthly price level index/series.

    Expects an input dataframe with at least:
      - date column (default: "date")
      - value column (default: "value")

    Output:
      - a copy of the input with an added column (default: "inflation_yoy"):
        100 * (value / value.shift(12) - 1)

    Robustness:
      - coerces value to numeric
      - leaves rows with missing / non-numeric values as NaN
      - converts +/- inf (e.g., division by 0) to NaN
    """

    meta: RecipeMeta = RecipeMeta(
        id="inflation_yoy",
        name="Inflation YoY (%)",
        description="Adds inflation_yoy = 100*(value/value.shift(12)-1) for monthly series.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        date_col = str(params.get("date_col", "date"))
        value_col = str(params.get("value_col", "value"))
        out_col = str(params.get("out_col", "inflation_yoy"))

        if date_col not in df.columns:
            raise ValueError(f"Expected date_col '{date_col}' not found. Columns: {list(df.columns)}")
        if value_col not in df.columns:
            raise ValueError(f"Expected value_col '{value_col}' not found. Columns: {list(df.columns)}")
        if not out_col:
            raise ValueError("out_col must be a non-empty string")

        ctx.log("INFO", f"[{self.meta.id}] Starting. date_col={date_col}, value_col={value_col}, out_col={out_col}")

        x = pd.to_numeric(df[value_col], errors="coerce")
        denom = x.shift(12)
        yoy = 100.0 * (x / denom - 1.0)
        yoy = yoy.replace([float("inf"), float("-inf")], pd.NA)

        df2 = df.copy()
        df2[out_col] = yoy

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

