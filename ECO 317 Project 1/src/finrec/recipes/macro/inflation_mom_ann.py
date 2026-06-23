from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class InflationMoMAnnRecipe(Recipe):
    """
    Compute annualized month-over-month inflation (%) using log differences.

    Expects an input dataframe with at least:
      - date column (default: "date")
      - value column (default: "value") that is strictly positive

    Output:
      - a copy of the input with an added column (default: "inflation_mom_ann"):
        1200 * (log(value) - log(value.shift(1)))

    Robustness:
      - coerces value to numeric
      - values <= 0 are treated as missing (log undefined)
    """

    meta: RecipeMeta = RecipeMeta(
        id="inflation_mom_ann",
        name="Inflation MoM annualized (%)",
        description="Adds inflation_mom_ann = 1200*(log(value)-log(value.shift(1))).",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        date_col = str(params.get("date_col", "date"))
        value_col = str(params.get("value_col", "value"))
        out_col = str(params.get("out_col", "inflation_mom_ann"))

        if date_col not in df.columns:
            raise ValueError(f"Expected date_col '{date_col}' not found. Columns: {list(df.columns)}")
        if value_col not in df.columns:
            raise ValueError(f"Expected value_col '{value_col}' not found. Columns: {list(df.columns)}")
        if not out_col:
            raise ValueError("out_col must be a non-empty string")

        ctx.log("INFO", f"[{self.meta.id}] Starting. date_col={date_col}, value_col={value_col}, out_col={out_col}")

        x = pd.to_numeric(df[value_col], errors="coerce")
        x = x.where(x > 0)  # log only defined for positive values
        mom_ann = 1200.0 * (np.log(x) - np.log(x.shift(1)))

        df2 = df.copy()
        df2[out_col] = mom_ann

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

