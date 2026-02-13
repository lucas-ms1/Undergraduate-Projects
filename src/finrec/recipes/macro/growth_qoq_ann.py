from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class GrowthQoQAnnRecipe(Recipe):
    """
    Compute annualized quarter-over-quarter growth (%) using log differences.

    Intended for quarterly series (i.e., adjacent rows are one quarter apart).

    Expects an input dataframe with at least:
      - date column (default: "date")
      - value column (default: "value") that is strictly positive

    Output:
      - a copy of the input with an added column (default: "growth_qoq_ann"):
        400 * (log(value) - log(value.shift(1)))

    Robustness:
      - coerces value to numeric
      - values <= 0 are treated as missing (log undefined)
    """

    meta: RecipeMeta = RecipeMeta(
        id="growth_qoq_ann",
        name="Growth QoQ annualized (%)",
        description="Adds growth_qoq_ann = 400*(log(value)-log(value.shift(1))) for quarterly series.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        date_col = str(params.get("date_col", "date"))
        value_col = str(params.get("value_col", "value"))
        out_col = str(params.get("out_col", "growth_qoq_ann"))

        if date_col not in df.columns:
            raise ValueError(f"Expected date_col '{date_col}' not found. Columns: {list(df.columns)}")
        if value_col not in df.columns:
            raise ValueError(f"Expected value_col '{value_col}' not found. Columns: {list(df.columns)}")
        if not out_col:
            raise ValueError("out_col must be a non-empty string")

        ctx.log("INFO", f"[{self.meta.id}] Starting. date_col={date_col}, value_col={value_col}, out_col={out_col}")

        x = pd.to_numeric(df[value_col], errors="coerce")
        x = x.where(x > 0)
        growth = 400.0 * (np.log(x) - np.log(x.shift(1)))

        df2 = df.copy()
        df2[out_col] = growth

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

