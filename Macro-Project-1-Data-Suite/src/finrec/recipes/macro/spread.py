from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class SpreadRecipe(Recipe):
    """
    Compute a simple spread between two columns: seriesA - seriesB.

    Expects an input dataframe with at least:
      - date column (default: "date") (passed through)
      - two numeric columns specified by params:
        - series_a (required)
        - series_b (required)

    Output:
      - a copy of the input with an added column (default: "{series_a}_minus_{series_b}").
    """

    meta: RecipeMeta = RecipeMeta(
        id="spread",
        name="Spread (A - B)",
        description="Adds a spread column: series_a - series_b.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        date_col = str(params.get("date_col", "date"))
        series_a = str(params.get("series_a", "")).strip()
        series_b = str(params.get("series_b", "")).strip()
        out_col = str(params.get("out_col", "")).strip() or (f"{series_a}_minus_{series_b}" if series_a and series_b else "")

        if date_col not in df.columns:
            raise ValueError(f"Expected date_col '{date_col}' not found. Columns: {list(df.columns)}")
        if not series_a or not series_b:
            raise ValueError("series_a and series_b are required params (column names in the input dataframe).")
        missing = [c for c in [series_a, series_b] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Columns: {list(df.columns)}")
        if not out_col:
            raise ValueError("out_col must be a non-empty string")

        ctx.log("INFO", f"[{self.meta.id}] Starting. series_a={series_a}, series_b={series_b}, out_col={out_col}")

        a = pd.to_numeric(df[series_a], errors="coerce")
        b = pd.to_numeric(df[series_b], errors="coerce")
        spread = a - b

        df2 = df.copy()
        df2[out_col] = spread

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

