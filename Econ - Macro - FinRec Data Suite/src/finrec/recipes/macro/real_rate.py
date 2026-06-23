from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class RealRateRecipe(Recipe):
    """
    Compute an ex-post real rate: nominal_rate - inflation_yoy.

    Expects an input dataframe with at least:
      - date column (default: "date") (passed through)
      - nominal rate column (default: "nominal_rate")
      - inflation YoY column (default: "inflation_yoy")

    Output:
      - a copy of the input with an added column (default: "real_rate"):
        real_rate = nominal_rate - inflation_yoy
    """

    meta: RecipeMeta = RecipeMeta(
        id="real_rate",
        name="Real rate (nominal - inflation)",
        description="Adds real_rate = nominal_rate - inflation_yoy.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        date_col = str(params.get("date_col", "date"))
        nominal_col = str(params.get("nominal_col", "nominal_rate")).strip()
        inflation_col = str(params.get("inflation_col", "inflation_yoy")).strip()
        out_col = str(params.get("out_col", "real_rate")).strip()

        if date_col not in df.columns:
            raise ValueError(f"Expected date_col '{date_col}' not found. Columns: {list(df.columns)}")
        missing = [c for c in [nominal_col, inflation_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Columns: {list(df.columns)}")
        if not out_col:
            raise ValueError("out_col must be a non-empty string")

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. nominal_col={nominal_col}, inflation_col={inflation_col}, out_col={out_col}",
        )

        nominal = pd.to_numeric(df[nominal_col], errors="coerce")
        infl = pd.to_numeric(df[inflation_col], errors="coerce")
        real = nominal - infl

        df2 = df.copy()
        df2[out_col] = real

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

