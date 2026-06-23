from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class RollingZScoreRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="rolling_zscore",
        name="Rolling Z-score",
        description="Adds rolling mean/std/z columns for a numeric column.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        col = str(params.get("value_col", "value"))
        window = int(params.get("window", 12))
        prefix = str(params.get("prefix", "roll"))

        ctx.log("INFO", f"[{self.meta.id}] Starting. value_col={col}, window={window}, prefix={prefix}")

        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. Columns: {list(df.columns)}")

        x = pd.to_numeric(df[col], errors="coerce")
        if x.isna().all():
            raise ValueError(f"Column '{col}' could not be parsed as numeric.")

        df2 = df.copy()
        mu = x.rolling(window=window, min_periods=max(2, window // 3)).mean()
        sd = x.rolling(window=window, min_periods=max(2, window // 3)).std(ddof=0)

        df2[f"{prefix}_mean"] = mu
        df2[f"{prefix}_std"] = sd
        df2[f"{prefix}_z"] = (x - mu) / sd

        ctx.log("INFO", f"[{self.meta.id}] Done. Added columns '{prefix}_mean', '{prefix}_std', '{prefix}_z'.")
        return df2

