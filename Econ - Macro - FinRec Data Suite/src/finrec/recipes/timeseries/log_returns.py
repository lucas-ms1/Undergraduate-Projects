from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class LogReturnsRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="log_returns",
        name="Log returns",
        description="Adds a log_return column computed from a price-like column.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        col = str(params.get("price_col", "close"))
        out_col = str(params.get("out_col", "log_return"))

        ctx.log("INFO", f"[{self.meta.id}] Starting. price_col={col}, out_col={out_col}")

        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. Columns: {list(df.columns)}")

        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().all():
            raise ValueError(f"Column '{col}' could not be parsed as numeric.")

        # log return = log(p_t) - log(p_{t-1})
        # guard nonpositive values
        if (s <= 0).any():
            ctx.log("WARNING", f"[{self.meta.id}] Nonpositive values found in '{col}'. They will become NaN.")
        logp = s.apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else float("nan"))
        df2 = df.copy()
        df2[out_col] = logp.diff()

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

