from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class RSIRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="rsi",
        name="Relative Strength Index (RSI)",
        description="Adds an RSI column computed from a price-like column.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        price_col = str(params.get("price_col", "close"))
        window = int(params.get("window", 14))
        out_col = str(params.get("out_col", f"rsi_{window}"))

        ctx.log("INFO", f"[{self.meta.id}] Starting. price_col={price_col}, window={window}, out_col={out_col}")

        if price_col not in df.columns:
            raise ValueError(f"Expected column '{price_col}' not found. Columns: {list(df.columns)}")

        s = pd.to_numeric(df[price_col], errors="coerce")
        if s.isna().all():
            raise ValueError(f"Column '{price_col}' could not be parsed as numeric.")

        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Classic RSI uses Wilder's smoothing; approximate with EMA(alpha=1/window)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df2 = df.copy()
        df2[out_col] = rsi

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

