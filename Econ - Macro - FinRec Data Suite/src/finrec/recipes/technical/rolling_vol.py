from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class RollingVolatilityRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="rolling_vol",
        name="Rolling volatility",
        description="Adds a rolling volatility column from returns of a price-like column.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        price_col = str(params.get("price_col", "close"))
        window = int(params.get("window", 20))
        annualize = bool(params.get("annualize", True))
        periods_per_year = int(params.get("periods_per_year", 252))
        out_col = str(params.get("out_col", f"vol_{window}"))

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. price_col={price_col}, window={window}, annualize={annualize}, out_col={out_col}",
        )

        if price_col not in df.columns:
            raise ValueError(f"Expected column '{price_col}' not found. Columns: {list(df.columns)}")

        p = pd.to_numeric(df[price_col], errors="coerce")
        if p.isna().all():
            raise ValueError(f"Column '{price_col}' could not be parsed as numeric.")

        # Simple returns
        r = p.pct_change()
        vol = r.rolling(window=window, min_periods=max(2, window // 3)).std(ddof=0)
        if annualize:
            vol = vol * math.sqrt(periods_per_year)

        df2 = df.copy()
        df2[out_col] = vol

        ctx.log("INFO", f"[{self.meta.id}] Done. Added column '{out_col}'.")
        return df2

