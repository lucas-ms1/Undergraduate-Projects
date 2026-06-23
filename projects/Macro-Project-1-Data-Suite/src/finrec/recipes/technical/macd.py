from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finrec.recipes.base import Recipe, RecipeMeta


@dataclass
class MACDRecipe(Recipe):
    meta: RecipeMeta = RecipeMeta(
        id="macd",
        name="MACD",
        description="Adds MACD line, signal line, and histogram columns.",
    )

    def run(self, df: pd.DataFrame, params: dict, ctx) -> pd.DataFrame:
        price_col = str(params.get("price_col", "close"))
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal = int(params.get("signal", 9))
        prefix = str(params.get("prefix", "macd"))

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Starting. price_col={price_col}, fast={fast}, slow={slow}, signal={signal}, prefix={prefix}",
        )

        if price_col not in df.columns:
            raise ValueError(f"Expected column '{price_col}' not found. Columns: {list(df.columns)}")

        s = pd.to_numeric(df[price_col], errors="coerce")
        if s.isna().all():
            raise ValueError(f"Column '{price_col}' could not be parsed as numeric.")

        ema_fast = s.ewm(span=fast, adjust=False, min_periods=max(2, fast // 3)).mean()
        ema_slow = s.ewm(span=slow, adjust=False, min_periods=max(2, slow // 3)).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=max(2, signal // 3)).mean()
        hist = macd_line - signal_line

        df2 = df.copy()
        df2[f"{prefix}_line"] = macd_line
        df2[f"{prefix}_signal"] = signal_line
        df2[f"{prefix}_hist"] = hist

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Done. Added columns '{prefix}_line', '{prefix}_signal', '{prefix}_hist'.",
        )
        return df2

