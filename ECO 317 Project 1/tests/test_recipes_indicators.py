from __future__ import annotations

import pandas as pd

from finrec.recipes.technical.ema import ExponentialMovingAverageRecipe
from finrec.recipes.technical.macd import MACDRecipe
from finrec.recipes.technical.rolling_vol import RollingVolatilityRecipe
from finrec.recipes.technical.rsi import RSIRecipe
from finrec.recipes.technical.sma import SimpleMovingAverageRecipe


class _Ctx:
    def log(self, level: str, message: str) -> None:
        pass


def test_sma_adds_column():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    out = SimpleMovingAverageRecipe().run(df, params={"value_col": "close", "window": 3, "out_col": "sma3"}, ctx=_Ctx())
    assert "sma3" in out.columns


def test_ema_adds_column():
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    out = ExponentialMovingAverageRecipe().run(df, params={"value_col": "close", "span": 3, "out_col": "ema3"}, ctx=_Ctx())
    assert "ema3" in out.columns


def test_rsi_adds_column():
    df = pd.DataFrame({"close": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 108, 109, 110, 111, 112]})
    out = RSIRecipe().run(df, params={"price_col": "close", "window": 14, "out_col": "rsi14"}, ctx=_Ctx())
    assert "rsi14" in out.columns


def test_macd_adds_columns():
    df = pd.DataFrame({"close": list(range(1, 100))})
    out = MACDRecipe().run(df, params={"price_col": "close", "fast": 12, "slow": 26, "signal": 9, "prefix": "macd"}, ctx=_Ctx())
    assert "macd_line" in out.columns
    assert "macd_signal" in out.columns
    assert "macd_hist" in out.columns


def test_rolling_vol_adds_column():
    df = pd.DataFrame({"close": list(range(1, 100))})
    out = RollingVolatilityRecipe().run(df, params={"price_col": "close", "window": 20, "out_col": "vol20"}, ctx=_Ctx())
    assert "vol20" in out.columns

