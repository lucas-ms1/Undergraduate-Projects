from __future__ import annotations

import pytest
import pandas as pd

from finrec.recipes.econ.ar1 import AR1Recipe
from finrec.recipes.econ.ols import OLSRecipe
from finrec.recipes.econ.lp_irf import LocalProjectionIRFRecipe


class _Ctx:
    def log(self, level: str, message: str) -> None:
        pass


statsmodels = pytest.importorskip("statsmodels")


def test_ols_runs():
    df = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6],
            "x1": [1, 1, 2, 2, 3, 3],
            "x2": [0, 1, 0, 1, 0, 1],
        }
    )
    out = OLSRecipe().run(df, params={"y_col": "y", "x_cols": "x1,x2", "add_constant": True}, ctx=_Ctx())
    assert "term" in out.columns
    assert len(out) >= 3  # const + 2 regressors


def test_ar1_runs():
    df = pd.DataFrame({"value": list(range(1, 100))})
    out = AR1Recipe().run(df, params={"y_col": "value"}, ctx=_Ctx())
    assert "term" in out.columns


def test_lp_irf_runs():
    n = 100
    df = pd.DataFrame(
        {
            "value": list(range(n)),
            "shock": [0.0] * (n // 2) + [1.0] * (n - n // 2),
        }
    )
    out = LocalProjectionIRFRecipe().run(
        df,
        params={"y_col": "value", "shock_col": "shock", "controls": "", "horizons": 5, "ci_level": 0.95, "add_constant": True},
        ctx=_Ctx(),
    )
    assert "horizon" in out.columns

