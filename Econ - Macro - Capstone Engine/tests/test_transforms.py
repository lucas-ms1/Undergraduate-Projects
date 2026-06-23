import os
import sys
import traceback

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from empirical.transforms import annualized_mom, hp_filter, log_difference, yoy_growth  # noqa: E402


FAILURES = []


def check(name, func):
    try:
        func()
        print(f"PASS  {name}")
    except Exception:
        print(f"FAIL  {name}")
        print(traceback.format_exc())
        FAILURES.append(name)


def test_log_difference():
    t = np.arange(100)
    series = pd.Series(np.exp(0.05 * t))
    ld = log_difference(series)
    assert len(ld) == 99
    assert np.allclose(ld.values, 0.05, atol=1e-10), f"Log-diff not constant: {ld.describe()}"
    ld_q = log_difference(series, periods=4)
    assert len(ld_q) == 96
    assert np.allclose(ld_q.values, 0.20, atol=1e-10)


def test_yoy_growth():
    monthly_factor = 1.10 ** (1 / 12)
    monthly_series = pd.Series(100 * monthly_factor ** np.arange(36))
    yoy = yoy_growth(monthly_series, periods=12)
    assert len(yoy) == 24
    assert np.allclose(yoy.values, 10.0, atol=0.1), f"YoY not near 10%: {yoy.describe()}"


def test_hp_filter_reconstruction():
    np.random.seed(42)
    raw = pd.Series(np.cumsum(np.random.randn(200)) + np.linspace(0, 10, 200))
    trend, cycle = hp_filter(raw, lamb=1600)
    common = raw.index.intersection(trend.index).intersection(cycle.index)
    assert len(common) > 0, "No overlapping indices"
    assert np.allclose(raw.loc[common].values, (trend.loc[common] + cycle.loc[common]).values, atol=1e-8)


def test_annualized_mom():
    cpi = pd.Series(100 * 1.005 ** np.arange(24))
    ann = annualized_mom(cpi)
    expected = (1.005**12 - 1) * 100
    assert np.allclose(ann.values, expected, atol=0.1)


def main():
    check("Log-difference", test_log_difference)
    check("YoY growth", test_yoy_growth)
    check("HP filter reconstruction", test_hp_filter_reconstruction)
    check("Annualized MoM", test_annualized_mom)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
