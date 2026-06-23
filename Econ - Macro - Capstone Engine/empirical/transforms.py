"""
empirical/transforms.py
=======================
Data transformations for macroeconomic time series.

Provides log-differencing, year-over-year growth rates, HP filtering,
and other standard transformations used in empirical macro.
"""

import numpy as np
import pandas as pd


def log_difference(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Compute log-difference (approximate growth rate).

    Δ ln(x_t) = ln(x_t) - ln(x_{t-periods})

    For monthly data with periods=1, this gives month-over-month growth.
    Multiply by 100 for percentage.
    """
    return (np.log(series) - np.log(series.shift(periods))).dropna()


def yoy_growth(series: pd.Series, periods: int = 12) -> pd.Series:
    """
    Year-over-Year percentage growth rate.

    For monthly data: periods=12
    For quarterly data: periods=4
    """
    return ((series / series.shift(periods) - 1) * 100).dropna()


def annualized_mom(series: pd.Series) -> pd.Series:
    """
    Annualized month-over-month growth (for monthly CPI → annualized inflation).

    = [(x_t / x_{t-1})^12 - 1] * 100
    """
    ratio = series / series.shift(1)
    return ((ratio ** 12 - 1) * 100).dropna()


def hp_filter(series: pd.Series, lamb: float = 1600) -> tuple[pd.Series, pd.Series]:
    """
    Hodrick-Prescott filter. Returns (trend, cycle).

    Parameters
    ----------
    series : pd.Series (must not contain NaN)
    lamb : smoothing parameter (1600 for quarterly, 129600 for monthly)

    Returns
    -------
    (trend, cycle) as pd.Series
    """
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        cycle, trend = hpfilter(series.dropna(), lamb=lamb)
        return trend, cycle
    except ImportError:
        # Fallback: simple moving average as "trend"
        trend = series.rolling(window=20, center=True, min_periods=5).mean()
        cycle = series - trend
        return trend.dropna(), cycle.dropna()


def compute_spread(long_rate: pd.Series, short_rate: pd.Series) -> pd.Series:
    """Yield spread: long minus short rate."""
    aligned = pd.concat([long_rate, short_rate], axis=1).dropna()
    return (aligned.iloc[:, 0] - aligned.iloc[:, 1])


def real_rate(nominal_rate: pd.Series, inflation: pd.Series) -> pd.Series:
    """Fisher equation: real ≈ nominal - inflation."""
    aligned = pd.concat([nominal_rate, inflation], axis=1).dropna()
    return (aligned.iloc[:, 0] - aligned.iloc[:, 1])


# ---------------------------------------------------------------------------
# Transformation registry (for UI selection)
# ---------------------------------------------------------------------------
TRANSFORMS = {
    "Raw Level": lambda s: s,
    "Log-Difference": lambda s: log_difference(s) * 100,
    "YoY Growth (%)": lambda s: yoy_growth(s),
    "Annualized MoM (%)": lambda s: annualized_mom(s),
}
