"""
Summary statistics from simulated time series.
===============================================
Computes the moments required by the assignment:
    - Mean and variance of each series
    - First-order (lag-1) autocorrelation
    - Correlation with aggregate income / output
"""

import numpy as np


def _lag1_autocorrelation(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D series."""
    if len(x) < 2 or np.std(x) == 0:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def compute_moments(
    series: dict[str, np.ndarray],
    income_key: str = "y",
) -> dict:
    """Compute summary statistics for all simulated series.

    Parameters
    ----------
    series : dict[str, np.ndarray]
        Time-series paths keyed by variable name.
    income_key : str
        Key of the aggregate income/output series for
        correlation calculations.

    Returns
    -------
    dict
        Nested dict:  {var_name: {'mean', 'variance',
        'autocorrelation', 'corr_with_income'}}
    """
    income = series.get(income_key)
    results = {}
    for name, arr in series.items():
        arr = np.asarray(arr, dtype=float)
        entry: dict = {
            "mean": float(np.mean(arr)),
            "variance": float(np.var(arr)),
            "autocorrelation": _lag1_autocorrelation(arr),
        }
        if income is not None and len(income) == len(arr) and np.std(arr) > 0 and np.std(income) > 0:
            entry["corr_with_income"] = float(
                np.corrcoef(arr, income)[0, 1]
            )
        else:
            entry["corr_with_income"] = np.nan
        results[name] = entry
    return results
