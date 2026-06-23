"""
Summary statistics from simulated time series.
===============================================
Computes the moments required by the assignment:
    - Mean and variance of each series
    - First-order (lag-1) autocorrelation
    - Correlation with aggregate income / output
"""

import numpy as np


def compute_moments(series: dict[str, np.ndarray],
                    income_key: str = "y") -> dict:
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
    if income_key not in series:
        raise KeyError(f"income_key '{income_key}' not found in series.")

    y = np.asarray(series[income_key], dtype=float)
    T_income = y.size

    moments: dict[str, dict[str, float]] = {}

    for name, x_raw in series.items():
        x = np.asarray(x_raw, dtype=float)

        if x.size != T_income:
            raise ValueError(
                f"Series '{name}' has length {x.size}, "
                f"but income series has length {T_income}."
            )

        # Mean and variance (population definitions)
        mean_x = float(np.mean(x))
        var_x = float(np.var(x))

        # Lag-1 autocorrelation
        if x.size < 2 or var_x == 0.0:
            autocorr = np.nan
        else:
            x_centered = x - mean_x
            num = np.dot(x_centered[1:], x_centered[:-1])
            den = np.dot(x_centered, x_centered)
            autocorr = float(num / den)

        # Correlation with income
        mean_y = float(np.mean(y))
        var_y = float(np.var(y))

        if var_x == 0.0 or var_y == 0.0:
            corr_xy = np.nan
        else:
            cov_xy = float(np.mean((x - mean_x) * (y - mean_y)))
            std_x = float(np.sqrt(var_x))
            std_y = float(np.sqrt(var_y))
            corr_xy = cov_xy / (std_x * std_y)

        moments[name] = {
            "mean": mean_x,
            "variance": var_x,
            "autocorrelation": autocorr,
            "corr_with_income": corr_xy,
        }

    return moments
