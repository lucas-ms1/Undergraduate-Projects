"""
simulation/moments.py
Step 8 – Moment computation
----------------------------
Computes at least 10 moments from a dict of observable time series:
  • Variance of each series            (5 moments)
  • Correlation of each with output y  (5 moments, output vs itself = 1)
  • First-order autocorrelation        (5 moments)

Two sets of moments are returned:
  raw_moments  – computed directly on the log-deviation series (model output)
  hp_moments   – computed after HP-filtering with λ=1600 (for FRED comparison)

HP filter is the standard Hodrick–Prescott implementation via scipy.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
from statsmodels.tsa.filters.hp_filter import hpfilter


# ---------------------------------------------------------------------------
# HP filter helper
# ---------------------------------------------------------------------------

def hp_cycle(series: np.ndarray, lamb: float = 1600.0) -> np.ndarray:
    """
    Return the cyclical component of `series` after HP filtering.

    Parameters
    ----------
    series : 1-D array of the time series
    lamb   : smoothing parameter (1 600 for quarterly data)
    """
    cycle, _ = hpfilter(series, lamb=lamb)
    return cycle


# ---------------------------------------------------------------------------
# Core moment computations
# ---------------------------------------------------------------------------

def _variance(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1))


def _corr_with_output(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of x with output y."""
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _autocorr1(x: np.ndarray) -> float:
    """First-order autocorrelation (lag-1 Pearson)."""
    if len(x) < 2 or np.std(x) == 0:
        return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# ---------------------------------------------------------------------------
# Main moment function
# ---------------------------------------------------------------------------

def compute_moments(
    series: Dict[str, np.ndarray],
    output_key: str = "y",
    apply_hp: bool = False,
    hp_lambda: float = 1600.0,
) -> Dict[str, float]:
    """
    Compute variance, output-correlation, and AR(1) for each observable.

    Parameters
    ----------
    series     : dict of {name: (T,) array} – e.g. from extract_observables
    output_key : which key is treated as "output" for correlations
    apply_hp   : if True, HP-filter each series before computing moments
    hp_lambda  : HP smoothing parameter (default 1 600 for quarterly)

    Returns
    -------
    Flat dict with keys like "var_y", "corr_c", "ac1_i", etc.
    Moments:
      var_{name}  – variance
      corr_{name} – correlation with output
      ac1_{name}  – first-order autocorrelation
    """
    if output_key not in series:
        raise KeyError(f"output_key '{output_key}' not found in series dict")

    # Optionally HP-filter
    if apply_hp:
        processed = {k: hp_cycle(v, lamb=hp_lambda) for k, v in series.items()}
    else:
        processed = series

    y = processed[output_key]
    moments: Dict[str, float] = {}

    for name, x in processed.items():
        moments[f"var_{name}"]  = _variance(x)
        moments[f"corr_{name}"] = _corr_with_output(x, y)
        moments[f"ac1_{name}"]  = _autocorr1(x)

    return moments


# ---------------------------------------------------------------------------
# Convenience: return both raw and HP-filtered moments together
# ---------------------------------------------------------------------------

def compute_all_moments(
    series: Dict[str, np.ndarray],
    output_key: str = "y",
    hp_lambda: float = 1600.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return (raw_moments, hp_moments) for the same observable series.

    raw_moments  – log-deviations, no filtering (model native)
    hp_moments   – HP-filtered at λ=1600 (matches FRED empirical side)
    """
    raw = compute_moments(series, output_key=output_key, apply_hp=False)
    hp  = compute_moments(series, output_key=output_key, apply_hp=True, hp_lambda=hp_lambda)
    return raw, hp


# ---------------------------------------------------------------------------
# Sanity-check helpers (used by tests and commentary)
# ---------------------------------------------------------------------------

def consumption_smoother_than_output(moments: Dict[str, float]) -> bool:
    """
    Stylized fact: Var(c) < Var(y).
    Returns True when this holds in the given moments dict.
    """
    return moments.get("var_c", float("inf")) < moments.get("var_y", 0.0)


def investment_more_volatile_than_output(moments: Dict[str, float]) -> bool:
    """
    Stylized fact: Var(i) > Var(y).
    Returns True when this holds in the given moments dict.
    """
    return moments.get("var_i", 0.0) > moments.get("var_y", float("inf"))
