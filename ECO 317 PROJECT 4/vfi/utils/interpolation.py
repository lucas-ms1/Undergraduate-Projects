"""
Off-grid policy evaluation.
============================
Wraps numpy.interp so simulation and forecasting modules can
evaluate policy functions at arbitrary state values using one
shared interpolation rule.
"""

import numpy as np


def interp_policy(
    grid: np.ndarray,
    policy: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate *policy* defined on *grid* at *query_points*.

    Values outside the grid are clamped to the nearest boundary value
    (numpy.interp default behaviour).

    Parameters
    ----------
    grid : 1-D array
        Sorted state-space grid on which *policy* is defined.
    policy : 1-D array, same length as *grid*
        Policy values at each grid node.
    query_points : scalar or array
        Points at which to evaluate the policy.

    Returns
    -------
    np.ndarray
        Interpolated policy values.
    """
    return np.interp(np.asarray(query_points), grid, policy)
