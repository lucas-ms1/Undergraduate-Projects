"""
Off-grid policy evaluation.
============================
Will wrap numpy.interp so simulation and forecasting modules
can evaluate policy functions at arbitrary state values.

Implementation deferred to Step 4.
"""

import numpy as np


def interp_policy(grid: np.ndarray, policy: np.ndarray,
                  query_points: np.ndarray) -> np.ndarray:
    """Evaluate a 1D policy function off-grid using linear interpolation.

    Parameters
    ----------
    grid : np.ndarray
        Monotone 1D state grid (e.g. assets, capital).
    policy : np.ndarray
        Policy values defined on ``grid`` (same length).
    query_points : np.ndarray
        State values at which to evaluate the policy. Can be scalar
        or array-like.

    Returns
    -------
    np.ndarray
        Interpolated policy values with the same shape as
        ``query_points``.

    Notes
    -----
    This is a thin wrapper around ``numpy.interp`` that:
      - Works with scalar or array ``query_points``.
      - Clips out-of-bounds queries to the endpoints of ``grid``
        (matching ``numpy.interp``'s default behaviour).
    """
    grid = np.asarray(grid, dtype=float)
    policy = np.asarray(policy, dtype=float)
    q = np.asarray(query_points, dtype=float)

    # numpy.interp requires 1D inputs; we preserve the original shape
    original_shape = q.shape
    q_flat = q.ravel()

    interpolated = np.interp(q_flat, grid, policy)
    return interpolated.reshape(original_shape)
