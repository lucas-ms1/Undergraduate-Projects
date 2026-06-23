"""
policy/multipliers.py
Step 11 - Fiscal multipliers and drag horizon
------------------------------------------------
Owner: Ben (Steps 10-11)

Computes:
  - Impact multiplier:     IM = y_hat_0 / g_hat_0
  - Cumulative multiplier: CM(H) = sum beta^t y_hat_t / sum beta^t g_hat_t
  - Fiscal drag horizon:   smallest t >= 1 where y_hat_t < 0

The dashboard reports multipliers from normalized IRFs.  The steady-state
arguments remain in the public API for compatibility, but they are not applied
as Y/G scale factors.
"""

from __future__ import annotations

import numpy as np


def to_level_response(log_deviation_irf: np.ndarray, steady_state_level: float) -> np.ndarray:
    """Convert log-deviation IRF into level deviations using a steady-state level."""
    return steady_state_level * np.asarray(log_deviation_irf, dtype=float)


def _drop_leading_steady_state(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Accept impact-aligned IRFs and full simulate_irf() output with row 0 included.

    simulate_irf() keeps row 0 as the pre-shock steady state, so the first
    denominator can be zero even though the impact response is in row 1.
    """
    converted = tuple(np.asarray(arr, dtype=float) for arr in arrays)
    if not converted or any(len(arr) == 0 for arr in converted):
        return converted
    first_values_zero = all(np.isclose(arr[0], 0.0) for arr in converted)
    later_values_move = any(np.any(~np.isclose(arr[1:], 0.0)) for arr in converted)
    if first_values_zero and later_values_move:
        return tuple(arr[1:] for arr in converted)
    return converted


def impact_multiplier(
    y_log_irf: np.ndarray,
    g_log_irf: np.ndarray,
    y_ss: float,
    g_ss: float,
) -> float:
    """IM = y_hat_0 / g_hat_0 on the normalized IRF scale."""
    y_log_irf, g_log_irf = _drop_leading_steady_state(y_log_irf, g_log_irf)
    dy0 = np.asarray(y_log_irf, dtype=float)[0]
    dg0 = np.asarray(g_log_irf, dtype=float)[0]
    if np.isclose(dg0, 0.0):
        return float("nan")
    return float(dy0 / dg0)


def cumulative_multiplier(
    y_log_irf: np.ndarray,
    g_log_irf: np.ndarray,
    y_ss: float,
    g_ss: float,
    beta: float = 0.99,
    horizon: int | None = None,
) -> float:
    """Discounted cumulative multiplier over [0, H] on the normalized IRF scale."""
    y_log_irf, g_log_irf = _drop_leading_steady_state(y_log_irf, g_log_irf)
    y_response = np.asarray(y_log_irf, dtype=float)
    g_response = np.asarray(g_log_irf, dtype=float)

    h = len(y_response) - 1 if horizon is None else min(horizon, len(y_response) - 1)
    weights = np.power(beta, np.arange(h + 1, dtype=float))
    num = np.sum(weights * y_response[: h + 1])
    den = np.sum(weights * g_response[: h + 1])

    if np.isclose(den, 0.0):
        return float("nan")
    return float(num / den)


def fiscal_drag_horizon(y_log_irf: np.ndarray) -> int | None:
    """
    Return smallest t >= 1 where output response crosses below zero.

    Returns None when output does not turn negative in the available horizon.
    """
    y = np.asarray(y_log_irf, dtype=float)
    for t in range(1, len(y)):
        if y[t] < 0.0:
            return t
    return None
