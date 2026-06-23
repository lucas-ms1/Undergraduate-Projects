"""Step 11 multiplier and fiscal-drag metrics."""

from __future__ import annotations

import numpy as np


def to_level_response(log_deviation_irf: np.ndarray, steady_state_level: float) -> np.ndarray:
    """Convert log-deviation IRF into level deviations using a steady-state level."""
    return steady_state_level * np.asarray(log_deviation_irf, dtype=float)


def impact_multiplier(
    y_log_irf: np.ndarray,
    g_log_irf: np.ndarray,
    y_ss: float,
    g_ss: float,
) -> float:
    """IM = Delta Y_0 / Delta G_0 in level terms."""
    dy0 = to_level_response(y_log_irf, y_ss)[0]
    dg0 = to_level_response(g_log_irf, g_ss)[0]
    if np.isclose(dg0, 0.0):
        raise ZeroDivisionError("Impact shock is zero; impact multiplier undefined.")
    return float(dy0 / dg0)


def cumulative_multiplier(
    y_log_irf: np.ndarray,
    g_log_irf: np.ndarray,
    y_ss: float,
    g_ss: float,
    beta: float = 0.99,
    horizon: int | None = None,
) -> float:
    """Discounted cumulative multiplier over [0, H]."""
    y_level = to_level_response(y_log_irf, y_ss)
    g_level = to_level_response(g_log_irf, g_ss)

    h = len(y_level) - 1 if horizon is None else min(horizon, len(y_level) - 1)
    weights = np.power(beta, np.arange(h + 1, dtype=float))
    num = np.sum(weights * y_level[: h + 1])
    den = np.sum(weights * g_level[: h + 1])

    if np.isclose(den, 0.0):
        raise ZeroDivisionError("Cumulative fiscal shock sum is zero; CM undefined.")
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
