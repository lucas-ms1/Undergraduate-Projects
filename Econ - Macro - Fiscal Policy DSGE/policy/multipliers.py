"""
policy/multipliers.py
Step 11 – Fiscal multipliers and drag horizon
------------------------------------------------
Owner: Ben (Steps 10-11)

Computes:
  • Impact multiplier:     IM = ΔY_0 / ΔG_0  (in level terms)
  • Cumulative multiplier: CM(H) = Σ β^t ΔY_t / Σ β^t ΔG_t
  • Fiscal drag horizon:   smallest t ≥ 1 where ΔY_t < 0

IMPORTANT (plan §3, hallucination trap):
  Converting log-deviation IRFs to level multipliers requires multiplying
  by the steady-state Y/G ratio.  Forgetting this scales the multiplier
  by 5–10× in either direction.
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
    """IM = ΔY_0 / ΔG_0 in level terms."""
    y_log_irf, g_log_irf = _drop_leading_steady_state(y_log_irf, g_log_irf)
    dy0 = to_level_response(y_log_irf, y_ss)[0]
    dg0 = to_level_response(g_log_irf, g_ss)[0]
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
    """Discounted cumulative multiplier over [0, H]."""
    y_log_irf, g_log_irf = _drop_leading_steady_state(y_log_irf, g_log_irf)
    y_level = to_level_response(y_log_irf, y_ss)
    g_level = to_level_response(g_log_irf, g_ss)

    h = len(y_level) - 1 if horizon is None else min(horizon, len(y_level) - 1)
    weights = np.power(beta, np.arange(h + 1, dtype=float))
    num = np.sum(weights * y_level[: h + 1])
    den = np.sum(weights * g_level[: h + 1])

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
