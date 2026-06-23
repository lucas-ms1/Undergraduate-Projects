"""Advanced model-layer utilities (optional, advanced-mode only)."""

from __future__ import annotations

import numpy as np


def apply_adjustment_cost(kp: np.ndarray, k: np.ndarray, phi: float) -> np.ndarray:
    """Quadratic adjustment-cost term phi/2 * (kp-k)^2."""
    kp_arr = np.asarray(kp, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    return 0.5 * float(phi) * (kp_arr - k_arr) ** 2


def endogenous_borrowing_limit(y_state: float, floor_share: float = 0.5) -> float:
    """Simple state-dependent borrowing limit proportional to income/wage state."""
    return -abs(float(floor_share) * float(y_state))


def stochastic_interest_rate_states(r_bar: float, spread: float) -> np.ndarray:
    """Two-state interest-rate process values."""
    r0 = float(r_bar) - abs(float(spread))
    r1 = float(r_bar) + abs(float(spread))
    return np.array([r0, r1], dtype=float)
