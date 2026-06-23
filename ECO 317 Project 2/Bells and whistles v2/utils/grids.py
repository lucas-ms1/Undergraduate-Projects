"""
Grid construction helpers.
==========================
Provides linear and curved grids for state-space discretisation,
plus clipping / bounds-check utilities shared by all three models.
"""

import numpy as np


def linear_grid(x_min: float, x_max: float, n: int) -> np.ndarray:
    """Evenly spaced grid from *x_min* to *x_max* with *n* points."""
    return np.linspace(x_min, x_max, n)


def curved_grid(x_min: float, x_max: float, n: int) -> np.ndarray:
    """Grid denser near *x_min* (captures curvature in value function)."""
    unit = np.linspace(0.0, 1.0, n) ** 2
    return unit * (x_max - x_min) + x_min


def clip_to_grid(value, x_min: float, x_max: float):
    """Clip *value* (scalar or array) to [x_min, x_max]."""
    return np.clip(value, x_min, x_max)


def is_consumption_positive(c) -> bool:
    """True when every element of *c* is strictly positive."""
    return bool(np.all(np.asarray(c) > 0))


def is_labor_feasible(L, strict: bool = False) -> bool:
    """True when labour lies in [0, 1] (strict ⇒ (0, 1))."""
    L = np.asarray(L)
    if strict:
        return bool(np.all((L > 0) & (L < 1)))
    return bool(np.all((L >= 0) & (L <= 1)))
