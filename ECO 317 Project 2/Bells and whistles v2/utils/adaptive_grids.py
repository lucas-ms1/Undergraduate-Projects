"""Helpers for simple adaptive one-dimensional grids."""

from __future__ import annotations

import numpy as np


def curvature_weights(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or vals.size < 3:
        return np.ones_like(vals, dtype=float)
    second = np.zeros_like(vals)
    second[1:-1] = np.abs(vals[2:] - 2.0 * vals[1:-1] + vals[:-2])
    w = second + eps
    w /= np.sum(w)
    return w


def refine_grid_from_policy(base_grid: np.ndarray, policy: np.ndarray, n_points: int) -> np.ndarray:
    """Create a refined grid by allocating density to high-curvature regions."""
    g = np.asarray(base_grid, dtype=float)
    p = np.asarray(policy, dtype=float)
    n_points = int(max(8, n_points))
    weights = curvature_weights(p)
    cdf = np.cumsum(weights)
    cdf = cdf / max(cdf[-1], 1e-12)
    u = np.linspace(0.0, 1.0, n_points)
    refined = np.interp(u, cdf, g)
    refined[0], refined[-1] = g[0], g[-1]
    return np.unique(refined)
