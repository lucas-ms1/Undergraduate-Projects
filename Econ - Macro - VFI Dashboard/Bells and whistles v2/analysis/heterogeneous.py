"""Minimal heterogeneous-agent scaffolding for advanced roadmap."""

from __future__ import annotations

import numpy as np
import pandas as pd


def iterate_stationary_distribution(policy_indices: np.ndarray, P: np.ndarray, max_iter: int = 400, tol: float = 1e-9):
    """Compute stationary distribution over (asset_index, shock_index)."""
    pol = np.asarray(policy_indices, dtype=int)
    n_a, n_s = pol.shape
    dist = np.full((n_a, n_s), 1.0 / (n_a * n_s))
    P = np.asarray(P, dtype=float)
    for _ in range(int(max_iter)):
        nxt = np.zeros_like(dist)
        for i in range(n_a):
            for s in range(n_s):
                ip = int(np.clip(pol[i, s], 0, n_a - 1))
                nxt[ip, :] += dist[i, s] * P[s, :]
        if float(np.max(np.abs(nxt - dist))) < tol:
            dist = nxt
            break
        dist = nxt
    return dist


def summarize_distribution(dist: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    d = np.asarray(dist, dtype=float)
    g = np.asarray(grid, dtype=float)
    mass_by_a = d.sum(axis=1)
    mean_assets = float(np.sum(mass_by_a * g))
    return pd.DataFrame([{"mean_assets": mean_assets, "mass_at_borrowing_limit": float(mass_by_a[0])}])
