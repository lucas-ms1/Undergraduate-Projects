"""Unconditional stochastic simulation."""

import numpy as np


def simulate_paths(A, B, horizon=1000, seed=42, shock_std=1.0, return_shocks=False):
    rng = np.random.default_rng(seed)
    n, m = B.shape
    x = np.zeros((horizon, n))
    eps = rng.normal(0.0, float(shock_std), size=(horizon, m))

    for t in range(horizon - 1):
        x[t + 1] = A @ x[t] + B @ eps[t]
    if return_shocks:
        return x, eps
    return x
