"""Unconditional stochastic simulation."""

import numpy as np


def simulate_paths(A, B, horizon=1000, seed=42):
    rng = np.random.default_rng(seed)
    n, m = B.shape
    x = np.zeros((horizon, n))
    eps = rng.normal(0.0, 1.0, size=(horizon, m))

    for t in range(1, horizon):
        x[t] = A @ x[t - 1] + B @ eps[t]
    return x
