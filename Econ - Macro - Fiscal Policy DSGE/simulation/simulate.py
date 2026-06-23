"""
simulation/simulate.py
Step 8 – Unconditional simulation
-----------------------------------
Owner: Steps 8-9 contributor

Given state-space matrices A (n×n) and B (n×k), draw T periods of i.i.d.
normal shocks and propagate  x_{t+1} = A x_t + B ε_{t+1}.

The model produces log-deviations from steady state directly, so no
detrending is applied here.  HP-filtering lives in moments.py and is
used only when comparing to empirical FRED moments.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Core simulator
# ──────────────────────────────────────────────────────────────────────────────

def simulate(
    A: np.ndarray,
    B: np.ndarray,
    T: int = 1000,
    seed: int = 42,
    x0: Optional[np.ndarray] = None,
    shock_stds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate the linear state-space model x_{t+1} = A x_t + B ε_{t+1}.

    Parameters
    ----------
    A    : (n, n) transition matrix
    B    : (n, k) shock-loading matrix
    T    : number of periods to simulate (default 1,000)
    seed : RNG seed for reproducibility
    x0   : initial state vector (n,); defaults to zeros

    Returns
    -------
    X : (T, n) array of simulated states (log-deviations from SS)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    k = B.shape[1]

    if x0 is None:
        x0 = np.zeros(n)

    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((T, k))   # i.i.d. N(0,I) shocks
    if shock_stds is not None:
        scales = np.asarray(shock_stds, dtype=float)
        if scales.shape != (k,):
            raise ValueError(f"shock_stds must have shape ({k},), got {scales.shape}.")
        eps = eps * scales

    X = np.empty((T, n))
    x = x0.copy()
    for t in range(T):
        x = A @ x + B @ eps[t]
        X[t] = x

    return X


# ──────────────────────────────────────────────────────────────────────────────
# Observable extraction
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_OBS_INDEX: Dict[str, int] = {
    "y": 0,
    "c": 3,
    "i": 4,
    "l": 12,
    "pi": 10,
    "r": 14,
}


def extract_observables(
    X: np.ndarray,
    obs_index: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Slice the simulation array into named observable series.

    Parameters
    ----------
    X         : (T, n) simulation output from `simulate`
    obs_index : mapping from name → column index in X
                (defaults to DEFAULT_OBS_INDEX)

    Returns
    -------
    dict of {name: (T,) array}
    """
    if obs_index is None:
        obs_index = DEFAULT_OBS_INDEX

    return {name: X[:, idx] for name, idx in obs_index.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: one-shot simulate + extract
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(
    A: np.ndarray,
    B: np.ndarray,
    T: int = 1000,
    seed: int = 42,
    obs_index: Optional[Dict[str, int]] = None,
    x0: Optional[np.ndarray] = None,
    shock_stds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate the model and return a dict of observable series.

    Returns
    -------
    dict with keys matching obs_index (default: y, c, i, l, pi)
    """
    X = simulate(A, B, T=T, seed=seed, x0=x0, shock_stds=shock_stds)
    return extract_observables(X, obs_index=obs_index)
