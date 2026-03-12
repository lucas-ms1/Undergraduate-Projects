"""
Markov-chain utilities.
=======================
Transition-matrix validation and shock-path simulation shared
by all three models.
"""

import numpy as np


def validate_transition_matrix(P: np.ndarray) -> bool:
    """Check that *P* is a valid stochastic matrix.

    Raises ValueError on failure; returns True on success.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square.")
    if np.any(P < 0):
        raise ValueError("Probabilities cannot be negative.")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Each row of the transition matrix must sum to 1.")
    return True


def simulate_shock_path(
    P: np.ndarray,
    T: int,
    initial_idx: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """Draw a length-*T* index path from the Markov chain defined by *P*.

    Returns
    -------
    np.ndarray of int, shape (T,)
        Shock-state indices (0-based).
    """
    P = np.asarray(P, dtype=float)
    rng = np.random.default_rng(seed)
    n_states = P.shape[0]
    path = np.empty(T, dtype=int)
    idx = initial_idx
    for t in range(T):
        path[t] = idx
        idx = rng.choice(n_states, p=P[idx])
    return path
