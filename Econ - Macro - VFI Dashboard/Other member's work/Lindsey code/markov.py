"""
Markov-chain utilities.
=======================
Will provide:
    - validate_transition_matrix(P) – check shape, non-negative, rows sum to 1
    - simulate_shock_path(P, T, initial_idx, seed) – draw a shock index path

Implementation deferred to Step 4.
"""

import numpy as np


def validate_transition_matrix(P: np.ndarray) -> bool:
    """Validate a Markov transition matrix.

    Checks:
      - 2D square array
      - All entries non-negative
      - Each row sums to 1 (within a small tolerance)

    Parameters
    ----------
    P : np.ndarray
        Candidate transition matrix.

    Returns
    -------
    bool
        True if ``P`` passes all checks, False otherwise.
    """
    P = np.asarray(P, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    if np.any(P < 0):
        return False

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        return False

    return True


def simulate_shock_path(P: np.ndarray, T: int,
                        initial_idx: int = 0,
                        seed: int = 42) -> np.ndarray:
    """Simulate a sequence of discrete shock indices from a Markov chain.

    Parameters
    ----------
    P : np.ndarray
        Transition matrix, shape (n_states, n_states).
    T : int
        Length of the desired path.
    initial_idx : int
        Index of the initial shock state.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of length ``T`` with integer shock indices. The first
        element is ``initial_idx``.
    """
    P = np.asarray(P, dtype=float)

    if not validate_transition_matrix(P):
        raise ValueError("Invalid transition matrix P.")

    n_states = P.shape[0]
    if not (0 <= initial_idx < n_states):
        raise ValueError("initial_idx out of bounds for transition matrix.")

    rng = np.random.default_rng(seed)
    shocks = np.empty(T, dtype=int)
    shocks[0] = initial_idx

    for t in range(1, T):
        current = shocks[t - 1]
        shocks[t] = rng.choice(n_states, p=P[current])

    return shocks
