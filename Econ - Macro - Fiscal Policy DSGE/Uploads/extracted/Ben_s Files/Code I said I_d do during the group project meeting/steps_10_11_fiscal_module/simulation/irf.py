"""Step 10 deterministic IRF simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IRFResult:
    """Container for a deterministic impulse response run."""

    states: np.ndarray
    series: dict[str, np.ndarray]


def simulate_irf(
    A_matrix: np.ndarray,
    B_matrix: np.ndarray,
    shock_vector: np.ndarray,
    variable_index: dict[str, int],
    horizon: int = 40,
    tracked_variables: Iterable[str] = ("y_hat", "c_hat", "i_hat", "b_hat"),
) -> IRFResult:
    """
    Simulate x_{t+1} = A x_t + B e_{t+1} with one shock at t=0.

    The returned `states` has shape (horizon + 1, n_states) and includes
    x_0 = 0 in row 0.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    n_states = A_matrix.shape[0]
    states = np.zeros((horizon + 1, n_states), dtype=float)

    for t in range(horizon):
        innovation = shock_vector if t == 0 else np.zeros_like(shock_vector)
        states[t + 1] = A_matrix @ states[t] + B_matrix @ innovation

    series: dict[str, np.ndarray] = {}
    for name in tracked_variables:
        if name not in variable_index:
            raise KeyError(f"Variable index is missing '{name}'.")
        series[name] = states[:, variable_index[name]]

    return IRFResult(states=states, series=series)
