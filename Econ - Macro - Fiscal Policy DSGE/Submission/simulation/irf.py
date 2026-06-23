"""
simulation/irf.py
Step 10 - Deterministic impulse-response simulation
------------------------------------------------------
Owner: Ben (Steps 10-11)

Simulates x_{t+1} = A x_t + B e_{t+1} with a single shock at t=0 and
zero innovations thereafter.  Returns the full state path plus named
series for the tracked variables.

When a financing rule other than lump_sum is active, an additive
feedback term is applied each period: the financing instrument adjusts
in proportion to the current debt deviation, stabilising debt through
the specified fiscal channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class IRFResult:
    """Container for a deterministic impulse-response run."""

    states: np.ndarray                   # (horizon+1, n_states)
    series: dict[str, np.ndarray]        # {var_name: (horizon+1,) array}


# Financing-rule shock channel mapping
_FINANCING_SHOCK_MAP = {
    "lump_sum":        ("lump_sum", -1.0),   # transfers fall when debt rises
    "consumption_tax": ("tau_c",    +1.0),   # tax rises when debt rises
    "labor_tax":       ("tau_l",    +1.0),
    "capital_tax":     ("tau_k",    +1.0),
    "gc_cut":          ("g_c",      -1.0),   # spending falls when debt rises
}


def simulate_irf(
    A_matrix: np.ndarray,
    B_matrix: np.ndarray,
    shock_vector: np.ndarray,
    variable_index: dict[str, int],
    horizon: int = 40,
    tracked_variables: Iterable[str] = ("y_hat", "c_hat", "i_hat", "b_hat"),
    financing_rule: str = "lump_sum",
    shock_index: Optional[dict[str, int]] = None,
    phi_b: float = 0.05,
) -> IRFResult:
    """
    Simulate x_{t+1} = A x_t + B e_{t+1} with one shock at t=0.

    The returned `states` has shape (horizon + 1, n_states) and includes
    x_0 = 0 in row 0.  Row 1 = B @ shock_vector (the impact period).

    When financing_rule is not "lump_sum", an additive debt-feedback
    term is injected each period through the relevant shock channel.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    n_states = A_matrix.shape[0]
    n_shocks = B_matrix.shape[1]
    states = np.zeros((horizon + 1, n_states), dtype=float)

    # Financing feedback setup
    use_feedback = (
        financing_rule != "lump_sum"
        and financing_rule in _FINANCING_SHOCK_MAP
        and shock_index is not None
        and "b_hat" in variable_index
    )
    if use_feedback:
        fin_shock_name, fin_sign = _FINANCING_SHOCK_MAP[financing_rule]
        fin_shock_idx = shock_index.get(fin_shock_name)
        b_var_idx = variable_index["b_hat"]
        if fin_shock_idx is None:
            use_feedback = False

    for t in range(horizon):
        innovation = shock_vector if t == 0 else np.zeros(n_shocks)
        states[t + 1] = A_matrix @ states[t] + B_matrix @ innovation

        # Additive financing feedback: instrument adjusts proportional to debt
        if use_feedback and t > 0:
            debt_dev = states[t + 1, b_var_idx]
            feedback = np.zeros(n_shocks)
            feedback[fin_shock_idx] = fin_sign * phi_b * debt_dev
            states[t + 1] += B_matrix @ feedback

    series: dict[str, np.ndarray] = {}
    for name in tracked_variables:
        if name not in variable_index:
            raise KeyError(f"Variable index is missing \'{name}\'.")
        series[name] = states[:, variable_index[name]]

    return IRFResult(states=states, series=series)
