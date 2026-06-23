"""Step 10 fiscal shock definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiscalShock:
    """Single-period innovation vector for a fiscal experiment."""

    name: str
    vector: np.ndarray


def build_unit_impulse_shock(
    shock_name: str,
    shock_index: dict[str, int],
    impulse_size: float = 1.0,
) -> FiscalShock:
    """
    Build the Step-10 unit impulse to a selected innovation.

    Mapping:
    - `gc`: positive government-consumption shock
    - `gi`: positive government-investment shock
    - `tau_l_cut`: negative labor-tax innovation
    - `tau_k_cut`: negative capital-tax innovation
    """
    canonical = shock_name.strip().lower()
    key_map = {
        "gc": "eps_gc",
        "gi": "eps_gi",
        "tau_l_cut": "eps_tau_l",
        "tau_k_cut": "eps_tau_k",
    }
    if canonical not in key_map:
        allowed = ", ".join(sorted(key_map))
        raise ValueError(f"Unknown fiscal shock '{shock_name}'. Allowed: {allowed}")

    vector = np.zeros(len(shock_index), dtype=float)
    idx_key = key_map[canonical]
    if idx_key not in shock_index:
        raise KeyError(f"Shock index is missing '{idx_key}'.")

    signed_impulse = impulse_size
    if canonical in {"tau_l_cut", "tau_k_cut"}:
        signed_impulse = -abs(impulse_size)

    vector[shock_index[idx_key]] = signed_impulse
    return FiscalShock(name=canonical, vector=vector)
