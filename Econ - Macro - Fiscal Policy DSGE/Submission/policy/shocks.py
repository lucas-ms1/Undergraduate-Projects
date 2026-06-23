"""
policy/shocks.py
Step 10 – Fiscal shock definitions
------------------------------------
Owner: Ben (Steps 10-11)

Defines unit-impulse shock vectors for fiscal experiments in Tab 2:
  • Gc shock  (positive government consumption)
  • GI shock  (positive government investment)
  • τ_L cut   (negative labor-tax innovation)
  • τ_K cut   (negative capital-tax innovation)
"""

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
    - `gc`: positive government-consumption shock  → eps_g_c
    - `gi`: positive government-investment shock   → eps_g_i
    - `tau_l_cut`: negative labor-tax innovation    → eps_tau_l
    - `tau_k_cut`: negative capital-tax innovation  → eps_tau_k
    """
    canonical = shock_name.strip().lower()
    key_map = {
        "gc":         "g_c",
        "gi":         "g_i",
        "tau_l_cut":  "tau_l",
        "tau_k_cut":  "tau_k",
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


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: map UI labels to canonical shock names
# ──────────────────────────────────────────────────────────────────────────────

_UI_TO_CANONICAL = {
    "Gc Shock":         "gc",
    "GI Shock":         "gi",
    "Labor-Tax Cut":    "tau_l_cut",
    "Capital-Tax Cut":  "tau_k_cut",
}


def ui_label_to_canonical(label: str) -> str:
    """Convert a Streamlit selectbox label to a canonical shock name."""
    if label in _UI_TO_CANONICAL:
        return _UI_TO_CANONICAL[label]
    raise ValueError(f"Unknown UI shock label '{label}'.")
