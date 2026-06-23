"""
policy/financing.py
Step 10 – Financing-rule helpers
----------------------------------
Owner: Ben (Steps 10-11)

Implements the five financing rules from the plan §3.4:
  • Lump-Sum transfers
  • Consumption Tax Hikes
  • Labor Tax Hikes
  • Capital Tax Hikes
  • Government Spending Cuts (Gc cut)

Key sign convention (plan §3.4):
  • Tax instruments: φ_b > 0  (higher debt → higher taxes)
  • Lump-sum:        φ_b < 0  (higher debt → lower transfers)
  • Gc-cut:          φ_b < 0  (higher debt → lower spending)
  The UI slider exposes the magnitude; the sign is applied HERE.
"""

from __future__ import annotations


FINANCING_RULES = (
    "lump_sum",
    "consumption_tax",
    "labor_tax",
    "capital_tax",
    "gc_cut",
)

# Map UI labels → canonical names
_UI_TO_CANONICAL = {
    "Lump-Sum transfers":       "lump_sum",
    "Consumption Tax Hikes":    "consumption_tax",
    "Labor Tax Hikes":          "labor_tax",
    "Capital Tax Hikes":        "capital_tax",
    "Government Spending Cuts": "gc_cut",
}


def ui_label_to_canonical(label: str) -> str:
    """Convert a Streamlit selectbox label to a canonical financing rule."""
    if label in _UI_TO_CANONICAL:
        return _UI_TO_CANONICAL[label]
    raise ValueError(f"Unknown UI financing label '{label}'.")


def debt_feedback_sign(rule: str) -> float:
    """
    Return the sign applied to |φ_b| based on financing instrument.

    Step-10 sign convention (plan §3.4):
    - taxes react positively to debt       → +φ_b
    - lump-sum transfers shrink with debt  → -φ_b
    - gc-cut reduces spending with debt    → -φ_b
    """
    canonical = rule.strip().lower()
    if canonical not in FINANCING_RULES:
        allowed = ", ".join(FINANCING_RULES)
        raise ValueError(f"Unknown financing rule '{rule}'. Allowed: {allowed}")

    if canonical in {"lump_sum", "gc_cut"}:
        return -1.0
    return 1.0


def build_financing_coefficients(phi_b: float, rule: str) -> dict[str, float]:
    """
    Build instrument-specific coefficients for the debt feedback rule.

    Returns coefficients for all instruments, but only one gets a non-zero
    response in the selected rule.
    """
    magnitude = abs(phi_b)
    signed_phi = debt_feedback_sign(rule) * magnitude

    coeffs = {
        "phi_b_lump_sum": 0.0,
        "phi_b_tau_c":    0.0,
        "phi_b_tau_l":    0.0,
        "phi_b_tau_k":    0.0,
        "phi_b_gc":       0.0,
    }
    target = {
        "lump_sum":       "phi_b_lump_sum",
        "consumption_tax": "phi_b_tau_c",
        "labor_tax":       "phi_b_tau_l",
        "capital_tax":     "phi_b_tau_k",
        "gc_cut":          "phi_b_gc",
    }[rule.strip().lower()]
    coeffs[target] = signed_phi
    return coeffs
