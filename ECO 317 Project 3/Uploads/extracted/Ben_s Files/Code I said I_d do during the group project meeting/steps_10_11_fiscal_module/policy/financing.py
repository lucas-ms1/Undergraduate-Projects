"""Step 10 financing-rule helpers."""

from __future__ import annotations


FINANCING_RULES = (
    "lump_sum",
    "consumption_tax",
    "labor_tax",
    "capital_tax",
    "gc_cut",
)


def debt_feedback_sign(rule: str) -> float:
    """
    Return the sign applied to `phi_b` based on financing instrument.

    Step-10 sign convention:
    - taxes react positively to debt -> +phi_b
    - transfer instrument is signed as transfer-to-households -> -phi_b
    - gc-cut instrument reduces spending as debt rises -> -phi_b
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
        "phi_b_tau_c": 0.0,
        "phi_b_tau_l": 0.0,
        "phi_b_tau_k": 0.0,
        "phi_b_gc": 0.0,
    }
    target = {
        "lump_sum": "phi_b_lump_sum",
        "consumption_tax": "phi_b_tau_c",
        "labor_tax": "phi_b_tau_l",
        "capital_tax": "phi_b_tau_k",
        "gc_cut": "phi_b_gc",
    }[rule.strip().lower()]
    coeffs[target] = signed_phi
    return coeffs
