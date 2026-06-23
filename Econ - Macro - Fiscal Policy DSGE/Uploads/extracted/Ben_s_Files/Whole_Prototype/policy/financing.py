"""Financing rule utilities.

Each financing rule specifies HOW the government stabilises debt after
a fiscal shock.  The rule maps to a specific shock channel that feeds
back proportionally to the debt deviation each period.
"""


def financing_channel(rule):
    """Return (shock_index_name, sign) for the financing instrument.

    The sign indicates the direction: +1 means the instrument rises
    when debt is above steady state (tax hikes), -1 means the
    instrument falls (spending cuts / transfer cuts).
    """
    mapping = {
        "Lump-Sum transfers":    (None,  0.0),   # Ricardian: no distortion
        "Consumption Tax Hikes": ("g",  +1.0),   # proxied via demand channel
        "Labor Tax Hikes":       ("tau_l", +1.0),
        "Capital Tax Hikes":     ("tau_k", +1.0),
        "Government Spending Cuts": ("g", -1.0),
    }
    return mapping.get(rule, (None, 0.0))


# Keep legacy helper for any code that still calls it
def financing_sign(rule):
    _, sign = financing_channel(rule)
    return sign
