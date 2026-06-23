"""Fiscal shock specifications."""


def shock_vector_name(shock_name):
    mapping = {
        "Gc Shock": "g",
        "GI Shock": "g",
        "Labor-Tax Cut": "tau_l",
        "Capital-Tax Cut": "tau_k",
    }
    return mapping[shock_name]
