import numpy as np

from steps_10_11_fiscal_module.policy.financing import build_financing_coefficients
from steps_10_11_fiscal_module.policy.multipliers import (
    cumulative_multiplier,
    fiscal_drag_horizon,
    impact_multiplier,
)


def test_financing_rule_signs_match_step_10_convention():
    phi_b = 0.05
    lump_sum = build_financing_coefficients(phi_b, "lump_sum")
    gc_cut = build_financing_coefficients(phi_b, "gc_cut")
    tau_k = build_financing_coefficients(phi_b, "capital_tax")

    assert np.isclose(lump_sum["phi_b_lump_sum"], -phi_b)
    assert np.isclose(gc_cut["phi_b_gc"], -phi_b)
    assert np.isclose(tau_k["phi_b_tau_k"], phi_b)


def test_impact_and_cumulative_multipliers_with_simple_paths():
    y_log = np.array([0.0100, 0.0050, 0.0025, 0.00125])
    g_log = np.array([0.0100, 0.0040, 0.0016, 0.00064])
    y_ss, g_ss = 1.0, 0.2

    im = impact_multiplier(y_log, g_log, y_ss=y_ss, g_ss=g_ss)
    cm = cumulative_multiplier(y_log, g_log, y_ss=y_ss, g_ss=g_ss, beta=0.99, horizon=3)

    # With Y_ss / G_ss = 5 and equal impact log shocks, IM should be exactly 5.
    assert np.isclose(im, 5.0)
    assert cm > 0.0


def test_fiscal_drag_horizon_detects_first_negative_quarter():
    y_log = np.array([0.01, 0.006, 0.001, -0.0005, -0.001])
    assert fiscal_drag_horizon(y_log) == 3

    y_nonnegative = np.array([0.01, 0.006, 0.002, 0.0001])
    assert fiscal_drag_horizon(y_nonnegative) is None
