import numpy as np

from dsge.calibration import override_parameters
from dsge.model import solve_model_objects
from policy.financing import build_financing_coefficients
from policy.multipliers import cumulative_multiplier, impact_multiplier
from policy.shocks import build_unit_impulse_shock
from simulation.irf import simulate_irf


def test_only_selected_financing_instrument_moves():
    keys = (
        "phi_b_lump_sum",
        "phi_b_tau_c",
        "phi_b_tau_l",
        "phi_b_tau_k",
        "phi_b_gc",
    )
    for rule, expected in (
        ("lump_sum", "phi_b_lump_sum"),
        ("consumption_tax", "phi_b_tau_c"),
        ("labor_tax", "phi_b_tau_l"),
        ("capital_tax", "phi_b_tau_k"),
        ("gc_cut", "phi_b_gc"),
    ):
        coeffs = build_financing_coefficients(0.05, rule)
        nonzero = [key for key in keys if coeffs[key] != 0.0]
        assert nonzero == [expected]


def test_financing_rules_hit_different_rows():
    base = override_parameters()
    lump = solve_model_objects(base, financing_rule="lump_sum")["structural_system"]
    capital = solve_model_objects(base, financing_rule="capital_tax")["structural_system"]
    vidx = lump["variable_index"]
    row_t = lump["equation_labels"].index("lump_sum_rule")
    row_tau_k = lump["equation_labels"].index("capital_tax_rule")
    assert lump["Gamma_0"][row_t, vidx["b_hat"]] != 0.0
    assert lump["Gamma_0"][row_tau_k, vidx["b_hat"]] == 0.0
    assert capital["Gamma_0"][row_t, vidx["b_hat"]] == 0.0
    assert capital["Gamma_0"][row_tau_k, vidx["b_hat"]] != 0.0


def test_capital_tax_financing_changes_gc_cumulative_multiplier():
    params = override_parameters()
    lump = solve_model_objects(params, financing_rule="lump_sum")
    capital = solve_model_objects(params, financing_rule="capital_tax")
    shock_lump = build_unit_impulse_shock("gc", lump["shock_index"])
    shock_capital = build_unit_impulse_shock("gc", capital["shock_index"])
    tracked = ("y_hat", "g_c_hat")
    irf_lump = simulate_irf(
        lump["A_matrix"], lump["B_matrix"], shock_lump.vector, lump["variable_index"], 40, tracked
    )
    irf_capital = simulate_irf(
        capital["A_matrix"],
        capital["B_matrix"],
        shock_capital.vector,
        capital["variable_index"],
        40,
        tracked,
    )
    cm_lump = cumulative_multiplier(
        irf_lump.series["y_hat"][1:],
        irf_lump.series["g_c_hat"][1:],
        lump["steady_state"]["Y"],
        lump["steady_state"]["G_c"],
        beta=params["beta"],
    )
    cm_capital = cumulative_multiplier(
        irf_capital.series["y_hat"][1:],
        irf_capital.series["g_c_hat"][1:],
        capital["steady_state"]["Y"],
        capital["steady_state"]["G_c"],
        beta=params["beta"],
    )
    assert np.isfinite(cm_capital)
    assert np.isfinite(cm_lump)
    assert not np.isclose(cm_capital, cm_lump)


def test_cumulative_multiplier_is_well_defined():
    y_log = np.array([0.01, 0.005, 0.002])
    g_log = np.array([0.01, 0.004, 0.001])
    value = cumulative_multiplier(y_log, g_log, y_ss=1.0, g_ss=0.2, beta=0.99)
    assert np.isfinite(value)


def test_multipliers_do_not_apply_steady_state_y_over_g_scaling():
    y_log = np.array([0.02, 0.01])
    g_log = np.array([0.01, 0.005])

    im = impact_multiplier(y_log, g_log, y_ss=1.0, g_ss=0.2)
    cm = cumulative_multiplier(y_log, g_log, y_ss=1.0, g_ss=0.2, beta=0.99)

    assert np.isclose(im, 2.0)
    assert np.isclose(cm, 2.0)


def test_impact_multiplier_accepts_full_irf_with_leading_zero_row():
    y_log = np.array([0.0, 0.02, 0.01])
    g_log = np.array([0.0, 0.01, 0.005])
    value = impact_multiplier(y_log, g_log, y_ss=1.0, g_ss=0.5)
    assert np.isfinite(value)
    assert np.isclose(value, 2.0)


def test_impact_multiplier_accepts_impact_aligned_irf():
    y_log = np.array([0.02, 0.01])
    g_log = np.array([0.01, 0.005])
    value = impact_multiplier(y_log, g_log, y_ss=1.0, g_ss=0.5)
    assert np.isfinite(value)
    assert np.isclose(value, 2.0)
