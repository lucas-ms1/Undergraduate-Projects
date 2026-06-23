import numpy as np

from dsge.calibration import override_parameters
from dsge.steady_state import solve_steady_state
from solvers.state_space import build_structural_system


def _system():
    params = override_parameters()
    ss = solve_steady_state(params)
    return params, ss, build_structural_system(params, ss, financing_rule="lump_sum")


def test_euler_row_coefficients():
    params, _, system = _system()
    row = system["equation_labels"].index("euler_optimizer")
    idx = system["variable_index"]
    beta = params["beta"]
    habit = params["habit"]
    sigma = params["sigma"]
    assert np.isclose(system["Gamma_0"][row, idx["c_o_hat"]], 1.0)
    assert np.isclose(system["Gamma_f"][row, idx["c_o_hat"]], -1.0 / (1.0 + habit))
    assert np.isclose(system["Gamma_l"][row, idx["c_o_hat"]], -habit / (1.0 + habit))
    # No direct forward pi_hat term — expected inflation enters via r_real_hat (Fisher)
    assert np.isclose(system["Gamma_f"][row, idx["pi_hat"]], 0.0)
    assert np.isclose(system["Gamma_0"][row, idx["r_real_hat"]], (1.0 - habit) / (sigma * (1.0 + habit)))


def test_nkpc_and_wage_pc_coefficients():
    params, _, system = _system()
    idx = system["variable_index"]
    beta = params["beta"]
    theta_p = params["theta_p"]
    theta_w = params["theta_w"]
    phi_l = params["phi_l"]
    eps_w = params["eps_w"]
    kappa_p = (1.0 - theta_p) * (1.0 - beta * theta_p) / theta_p
    kappa_w = (1.0 - theta_w) * (1.0 - beta * theta_w) / (theta_w * (1.0 + phi_l * eps_w))

    row_nk = system["equation_labels"].index("nkpc")
    assert np.isclose(system["Gamma_f"][row_nk, idx["pi_hat"]], -beta)
    assert np.isclose(system["Gamma_0"][row_nk, idx["mc_hat"]], -kappa_p)

    row_w = system["equation_labels"].index("wage_phillips")
    assert np.isclose(system["Gamma_f"][row_w, idx["pi_w_hat"]], -beta)
    assert np.isclose(system["Gamma_0"][row_w, idx["w_hat"]], kappa_w)


def test_marginal_cost_capital_accumulation_and_resource_rows():
    _, ss, system = _system()
    idx = system["variable_index"]

    row_mc = system["equation_labels"].index("marginal_cost_identity")
    assert np.isclose(system["Gamma_0"][row_mc, idx["mc_hat"]], 1.0)
    assert np.isclose(system["Gamma_0"][row_mc, idx["a_hat"]], 1.0)

    row_k = system["equation_labels"].index("capital_accumulation")
    assert np.isclose(system["Gamma_0"][row_k, idx["k_hat"]], 1.0)
    assert np.isclose(system["Gamma_0"][row_k, idx["g_i_hat"]], -(ss["G_i"] / ss["K"]))

    row_res = system["equation_labels"].index("resource_constraint")
    assert np.isclose(system["Gamma_0"][row_res, idx["c_hat"]], -ss["c_y"])
    assert np.isclose(system["Gamma_0"][row_res, idx["i_hat"]], -ss["i_y"])
    assert np.isclose(system["Gamma_0"][row_res, idx["g_c_hat"]], -(ss["G_c"] / ss["Y"]))
    assert np.isclose(system["Gamma_0"][row_res, idx["g_i_hat"]], -(ss["G_i"] / ss["Y"]))


def test_government_budget_signs():
    params, ss, system = _system()
    idx = system["variable_index"]
    row = system["equation_labels"].index("government_budget")
    assert np.isclose(system["Gamma_0"][row, idx["g_c_hat"]], -(ss["G_c"] / ss["B"]))
    assert np.isclose(system["Gamma_0"][row, idx["g_i_hat"]], -(ss["G_i"] / ss["B"]))
    assert np.isclose(system["Gamma_0"][row, idx["t_hat"]], -(abs(ss["T"]) / ss["B"]))
    assert np.isclose(system["Gamma_0"][row, idx["tau_c_hat"]], params["tau_c"] * ss["C"] / ss["B"])
    assert np.isclose(system["Gamma_0"][row, idx["c_hat"]], params["tau_c"] * ss["C"] / ss["B"])
    assert np.isclose(system["Gamma_0"][row, idx["tau_l_hat"]], params["tau_l"] * ss["w"] * ss["L"] / ss["B"])
    assert np.isclose(system["Gamma_0"][row, idx["tau_k_hat"]], params["tau_k"] * ss["rk"] * ss["K"] / ss["B"])
    assert np.isclose(system["Gamma_0"][row, idx["rk_hat"]], params["tau_k"] * ss["rk"] * ss["K"] / ss["B"])


def test_fisher_and_taylor_rows():
    params, _, system = _system()
    idx = system["variable_index"]

    row_fisher = system["equation_labels"].index("fisher_relation")
    assert np.isclose(system["Gamma_0"][row_fisher, idx["r_real_hat"]], 1.0)
