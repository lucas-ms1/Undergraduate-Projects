import numpy as np

from steps_10_11_fiscal_module.policy.shocks import build_unit_impulse_shock
from steps_10_11_fiscal_module.simulation.irf import simulate_irf


def test_irf_tracks_required_variables_and_decays():
    A = np.diag([0.80, 0.70, 0.65, 0.90])
    B = np.eye(4)
    variable_index = {"y_hat": 0, "c_hat": 1, "i_hat": 2, "b_hat": 3}
    shock_index = {"eps_gc": 0, "eps_gi": 1, "eps_tau_l": 2, "eps_tau_k": 3}

    shock = build_unit_impulse_shock("gc", shock_index, impulse_size=0.01)
    out = simulate_irf(A, B, shock.vector, variable_index, horizon=40)

    assert set(out.series) == {"y_hat", "c_hat", "i_hat", "b_hat"}
    assert out.states.shape == (41, 4)
    assert out.series["y_hat"][1] > 0
    assert abs(out.series["y_hat"][-1]) < abs(out.series["y_hat"][1])


def test_tax_cut_shock_is_negative_innovation():
    shock_index = {"eps_gc": 0, "eps_gi": 1, "eps_tau_l": 2, "eps_tau_k": 3}
    tau_l = build_unit_impulse_shock("tau_l_cut", shock_index, impulse_size=0.02)
    tau_k = build_unit_impulse_shock("tau_k_cut", shock_index, impulse_size=0.02)

    assert np.isclose(tau_l.vector[shock_index["eps_tau_l"]], -0.02)
    assert np.isclose(tau_k.vector[shock_index["eps_tau_k"]], -0.02)
