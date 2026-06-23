import numpy as np

from dsge.calibration import override_parameters
from dsge.model import solve_model_objects
from policy.financing import FINANCING_RULES
from policy.shocks import build_unit_impulse_shock
from simulation.irf import simulate_irf


def test_gc_and_gi_map_to_distinct_shock_indices():
    model = solve_model_objects(override_parameters())
    sidx = model["shock_index"]
    gc = build_unit_impulse_shock("gc", sidx, impulse_size=0.01)
    gi = build_unit_impulse_shock("gi", sidx, impulse_size=0.01)
    assert sidx["g_c"] != sidx["g_i"]
    assert gc.vector[sidx["g_c"]] == 0.01
    assert gc.vector[sidx["g_i"]] == 0.0
    assert gi.vector[sidx["g_i"]] == 0.01
    assert gi.vector[sidx["g_c"]] == 0.0


def test_gc_and_gi_generate_different_irfs():
    model = solve_model_objects(override_parameters())
    A = model["A_matrix"]
    B = model["B_matrix"]
    idx = model["variable_index"]
    sidx = model["shock_index"]
    tracked = ("y_hat", "g_c_hat", "g_i_hat")
    gc = simulate_irf(A, B, build_unit_impulse_shock("gc", sidx).vector, idx, 12, tracked)
    gi = simulate_irf(A, B, build_unit_impulse_shock("gi", sidx).vector, idx, 12, tracked)
    assert not np.allclose(gc.series["y_hat"], gi.series["y_hat"])
    assert gc.series["g_c_hat"][1] != 0.0
    assert gi.series["g_i_hat"][1] != 0.0


def test_tax_cut_signs_remain_negative():
    model = solve_model_objects(override_parameters())
    sidx = model["shock_index"]
    tau_l = build_unit_impulse_shock("tau_l_cut", sidx, impulse_size=0.02)
    tau_k = build_unit_impulse_shock("tau_k_cut", sidx, impulse_size=0.02)
    assert np.isclose(tau_l.vector[sidx["tau_l"]], -0.02)
    assert np.isclose(tau_k.vector[sidx["tau_k"]], -0.02)


def test_gc_output_irfs_differ_across_financing_rules():
    params = override_parameters()
    output_irfs = []
    for rule in FINANCING_RULES:
        model = solve_model_objects(params, financing_rule=rule)
        if not model["diagnostics"]["solver_success"]:
            continue
        shock = build_unit_impulse_shock("gc", model["shock_index"])
        irf = simulate_irf(
            model["A_matrix"],
            model["B_matrix"],
            shock.vector,
            model["variable_index"],
            20,
            ("y_hat",),
        )
        output_irfs.append(irf.series["y_hat"])

    assert len(output_irfs) >= 2
    distinct_pairs = 0
    for i in range(len(output_irfs)):
        for j in range(i + 1, len(output_irfs)):
            if not np.allclose(output_irfs[i], output_irfs[j]):
                distinct_pairs += 1
    assert distinct_pairs >= 1


def test_fiscal_shocks_move_debt_with_assignment_signs():
    model = solve_model_objects(override_parameters())
    A = model["A_matrix"]
    B = model["B_matrix"]
    idx = model["variable_index"]
    sidx = model["shock_index"]

    for shock_name in ("gc", "gi"):
        shock = build_unit_impulse_shock(shock_name, sidx)
        irf = simulate_irf(A, B, shock.vector, idx, 1, ("b_hat",))
        assert irf.series["b_hat"][1] > 0

    for shock_name in ("tau_c", "tau_l", "tau_k"):
        shock = np.zeros(len(sidx))
        shock[sidx[shock_name]] = 1.0
        irf = simulate_irf(A, B, shock, idx, 1, ("b_hat",))
        assert irf.series["b_hat"][1] < 0
