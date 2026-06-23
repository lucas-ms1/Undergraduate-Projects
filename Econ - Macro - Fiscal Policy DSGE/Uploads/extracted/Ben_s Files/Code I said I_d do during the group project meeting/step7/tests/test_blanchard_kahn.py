from config import BASELINE_PARAMS
from dsge.model import check_budget_residuals, solve_model_objects
from simulation.irf import compute_irf
from simulation.simulate import simulate_paths
from solvers.rational_expectations import solve_with_qz


def test_qz_returns_diagnostics():
    model = solve_model_objects(BASELINE_PARAMS)
    out = solve_with_qz(model["A_matrix"], model["B_matrix"])
    assert "eigenvalues" in out
    assert "flag" in out
    assert out["flag"] == "converged"
    assert out["bk_ok"] is True


def test_qz_bad_calibration_reports_failure_mode():
    bad = dict(BASELINE_PARAMS)
    bad["phi_b"] = 0.0
    model = solve_model_objects(bad)
    out = solve_with_qz(model["A_matrix"], model["B_matrix"], jump_count=1)
    assert out["flag"] in {"indeterminacy", "no_solution"}
    assert out["solver_success"] is False


def test_equation_coverage_check_passes_baseline():
    model = solve_model_objects(BASELINE_PARAMS)
    assert model["diagnostics"]["equation_coverage_ok"] is True


def test_budget_residual_check_is_small():
    model = solve_model_objects(BASELINE_PARAMS)
    paths, shocks = simulate_paths(
        model["A_matrix"],
        model["B_matrix"],
        horizon=200,
        seed=123,
        shock_std=BASELINE_PARAMS["sigma_x"],
        return_shocks=True,
    )
    check = check_budget_residuals(paths, shocks, model["A_matrix"], model["B_matrix"], model["variable_index"])
    assert check["ok"] is True
    assert check["max_abs_residual"] < 1e-8


def test_financing_rule_changes_debt_direction():
    model = solve_model_objects(BASELINE_PARAMS)
    idx = model["variable_index"]
    sidx = model["shock_index"]
    irf_tax = compute_irf(
        model["A_matrix"], model["B_matrix"], idx, sidx, "g", "Consumption Tax Hikes", horizon=12, shock_size=1.0
    )
    irf_transfer = compute_irf(
        model["A_matrix"], model["B_matrix"], idx, sidx, "g", "Lump-Sum transfers", horizon=12, shock_size=1.0
    )
    b_tax = irf_tax[:, idx["b_hat"]]
    b_transfer = irf_transfer[:, idx["b_hat"]]
    assert (abs(b_tax).max() > 0.0) and (abs(b_transfer).max() > 0.0)
    assert abs(b_tax[1]) > abs(b_transfer[1])
