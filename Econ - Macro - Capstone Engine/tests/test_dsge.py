import os
import sys
import traceback

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsge_engine.config import BASELINE_PARAMS  # noqa: E402
from dsge_engine.dsge.calibration import baseline_parameters, override_parameters  # noqa: E402
from dsge_engine.dsge.model import solve_model_objects  # noqa: E402
from dsge_engine.policy.multipliers import cumulative_multiplier, fiscal_drag_horizon, impact_multiplier  # noqa: E402
from dsge_engine.policy.shocks import build_unit_impulse_shock  # noqa: E402
from dsge_engine.simulation.irf import simulate_irf  # noqa: E402
from dsge_engine.simulation.moments import compute_all_moments, compute_moments  # noqa: E402
from dsge_engine.simulation.simulate import run_simulation, simulate  # noqa: E402
from dsge_engine.utils.summaries import generate_tab1_commentary, generate_tab2_briefing  # noqa: E402


FAILURES = []
CTX = {}


def check(name, func):
    try:
        func()
        print(f"PASS  {name}")
    except Exception:
        print(f"FAIL  {name}")
        print(traceback.format_exc())
        FAILURES.append(name)


def test_parameter_override():
    params = override_parameters({"habit": 0.5, "theta_p": 0.6})
    assert params["habit"] == 0.5
    assert params["theta_p"] == 0.6
    assert params["beta"] == BASELINE_PARAMS["beta"]
    p1 = baseline_parameters()
    p2 = baseline_parameters()
    p1["beta"] = 999
    assert p2["beta"] == 0.99


def test_model_solution():
    params = override_parameters()
    model = solve_model_objects(params)
    CTX["model"] = model
    diag = model["diagnostics"]
    assert diag["solver_success"], f"Solver failed: {diag.get('message', 'no message')}"
    assert diag.get("bk_ok", True), "Blanchard-Kahn conditions not satisfied"
    assert diag["equation_coverage_ok"], "Equation coverage check failed"
    A = model["A_matrix"]
    B = model["B_matrix"]
    CTX["A"] = A
    CTX["B"] = B
    assert A is not None and B is not None
    assert np.all(np.isfinite(A)), "A matrix contains non-finite values"
    assert np.all(np.isfinite(B)), "B matrix contains non-finite values"
    assert "y_hat" in model["variable_index"]
    assert "c_hat" in model["variable_index"]
    assert "b_hat" in model["variable_index"]


def test_eigenvalue_stability():
    A = CTX["A"]
    eigenvalues = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigenvalues))
    assert max_eig < 1.0 + 1e-8, f"Unstable eigenvalue: max |eig| = {max_eig}"


def test_unconditional_simulation():
    A = CTX["A"]
    B = CTX["B"]
    X = simulate(A, B, T=1000, seed=42)
    assert X.shape == (1000, A.shape[0])
    assert np.all(np.isfinite(X)), "Simulation contains non-finite values"
    assert np.max(np.abs(X)) < 100, f"Simulation appears to explode: max |x| = {np.max(np.abs(X))}"
    obs = run_simulation(A, B, T=1000, seed=42)
    CTX["obs"] = obs
    for key in ["y", "c", "i"]:
        assert key in obs, f"Missing observable: {key}"
        assert len(obs[key]) == 1000
        assert np.all(np.isfinite(obs[key]))


def test_moment_computation():
    obs = CTX["obs"]
    raw_moments = compute_moments(obs, output_key="y")
    CTX["raw_moments"] = raw_moments
    assert "var_y" in raw_moments
    assert "corr_c" in raw_moments
    assert "ac1_y" in raw_moments
    raw, hp = compute_all_moments(obs, output_key="y")
    assert "var_y" in hp
    assert "var_y" in raw


def test_fiscal_irf_lump_sum():
    model = CTX["model"]
    A = CTX["A"]
    B = CTX["B"]
    shock_index = model["shock_index"]
    variable_index = model["variable_index"]
    gc_shock = build_unit_impulse_shock("gc", shock_index, impulse_size=0.01)
    irf_result = simulate_irf(
        A,
        B,
        gc_shock.vector,
        variable_index,
        horizon=40,
        tracked_variables=["y_hat", "c_hat", "i_hat", "b_hat"],
        financing_rule="lump_sum",
        shock_index=shock_index,
        phi_b=0.05,
    )
    CTX["gc_shock"] = gc_shock
    CTX["irf_lump"] = irf_result
    assert irf_result.states.shape == (41, A.shape[0])
    assert "y_hat" in irf_result.series
    assert len(irf_result.series["y_hat"]) == 41
    assert np.all(np.isfinite(irf_result.states))
    assert np.allclose(irf_result.states[0], 0.0)
    assert not np.allclose(irf_result.states[1], 0.0)


def test_fiscal_irf_alternative_financing():
    model = CTX["model"]
    A = CTX["A"]
    B = CTX["B"]
    gc_shock = CTX["gc_shock"]
    shock_index = model["shock_index"]
    variable_index = model["variable_index"]
    for rule in ["consumption_tax", "labor_tax"]:
        irf2 = simulate_irf(
            A,
            B,
            gc_shock.vector,
            variable_index,
            horizon=40,
            tracked_variables=["y_hat", "b_hat"],
            financing_rule=rule,
            shock_index=shock_index,
            phi_b=0.05,
        )
        assert np.all(np.isfinite(irf2.states)), f"IRF with {rule} financing has non-finite values"


def test_multiplier_computation():
    model = CTX["model"]
    A = CTX["A"]
    B = CTX["B"]
    gc_shock = CTX["gc_shock"]
    shock_index = model["shock_index"]
    variable_index = model["variable_index"]
    g_var = "g_hat" if "g_hat" in variable_index else "g_c_hat"
    assert g_var in variable_index, "No government spending state found for multiplier computation"
    g_irf_result = simulate_irf(
        A,
        B,
        gc_shock.vector,
        variable_index,
        horizon=40,
        tracked_variables=["y_hat", g_var],
        financing_rule="lump_sum",
        shock_index=shock_index,
        phi_b=0.05,
    )
    y_irf = g_irf_result.series["y_hat"]
    g_irf = g_irf_result.series[g_var]
    im = impact_multiplier(y_irf, g_irf, y_ss=1.0, g_ss=1.0)
    cm = cumulative_multiplier(y_irf, g_irf, y_ss=1.0, g_ss=1.0, beta=0.99)
    drag = fiscal_drag_horizon(y_irf)
    CTX["multipliers"] = (im, cm, drag)
    assert np.isfinite(im), f"Impact multiplier is not finite: {im}"
    assert np.isfinite(cm), f"Cumulative multiplier is not finite: {cm}"
    print(f"  Impact multiplier: {im:.4f}")
    print(f"  Cumulative multiplier: {cm:.4f}")
    print(f"  Fiscal drag horizon: {drag}")


def test_summary_generators():
    raw_moments = CTX["raw_moments"]
    commentary = generate_tab1_commentary(
        moments=raw_moments,
        h=0.7,
        utilisation=0.5,
        price_stickiness=0.75,
        wage_stickiness=0.75,
        debt_feedback=0.05,
    )
    assert isinstance(commentary, str) and len(commentary) > 100
    briefing = generate_tab2_briefing(
        shock_name="Gc Shock",
        financing_rule="Lump-Sum",
        impact_mult=0.8,
        cumulative_mult=0.6,
        drag_horizon=12,
    )
    assert isinstance(briefing, str) and len(briefing) > 50


def main():
    check("Parameter override", test_parameter_override)
    check("Model solution (Blanchard-Kahn)", test_model_solution)
    check("Eigenvalue stability", test_eigenvalue_stability)
    check("1000-period simulation", test_unconditional_simulation)
    check("Moment computation", test_moment_computation)
    check("Fiscal IRF (Gc shock, lump-sum)", test_fiscal_irf_lump_sum)
    check("Fiscal IRF (alternative financing)", test_fiscal_irf_alternative_financing)
    check("Multiplier computation", test_multiplier_computation)
    check("Summary generators", test_summary_generators)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
