import os
import sys
import traceback

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vfi.config import MODEL1_DEFAULTS, MODEL2_DEFAULTS, MODEL3_DEFAULTS  # noqa: E402
from vfi.simulation.forecast import forecast_model  # noqa: E402
from vfi.simulation.moments import compute_moments  # noqa: E402
from vfi.simulation.simulate import simulate_model  # noqa: E402


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


def assert_monotone(arr, label):
    diffs = np.diff(np.asarray(arr, dtype=float))
    assert np.all(diffs >= -1e-10), f"{label} not monotone; min diff={diffs.min()}"


def test_model1_solve():
    from vfi.models.consumption_savings import solve

    params = {**MODEL1_DEFAULTS, "n_a": 100}
    result = solve(params)
    CTX["m1"] = (result, params)
    assert result["diagnostics"]["converged"], f"VFI did not converge: {result['diagnostics']}"
    n_a = params["n_a"]
    assert result["value_function"].shape == (n_a, 2)
    assert result["policy_indices"].shape == (n_a, 2)
    assert result["policy_levels"].shape == (n_a, 2)
    assert result["c_policy"].shape == (n_a, 2)
    assert len(result["grid"]) == n_a
    assert np.all(result["c_policy"] > 0), "Negative consumption detected"


def test_model1_policy_monotonicity():
    result, _ = CTX["m1"]
    for s in range(2):
        assert_monotone(result["policy_levels"][:, s], f"Model 1 policy state {s}")
        assert_monotone(result["c_policy"][:, s], f"Model 1 consumption state {s}")


def test_model1_simulation():
    result, _ = CTX["m1"]
    y_vals = np.array(MODEL1_DEFAULTS["y_vals"])
    P = np.array(MODEL1_DEFAULTS["P"])
    sim = simulate_model(result, y_vals, P, initial_state=5.0, model_name="model1", T_sim=200, seed=42)
    CTX["m1_sim"] = sim
    for key in ["a", "c", "y", "shock_idx"]:
        assert key in sim, f"Missing simulation key: {key}"
        assert len(sim[key]) == 200
        assert not np.any(np.isnan(sim[key])), f"NaN in simulation {key}"


def test_model1_moments():
    sim = CTX["m1_sim"]
    moments = compute_moments(sim, income_key="y")
    assert "c" in moments and "mean" in moments["c"]
    assert "a" in moments and "variance" in moments["a"]


def test_model1_forecast():
    result, _ = CTX["m1"]
    y_vals = np.array(MODEL1_DEFAULTS["y_vals"])
    shock_path = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    fcast = forecast_model(result, y_vals, shock_path, current_state=5.0, model_name="model1")
    for key in ["a", "c", "y"]:
        assert key in fcast
        assert len(fcast[key]) == 10
        assert not np.any(np.isnan(fcast[key]))


def test_model2_solve():
    from vfi.models.robinson_crusoe import solve

    params = {**MODEL2_DEFAULTS, "n_k": 100}
    result = solve(params)
    CTX["m2"] = (result, params)
    assert result["diagnostics"]["converged"], f"VFI did not converge: {result['diagnostics']}"
    for s in range(2):
        assert_monotone(result["policy_levels"][:, s], f"Model 2 policy state {s}")


def test_model2_simulation():
    result, params = CTX["m2"]
    z_vals = np.array(MODEL2_DEFAULTS["z_vals"])
    P = np.array(MODEL2_DEFAULTS["P"])
    sim = simulate_model(
        result, z_vals, P, initial_state=10.0, model_name="model2", T_sim=200, seed=42, model_params=params
    )
    CTX["m2_sim"] = sim
    for key in ["k", "c", "y", "investment"]:
        assert key in sim and len(sim[key]) == 200
        assert not np.any(np.isnan(sim[key]))
    moments = compute_moments(sim, income_key="y")
    assert "k" in moments
    shock_path = [0] * 5 + [1] * 5
    fcast = forecast_model(result, z_vals, shock_path, current_state=10.0, model_name="model2", model_params=params)
    assert not np.any(np.isnan(fcast["y"]))


def test_model3_assets_solve():
    from vfi.models.labor_supply import solve

    params = {**MODEL3_DEFAULTS, "n_a": 80, "n_L": 40, "include_assets": True}
    result = solve(params)
    CTX["m3_assets"] = (result, params)
    assert result["diagnostics"]["converged"], f"VFI did not converge: {result['diagnostics']}"
    assert "savings" in result["policy_levels"]
    assert "labor" in result["policy_levels"]


def test_model3_assets_simulation():
    result, _ = CTX["m3_assets"]
    w_vals = np.array(MODEL3_DEFAULTS["w_vals"])
    P = np.array(MODEL3_DEFAULTS["P"])
    sim = simulate_model(result, w_vals, P, initial_state=5.0, model_name="model3", T_sim=200, seed=42)
    for key in ["a", "c", "labor", "earnings"]:
        assert key in sim and len(sim[key]) == 200
        assert not np.any(np.isnan(sim[key]))
    assert np.all(sim["labor"] >= 0) and np.all(sim["labor"] <= 1)


def test_model3_labor_only_solve():
    from vfi.models.labor_supply import solve

    params = {**MODEL3_DEFAULTS, "n_L": 40, "include_assets": False}
    result = solve(params)
    CTX["m3_labor_only"] = (result, params)
    assert result["diagnostics"]["converged"], f"VFI did not converge: {result['diagnostics']}"


def test_model3_labor_only_simulation():
    result, _ = CTX["m3_labor_only"]
    w_vals = np.array(MODEL3_DEFAULTS["w_vals"])
    P = np.array(MODEL3_DEFAULTS["P"])
    sim = simulate_model(result, w_vals, P, initial_state=0.0, model_name="model3_labor_only", T_sim=200, seed=42)
    for key in ["c", "labor", "earnings"]:
        assert key in sim and len(sim[key]) == 200
        assert not np.any(np.isnan(sim[key]))


def main():
    check("Model 1 (Consumption-Savings) solve", test_model1_solve)
    check("Model 1 policy monotonicity", test_model1_policy_monotonicity)
    check("Model 1 simulation (200 periods)", test_model1_simulation)
    check("Model 1 moments", test_model1_moments)
    check("Model 1 forecast", test_model1_forecast)
    check("Model 2 (Robinson Crusoe) solve", test_model2_solve)
    check("Model 2 simulation", test_model2_simulation)
    check("Model 3 (Labor Supply, assets) solve", test_model3_assets_solve)
    check("Model 3 simulation", test_model3_assets_simulation)
    check("Model 3 (Labor-only) solve", test_model3_labor_only_solve)
    check("Model 3 (Labor-only) simulation", test_model3_labor_only_simulation)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
