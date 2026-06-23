import os
import sys
import traceback

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from solow.model import (  # noqa: E402
    golden_rule,
    simulate_capital_destruction,
    simulate_transition,
    solow_diagram_curves,
    steady_state_k,
)


FAILURES = []


def check(name, func):
    try:
        func()
        print(f"PASS  {name}")
    except Exception:
        print(f"FAIL  {name}")
        print(traceback.format_exc())
        FAILURES.append(name)


def test_steady_state_k():
    s, n, g, delta, alpha = 0.3, 0.02, 0.02, 0.05, 0.33
    k_star = steady_state_k(s, n, g, delta, alpha)
    expected = (0.3 / 0.09) ** (1 / 0.67)
    assert abs(k_star - expected) < 1e-10, f"k* = {k_star}, expected {expected}"

    s, n, g, delta, alpha = 0.2, 0.01, 0.01, 0.1, 0.5
    expected = (0.2 / 0.12) ** (1 / 0.5)
    assert abs(steady_state_k(s, n, g, delta, alpha) - expected) < 1e-10

    k_star = steady_state_k(0.3, -0.05, -0.05, 0.05, 0.33)
    assert k_star == float("inf") or k_star > 1e15


def test_golden_rule():
    for alpha in [0.25, 0.33, 0.5, 0.75]:
        for n, g, delta in [(0.01, 0.02, 0.05), (0.0, 0.0, 0.1)]:
            result = golden_rule(n, g, delta, alpha)
            assert abs(result["s_gold"] - alpha) < 1e-12
            breakeven = n + g + delta
            expected_k = (alpha / breakeven) ** (1 / (1 - alpha))
            assert abs(result["k_gold"] - expected_k) < 1e-10
            assert abs(result["c_gold"] - (result["y_gold"] - breakeven * result["k_gold"])) < 1e-10


def test_simulate_transition_convergence():
    s, n, g, delta, alpha = 0.3, 0.02, 0.02, 0.05, 0.33
    result = simulate_transition(s, n, g, delta, alpha, T=500)
    k_star = result["k_star"]
    assert abs(result["k"][-1] - k_star) / k_star < 0.01

    result2 = simulate_transition(s, n, g, delta, alpha, k0=2 * k_star, T=500)
    assert abs(result2["k"][-1] - k_star) / k_star < 0.01

    result3 = simulate_transition(s, n, g, delta, alpha, k0=k_star, T=100)
    assert np.all(abs(result3["k"] - k_star) / k_star < 0.001)

    for key in ["t", "k", "y", "c", "investment", "breakeven", "k_star", "y_star"]:
        assert key in result, f"Missing key: {key}"

    for key in ["k", "y", "c", "investment", "breakeven"]:
        assert not np.any(np.isnan(result[key])), f"NaN found in {key}"


def test_solow_diagram_curves():
    result = solow_diagram_curves(0.3, 0.02, 0.02, 0.05, 0.33)
    k_star = result["k_star"]
    sf_at_kstar = 0.3 * k_star ** 0.33
    breakeven_at_kstar = 0.09 * k_star
    assert abs(sf_at_kstar - breakeven_at_kstar) < 1e-8
    assert len(result["k_grid"]) == 500
    assert len(result["sf_k"]) == 500
    assert len(result["breakeven_k"]) == 500
    assert result["sf_k"][0] == 0.0
    assert result["breakeven_k"][0] == 0.0


def test_simulate_capital_destruction():
    result = simulate_capital_destruction(
        0.3, 0.02, 0.02, 0.05, 0.33, destruction_frac=0.5, T_before=50, T_after=200
    )
    k_star = result["k_star"]
    k_before = result["k"][:49]
    assert np.all(abs(k_before - k_star) / k_star < 0.01)
    k_at_destruction = result["k"][result["destruction_period"]]
    assert k_at_destruction < 0.85 * k_star, "Capital did not drop enough after destruction"
    k_terminal = result["k"][-1]
    assert abs(k_terminal - k_star) / k_star < 0.05, f"Did not recover: k_T={k_terminal}"
    growth_after = result["growth_rate"][result["destruction_period"] + 1 : result["destruction_period"] + 10]
    assert all(g > 0 for g in growth_after), "Growth should be positive during recovery"


def test_edge_cases():
    k = steady_state_k(0.0, 0.02, 0.02, 0.05, 0.33)
    assert k == 0.0 or abs(k) < 1e-15

    k = steady_state_k(1.0, 0.02, 0.02, 0.05, 0.33)
    expected = (1.0 / 0.09) ** (1 / 0.67)
    assert abs(k - expected) < 1e-8

    k = steady_state_k(0.3, 0.02, 0.02, 0.05, 0.99)
    assert np.isfinite(k) and k > 0

    k = steady_state_k(0.3, 0.5, 0.5, 0.5, 0.33)
    expected = (0.3 / 1.5) ** (1 / 0.67)
    assert abs(k - expected) < 1e-8


def main():
    check("steady_state_k", test_steady_state_k)
    check("golden_rule", test_golden_rule)
    check("simulate_transition convergence", test_simulate_transition_convergence)
    check("solow_diagram_curves", test_solow_diagram_curves)
    check("simulate_capital_destruction", test_simulate_capital_destruction)
    check("Edge cases", test_edge_cases)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
