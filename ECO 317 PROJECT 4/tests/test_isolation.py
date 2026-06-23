import os
import sys
import traceback

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


FAILURES = []


def check(name, func):
    try:
        func()
        print(f"PASS  {name}")
    except Exception:
        print(f"FAIL  {name}")
        print(traceback.format_exc())
        FAILURES.append(name)


def test_sequential_run_without_state_pollution():
    from solow.model import simulate_transition, steady_state_k

    k1 = steady_state_k(0.3, 0.02, 0.02, 0.05, 0.33)
    sim1 = simulate_transition(0.3, 0.02, 0.02, 0.05, 0.33, T=50)

    from vfi.models.consumption_savings import solve as solve_m1

    result1 = solve_m1({"n_a": 50})

    from dsge_engine.dsge.calibration import override_parameters
    from dsge_engine.dsge.model import solve_model_objects

    params = override_parameters()
    model = solve_model_objects(params)
    assert model["diagnostics"]["solver_success"]

    k2 = steady_state_k(0.3, 0.02, 0.02, 0.05, 0.33)
    assert k1 == k2, "Solow steady_state_k changed after running VFI and DSGE"

    sim2 = simulate_transition(0.3, 0.02, 0.02, 0.05, 0.33, T=50)
    assert np.allclose(sim1["k"], sim2["k"]), "Solow simulation changed after running other modules"

    result2 = solve_m1({"n_a": 50})
    assert np.allclose(result1["value_function"], result2["value_function"]), (
        "VFI value function changed after running DSGE"
    )

    from dsge_engine.config import BASELINE_PARAMS as dsge_defaults
    from vfi.config import MODEL1_DEFAULTS as vfi_defaults

    assert "beta" in vfi_defaults and "beta" in dsge_defaults
    assert vfi_defaults["beta"] != dsge_defaults["beta"], (
        "VFI and DSGE share the same beta; verify this is intentional"
    )


def main():
    check("Sequential run without state pollution", test_sequential_run_without_state_pollution)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
