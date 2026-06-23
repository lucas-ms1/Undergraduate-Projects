"""
tests/test_steady_state.py
Step 4 - Steady-state verification
-------------------------------------
Checks that the deterministic steady state satisfies:
  1. Resource constraint:  Y = C + I + G
  2. Government budget:    tax_rev = G + r*B + T
  3. Labor first-order condition
  4. All levels are strictly positive
  5. Ratios are economically sensible
"""

import pytest
from dsge.calibration import baseline_parameters
from dsge.steady_state import solve_steady_state

TOL = 1e-10


@pytest.fixture
def ss():
    """Baseline steady state."""
    return solve_steady_state(baseline_parameters())


@pytest.fixture
def ss_custom():
    """Steady state with non-default overrides."""
    params = baseline_parameters()
    params["habit"] = 0.40
    params["theta_p"] = 0.60
    params["phi_b"] = 0.10
    params["lambda_rot"] = 0.20
    return solve_steady_state(params)


class TestEquilibriumResiduals:
    """All three equilibrium residuals must be zero at machine precision."""

    def test_resource_constraint(self, ss):
        assert abs(ss["resource_residual"]) < TOL, (
            f"Resource constraint violated: residual = {ss['resource_residual']}"
        )

    def test_gov_budget_constraint(self, ss):
        assert abs(ss["gov_budget_residual"]) < TOL, (
            f"Government budget violated: residual = {ss['gov_budget_residual']}"
        )

    def test_labor_foc(self, ss):
        assert abs(ss["labor_foc_residual"]) < TOL, (
            f"Labor FOC violated: residual = {ss['labor_foc_residual']}"
        )

    def test_residuals_custom_params(self, ss_custom):
        """Same checks hold under non-default calibration."""
        assert abs(ss_custom["resource_residual"]) < TOL
        assert abs(ss_custom["gov_budget_residual"]) < TOL
        assert abs(ss_custom["labor_foc_residual"]) < TOL


class TestPositiveLevels:
    """All real quantities and prices must be strictly positive."""

    @pytest.mark.parametrize("key", ["Y", "C", "I", "K", "L", "w", "rk", "G", "B"])
    def test_positive(self, ss, key):
        assert ss[key] > 0, f"Steady-state {key} = {ss[key]} is not positive"


class TestEconomicSanity:
    """Basic economic plausibility of the steady state."""

    def test_consumption_share(self, ss):
        assert 0.3 < ss["c_y"] < 0.85

    def test_investment_share(self, ss):
        assert 0.05 < ss["i_y"] < 0.40

    def test_capital_output_ratio(self, ss):
        assert 2.0 < ss["k_y"] < 20.0

    def test_rental_rate(self, ss):
        assert 0.0 < ss["rk"] < 0.10

    def test_real_rate(self, ss):
        expected = 1.0 / ss["beta"] - 1.0
        assert abs(ss["r"] - expected) < TOL

    def test_output_decomposition(self, ss):
        computed = ss["C"] + ss["I"] + ss["G"]
        assert abs(ss["Y"] - computed) < TOL

    def test_assignment_fiscal_convention(self, ss):
        expected_tax_revenue = (
            ss["tau_c"] * ss["C"]
            + ss["tau_l"] * ss["w"] * ss["L"]
            + ss["tau_k"] * ss["rk"] * ss["K"]
        )
        expected_transfer = expected_tax_revenue - ss["G"] - ss["r"] * ss["B"]
        assert abs(ss["tax_revenue"] - expected_tax_revenue) < TOL
        assert abs(ss["T"] - expected_transfer) < TOL


class TestDefaultValues:
    """Pin down expected baseline values so refactors don't silently change them."""

    def test_beta(self, ss):
        assert ss["beta"] == 0.99

    def test_alpha(self, ss):
        assert ss["alpha"] == 0.33

    def test_delta(self, ss):
        assert ss["delta"] == 0.025

    def test_labor(self, ss):
        assert abs(ss["L"] - 0.33) < TOL
