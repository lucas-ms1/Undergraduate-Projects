"""Step 4 code snapshot: calibration, steady state, and tests.

This file mirrors the current code from:
- dsge/calibration.py
- dsge/steady_state.py
- tests/test_steady_state.py
"""


# =========================
# File: dsge/calibration.py
# =========================
def baseline_parameters():
    """Return baseline calibration dictionary for the DSGE skeleton."""
    return {
        # Preferences / technology
        "beta": 0.99,
        "sigma": 1.0,  # log utility in consumption
        "alpha": 0.33,
        "delta": 0.025,
        "phi_l": 1.0,  # inverse Frisch elasticity in labor disutility
        "labor_ss": 0.33,
        # Markup elasticities (price, wage)
        "eps_p": 6.0,
        "eps_w": 6.0,
        # Fiscal targets
        "g_y": 0.20,  # government spending to output
        "debt_y": 0.60,  # public debt to output
        # Average tax rates
        "tau_c": 0.08,
        "tau_l": 0.20,
        "tau_k": 0.20,
        # AR(1) shock persistence
        "rho_a": 0.90,
        "rho_g": 0.85,
        "rho_m": 0.70,
        "rho_rp": 0.65,
        # Shock standard deviations
        "sigma_a": 0.01,
        "sigma_g": 0.01,
        "sigma_m": 0.005,
        "sigma_rp": 0.005,
    }


# ==========================
# File: dsge/steady_state.py
# ==========================
def solve_steady_state(params=None):
    """Compute deterministic steady-state levels from calibration targets."""
    calibrated = baseline_parameters()
    if params:
        calibrated.update(params)

    beta = calibrated["beta"]
    alpha = calibrated["alpha"]
    delta = calibrated["delta"]
    sigma = calibrated["sigma"]
    phi_l = calibrated["phi_l"]
    labor = calibrated["labor_ss"]
    g_y = calibrated["g_y"]
    debt_y = calibrated["debt_y"]

    # Capital rental rate from Euler equation with log utility benchmark.
    rk = 1.0 / beta - 1.0 + delta

    # Production-side ratios.
    k_l = (alpha / rk) ** (1.0 / (1.0 - alpha))
    y_l = k_l ** alpha

    capital = k_l * labor
    output = y_l * labor
    investment = delta * capital
    gov_spending = g_y * output
    consumption = output - investment - gov_spending
    wage = (1.0 - alpha) * output / labor

    # Labor FOC: w = psi * C^sigma * L^phi_l
    psi = wage / (consumption**sigma * labor**phi_l)

    debt = debt_y * output
    interest_on_debt = (1.0 / beta - 1.0) * debt

    tax_revenue = (
        calibrated["tau_c"] * consumption
        + calibrated["tau_l"] * wage * labor
        + calibrated["tau_k"] * (rk - delta) * capital
    )
    lump_sum = gov_spending + interest_on_debt - tax_revenue

    resource_residual = output - consumption - investment - gov_spending
    gov_budget_residual = tax_revenue + lump_sum - gov_spending - interest_on_debt
    labor_foc_residual = wage - psi * (consumption**sigma) * (labor**phi_l)

    return {
        **calibrated,
        "R": 1.0 / beta,
        "rk": rk,
        "k_l": k_l,
        "y_l": y_l,
        "K": capital,
        "L": labor,
        "Y": output,
        "I": investment,
        "C": consumption,
        "G": gov_spending,
        "w": wage,
        "psi": psi,
        "B": debt,
        "tax_revenue": tax_revenue,
        "lump_sum": lump_sum,
        "resource_residual": resource_residual,
        "gov_budget_residual": gov_budget_residual,
        "labor_foc_residual": labor_foc_residual,
    }


# ==============================
# File: tests/test_steady_state.py
# ==============================
def test_steady_state_conditions_hold():
    ss = solve_steady_state(baseline_parameters())

    tol = 1e-10
    assert abs(ss["resource_residual"]) < tol
    assert abs(ss["gov_budget_residual"]) < tol
    assert abs(ss["labor_foc_residual"]) < tol


def test_steady_state_levels_are_positive():
    ss = solve_steady_state()
    for key in ("Y", "C", "I", "G", "K", "L", "w", "rk"):
        assert ss[key] > 0.0
