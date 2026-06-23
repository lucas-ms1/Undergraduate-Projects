"""
dsge/steady_state.py
Step 4 – Deterministic steady-state solver
--------------------------------------------
Computes the deterministic steady state of the medium-scale Smets-Wouters
style DSGE model.  The solution is mostly analytical:

  1. Pin r^k from the Euler equation: r^k = 1/beta - 1 + delta
  2. Back out K/L from the firm FOC on capital
  3. Get Y/L from production, then w from the labor share
  4. Use fiscal ratios (G/Y, debt/Y, tax rates) to close the budget
  5. Verify three residuals at machine precision:
       - Resource constraint:   Y = C + I + G
       - Government budget:     tax_rev = G + r*B + T
       - Labor first-order condition

The returned dictionary contains all steady-state levels plus the
calibration parameters, so downstream modules have a single object.
"""

from __future__ import annotations

from dsge_engine.dsge.calibration import baseline_parameters


def solve_steady_state(params: dict | None = None) -> dict:
    """
    Compute deterministic steady-state levels from calibration targets.

    Parameters
    ----------
    params : calibration dict (uses baseline if None).  Slider overrides
             should already be merged in via calibration.override_parameters().

    Returns
    -------
    Dictionary with all steady-state levels, calibration values, and
    diagnostic residuals (resource_residual, gov_budget_residual,
    labor_foc_residual).
    """
    cal = baseline_parameters()
    if params:
        cal.update(params)

    # ── Unpack ────────────────────────────────────────────────────────────
    beta   = cal["beta"]
    sigma  = cal["sigma"]
    alpha  = cal["alpha"]
    delta  = cal["delta"]
    phi_l  = cal["phi_l"]
    g_y    = cal["g_y"]
    gi_y   = cal["gi_y"]
    debt_y = cal["debt_y"]
    tau_c  = cal["tau_c"]
    tau_l  = cal["tau_l"]
    tau_k  = cal["tau_k"]

    # ── Step 1: capital rental rate from Euler ────────────────────────────
    # In SS: 1 = beta * (1 + r^k - delta)  =>  r^k = 1/beta - 1 + delta
    rk = 1.0 / beta - 1.0 + delta

    # Real interest rate on government bonds
    r = 1.0 / beta - 1.0

    # ── Step 2: production-side ratios ────────────────────────────────────
    # From FOC on capital:  alpha * Y/K = r^k  =>  K/L = (alpha/r^k)^(1/(1-alpha))
    k_l = (alpha / rk) ** (1.0 / (1.0 - alpha))

    # Y/L from Cobb-Douglas:  Y = K^alpha * L^(1-alpha)
    y_l = k_l ** alpha

    # ── Step 3: pick labor, back out levels ───────────────────────────────
    # Normalise L = 0.33 (roughly one-third of time endowment)
    labor = 0.33

    capital    = k_l * labor
    output     = y_l * labor
    investment = delta * capital
    gov_c      = g_y * output          # government consumption
    gov_i      = gi_y * output         # government investment
    gov_total  = gov_c + gov_i         # total government spending
    consumption = output - investment - gov_total

    # Wage from labor share:  w = (1 - alpha) * Y / L
    wage = (1.0 - alpha) * output / labor

    # ── Step 4: labor FOC calibration ─────────────────────────────────────
    # In SS with separable utility:
    #   (1 - tau_l) * w / (1 + tau_c) = psi_labor * L^phi_l * C^sigma
    # Solve for psi_labor (the weight on labor disutility):
    psi_labor = ((1.0 - tau_l) * wage / (1.0 + tau_c)) / (
        consumption ** sigma * labor ** phi_l
    )

    # ── Step 5: fiscal block ──────────────────────────────────────────────
    debt = debt_y * output  # stock of one-period government debt

    # Tax revenue.  The assignment convention taxes the full capital rental
    # return, not the net-of-depreciation return.
    tax_revenue = (
        tau_c * consumption
        + tau_l * wage * labor
        + tau_k * rk * capital
    )

    # Government budget in SS:
    #   tax_rev = G_total + r * B + lump_sum
    # Solve for lump-sum transfers (T > 0 means transfer TO households).
    # A negative value is a net lump-sum tax.
    interest_on_debt = r * debt
    lump_sum = tax_revenue - gov_total - interest_on_debt

    # ── Diagnostic residuals (should all be ~ 0) ─────────────────────────
    resource_residual = output - consumption - investment - gov_total

    gov_budget_residual = (
        tax_revenue - gov_total - interest_on_debt - lump_sum
    )

    labor_foc_residual = (
        (1.0 - tau_l) * wage / (1.0 + tau_c)
        - psi_labor * consumption ** sigma * labor ** phi_l
    )

    # ── Steady-state ratios (useful for multipliers) ──────────────────────
    c_y = consumption / output
    i_y = investment / output
    k_y = capital / output

    # ── Return everything ─────────────────────────────────────────────────
    return {
        # Calibration parameters (pass-through)
        **cal,
        # Steady-state levels
        "Y":  output,
        "C":  consumption,
        "I":  investment,
        "K":  capital,
        "L":  labor,
        "w":  wage,
        "rk": rk,
        "r":  r,
        "R":  1.0 / beta,      # gross real rate
        "G_c": gov_c,
        "G_i": gov_i,
        "G":   gov_total,
        "B":  debt,
        "T":  lump_sum,        # lump-sum transfer (positive = to households)
        "tax_revenue": tax_revenue,
        "psi_labor":   psi_labor,
        # Ratios
        "c_y": c_y,
        "i_y": i_y,
        "k_y": k_y,
        "k_l": k_l,
        "y_l": y_l,
        # Diagnostics
        "resource_residual":    resource_residual,
        "gov_budget_residual":  gov_budget_residual,
        "labor_foc_residual":   labor_foc_residual,
    }
