"""
config.py
Global configuration and baseline parameters for the DSGE fiscal dashboard.

All default calibration values, slider ranges, and simulation settings live
here.  app.py reads from this file; dsge/calibration.py wraps it for the
model-solve pipeline.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Baseline structural parameters (Smets-Wouters 2007 style)
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_PARAMS = {
    # Preferences / technology
    "beta":    0.99,       # discount factor
    "sigma":   1.0,        # CRRA (1 = log utility)
    "alpha":   0.33,       # capital share in production
    "delta":   0.025,      # quarterly depreciation rate
    "phi_l":   1.0,        # inverse Frisch elasticity of labor supply

    # Frictions
    "habit":       0.70,   # internal habit persistence h
    "psi_util":    0.50,   # capital-utilization cost convexity
    "theta_p":     0.75,   # Calvo price stickiness
    "theta_w":     0.75,   # Calvo wage stickiness

    # Markup elasticities
    "eps_p":   6.0,        # price elasticity of substitution
    "eps_w":   6.0,        # wage elasticity of substitution

    # Fiscal block
    "phi_b":       0.05,   # debt-feedback intensity (magnitude; sign set per rule)
    "lambda_rot":  0.30,   # rule-of-thumb (non-Ricardian) household share
    "g_y":         0.20,   # government spending / output ratio
    "gi_y":        0.05,   # government investment / output ratio
    "debt_y":      0.60,   # public debt / annual output (quarterly: debt_y * 4)
    "tau_c":       0.08,   # average consumption tax rate
    "tau_l":       0.20,   # average labor tax rate
    "tau_k":       0.20,   # average capital tax rate

    # Monetary policy (Taylor rule)
    "rho_i":   0.80,       # interest-rate smoothing
    "phi_pi":  1.50,       # Taylor-rule inflation coefficient
    "phi_y":   0.125,      # Taylor-rule output-gap coefficient

    # Shock AR(1) persistences
    "rho_a":   0.90,       # TFP
    "rho_g":   0.85,       # government consumption
    "rho_gi":  0.85,       # government investment
    "rho_m":   0.70,       # monetary policy
    "rho_rp":  0.65,       # risk premium
    "rho_p":   0.80,       # price markup
    "rho_w":   0.80,       # wage markup

    # Shock standard deviations
    "sigma_a":   0.010,
    "sigma_g":   0.010,
    "sigma_gi":  0.010,
    "sigma_m":   0.005,
    "sigma_rp":  0.005,
    "sigma_p":   0.005,
    "sigma_w":   0.005,
}

# ──────────────────────────────────────────────────────────────────────────────
# Slider bounds (from the plan §4.13)
# ──────────────────────────────────────────────────────────────────────────────
SLIDER_BOUNDS = {
    "habit":       (0.00,  0.95,  0.01),   # (min, max, step)
    "psi_util":    (0.01,  5.00,  0.01),
    "theta_p":     (0.10,  0.95,  0.01),
    "theta_w":     (0.10,  0.95,  0.01),
    "phi_b":       (0.00,  0.20,  0.005),
    "lambda_rot":  (0.00,  0.60,  0.01),
    "rho_a":       (0.00,  0.99,  0.01),
    "sigma_a":     (0.001, 0.05,  0.001),
}

# ──────────────────────────────────────────────────────────────────────────────
# Simulation / IRF settings
# ──────────────────────────────────────────────────────────────────────────────
SIMULATION_HORIZON = 1_000   # periods for Tab 1 unconditional simulation
IRF_HORIZON        = 40      # quarters for Tab 2 fiscal IRFs
