"""Global configuration and baseline parameters for the dashboard."""

BASELINE_PARAMS = {
    "beta": 0.99,
    "sigma": 1.0,
    "alpha": 0.33,
    "delta": 0.025,
    "habit": 0.70,
    "psi": 0.50,
    "theta_p": 0.75,
    "theta_w": 0.75,
    "phi_b": 0.05,
    "lambda_rot": 0.30,
    "rho_i": 0.80,
    "phi_pi": 1.50,
    "phi_y": 0.125,
    "rho_x": 0.85,
    "sigma_x": 0.007,
    "gy_ratio": 0.20,
    "ig_ratio": 0.05,
    "debt_y_ratio": 0.60,
}

SLIDER_BOUNDS = {
    "habit": (0.00, 0.95),
    "psi": (0.01, 5.00),
    "theta_p": (0.10, 0.95),
    "theta_w": (0.10, 0.95),
    "phi_b": (0.00, 0.20),
    "lambda_rot": (0.00, 0.60),
    "rho_x": (0.00, 0.99),
    "sigma_x": (0.001, 0.05),
}

SIMULATION_HORIZON = 1_000
IRF_HORIZON = 40
