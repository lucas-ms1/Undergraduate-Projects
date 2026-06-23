"""
Shared default parameters for all three models.

Notation follows the cross-model coding conventions defined in
model_lockin_sheet.tex (Section: Cross-model coding conventions).
"""

# ── Model 1: Stochastic Consumption-Savings ────────────────────────────────
MODEL1_DEFAULTS = dict(
    beta=0.95,
    sigma=2.0,
    r=0.03,
    y_vals=(0.5, 1.5),       # (y_L, y_H) – PLACEHOLDER, verify in course notes
    P=((0.9, 0.1),
       (0.1, 0.9)),
    a_min=0.0,
    a_max=20.0,
    n_a=300,
)

# ── Model 2: Stochastic Robinson Crusoe ────────────────────────────────────
MODEL2_DEFAULTS = dict(
    beta=0.95,
    sigma=2.0,
    alpha=0.36,
    delta=0.10,
    A=1.0,
    z_vals=(0.9, 1.1),       # (z_L, z_H) – PLACEHOLDER, verify in course notes
    P=((0.9, 0.1),
       (0.1, 0.9)),
    k_min=0.01,
    k_max=50.0,
    n_k=300,
)

# ── Model 3: Endogenous Labor Supply ──────────────────────────────────────
MODEL3_DEFAULTS = dict(
    beta=0.95,
    sigma=2.0,
    psi=1.0,                  # PLACEHOLDER – verify in course notes
    nu=2.0,                   # PLACEHOLDER – verify in course notes
    r=0.03,
    w_vals=(0.8, 1.2),       # (w_L, w_H) – PLACEHOLDER, verify in course notes
    P=((0.9, 0.1),
       (0.1, 0.9)),
    n_L=80,
    a_min=0.0,
    a_max=20.0,
    n_a=300,
    include_assets=True,      # Set False if course notes exclude assets
)

# ── Simulation / Forecasting ──────────────────────────────────────────────
SIM_DEFAULTS = dict(
    T_sim=200,
    T_fcast=10,
    seed=42,
)

# ── VFI solver ────────────────────────────────────────────────────────────
VFI_DEFAULTS = dict(
    max_iter=2000,
    tol=1e-6,
)
