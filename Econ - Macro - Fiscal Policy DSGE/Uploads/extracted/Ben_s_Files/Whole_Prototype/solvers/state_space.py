"""Build the STRUCTURAL linear state-space representation.

The system is written in expectational form:
    E_t[z_{t+1}] = A z_t + B eps_{t+1}

Three of the seven equations are forward-looking (Euler, Q-theory,
Phillips curve), so A has three eigenvalues outside the unit circle.
The Blanchard-Kahn solver decomposes this structural form into a
unique stable reduced-form for simulation.

Variable ordering:
  0  y_hat   Output              (predetermined)
  1  c_hat   Consumption         (JUMP - Euler equation)
  2  i_hat   Investment          (JUMP - Tobin-Q / adjustment costs)
  3  b_hat   Government debt     (predetermined - budget constraint)
  4  pi_hat  Inflation           (JUMP - NK Phillips curve)
  5  l_hat   Hours worked        (predetermined - labour market)
  6  r_hat   Nominal interest    (predetermined - Taylor rule)

Jump indices: [1, 2, 4] => 3 explosive roots required by BK.
"""

import numpy as np


def build_state_space(params):
    n = 7          # state variables
    m = 6          # shocks: tfp, gov, monetary, tau_l, tau_k, risk

    beta    = params["beta"]
    sigma   = params["sigma"]
    alpha   = params["alpha"]
    delta   = params["delta"]
    h       = params["habit"]
    psi     = params["psi"]
    theta_p = params["theta_p"]
    theta_w = params["theta_w"]
    phi_b   = params["phi_b"]
    phi_pi  = params["phi_pi"]
    phi_y   = params["phi_y"]
    rho_i   = params["rho_i"]
    lam     = params["lambda_rot"]
    sigma_x = params["sigma_x"]

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    # ---- PREDETERMINED equations (eigenvalues inside the unit circle) ----

    # Row 0 - Output (aggregate resource constraint / demand identity)
    A[0, 0] = 0.55 + 0.15 * h       # output persistence
    A[0, 1] = 0.20                   # consumption share
    A[0, 2] = 0.15                   # investment share
    A[0, 3] = -0.08                  # debt-crowding-out channel

    # Row 3 - Government debt (linearised budget constraint)
    A[3, 3] = 0.96 - 0.50 * phi_b   # fiscal-rule stabilisation
    A[3, 0] = -0.05                  # automatic stabilisers (tax revenue)

    # Row 5 - Hours (wage-setting / labour-market block)
    A[5, 5] = 0.40 + 0.30 * (1.0 - theta_w)   # wage flexibility
    A[5, 0] = 0.25                              # labour demand

    # Row 6 - Nominal rate (Taylor rule with interest-rate smoothing)
    A[6, 6] = rho_i
    A[6, 4] = (1.0 - rho_i) * phi_pi
    A[6, 0] = (1.0 - rho_i) * phi_y

    # ---- FORWARD-LOOKING / JUMP equations (eigenvalues > 1) ----

    # Row 1 - Consumption Euler with external habit formation
    A[1, 1] = 1.0 + h                            # > 1 for any h > 0
    A[1, 0] = 0.15 * (1.0 - h) * (1.0 + lam)    # income channel (ROT-amplified)
    A[1, 6] =  (1.0 - h) / sigma * 0.50          # real-rate intertemporal channel
    A[1, 4] = -(1.0 - h) / sigma * 0.50          # Fisher: inflation offsets nominal rate

    # Row 2 - Investment Euler / Tobin-Q dynamics
    A[2, 2] = 1.0 / beta + 0.10 * psi            # > 1
    A[2, 0] = 0.30 * alpha                        # expected marginal product of K
    A[2, 6] = -0.15                                # user cost of capital

    # Row 4 - New-Keynesian Phillips curve with partial indexation
    kappa_p = (1.0 - theta_p) * (1.0 - beta * theta_p) / max(theta_p, 1e-6)
    gamma_p = 0.40                                 # degree of backward indexation
    A[4, 4] = (1.0 + gamma_p * beta) / beta        # > 1
    mc_load = min(kappa_p, 0.50) / beta * 0.15     # cap kappa for stability
    A[4, 0] = -mc_load                              # marginal-cost channel

    # ---- SHOCK LOADINGS (B matrix) ----
    scale = sigma_x
    B[0, 0] =  1.00 * scale    # TFP -> output
    B[0, 1] =  0.80 * scale    # Gov demand -> output
    B[1, 0] =  0.30 * scale    # TFP -> consumption (income)
    B[1, 1] =  0.40 * scale    # Gov demand -> consumption
    B[2, 0] =  0.70 * scale    # TFP -> investment
    B[3, 1] =  1.20 * scale    # Gov -> debt
    B[3, 3] = -0.80 * scale    # Labour-tax revenue -> debt
    B[3, 4] = -0.80 * scale    # Capital-tax revenue -> debt
    B[4, 2] =  0.50 * scale    # Monetary -> inflation
    B[5, 0] =  0.40 * scale    # TFP -> hours
    B[5, 5] =  0.60 * scale    # Risk -> hours
    B[6, 2] =  1.00 * scale    # Monetary -> rate

    # Rule-of-thumb households amplify fiscal demand channels
    B[0, 1] *= (1.0 + 0.8 * lam)
    B[1, 1] *= (1.0 + 1.0 * lam)

    variable_index = {
        "y_hat": 0, "c_hat": 1, "i_hat": 2, "b_hat": 3,
        "pi_hat": 4, "l_hat": 5, "r_hat": 6,
    }
    shock_index = {
        "tfp": 0, "g": 1, "monetary": 2,
        "tau_l": 3, "tau_k": 4, "risk": 5,
    }

    C = np.eye(n)
    D = np.zeros((n, m))
    return A, B, C, D, variable_index, shock_index
