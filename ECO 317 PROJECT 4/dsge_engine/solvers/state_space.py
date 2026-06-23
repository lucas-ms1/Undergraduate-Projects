from __future__ import annotations

import numpy as np

from dsge_engine.policy.financing import build_financing_coefficients


VARIABLES = (
    "y_hat",
    "c_o_hat",
    "c_rot_hat",
    "c_hat",
    "i_hat",
    "q_hat",
    "k_hat",
    "w_hat",
    "rk_hat",
    "mc_hat",
    "pi_hat",
    "pi_w_hat",
    "l_hat",
    "b_hat",
    "i_nom_hat",
    "r_real_hat",
    "a_hat",
    "g_c_hat",
    "g_i_hat",
    "t_hat",
    "tau_c_hat",
    "tau_l_hat",
    "tau_k_hat",
    "m_policy_hat",
    "risk_hat",
)

SHOCKS = (
    "tfp",
    "g_c",
    "g_i",
    "monetary",
    "risk",
    "lump_sum",
    "tau_c",
    "tau_l",
    "tau_k",
)

JUMP_VARIABLES = ("c_o_hat", "i_hat", "q_hat", "pi_hat", "pi_w_hat")


def _safe(value: float, floor: float = 1e-8) -> float:
    return float(value) if abs(value) > floor else float(floor)


def _matrix() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(VARIABLES)}


def _shock_matrix() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(SHOCKS)}


def build_structural_system(
    params: dict,
    steady_state: dict | None = None,
    financing_rule: str = "lump_sum",
):
    ss = steady_state or {}
    vidx = _matrix()
    sidx = _shock_matrix()
    n = len(VARIABLES)
    m = len(SHOCKS)

    beta = params["beta"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    delta = params["delta"]
    habit = params["habit"]
    phi_l = params["phi_l"]
    theta_p = params["theta_p"]
    theta_w = params["theta_w"]
    eps_w = params.get("eps_w", 6.0)
    eps_p = params.get("eps_p", 6.0)
    lambda_rot = params["lambda_rot"]
    phi_pi = params["phi_pi"]
    phi_y = params["phi_y"]
    rho_i = params["rho_i"]
    rho_a = params["rho_a"]
    rho_gc = params["rho_g"]
    rho_gi = params["rho_gi"]
    rho_m = params["rho_m"]
    rho_risk = params["rho_rp"]
    phi_b = params["phi_b"]

    coeffs = build_financing_coefficients(phi_b, financing_rule)

    Y = _safe(ss.get("Y", 1.0))
    C = _safe(ss.get("C", 0.55))
    I = _safe(ss.get("I", 0.20))
    K = _safe(ss.get("K", 8.0))
    B = _safe(ss.get("B", 0.60))
    L = _safe(ss.get("L", 0.33))
    w = _safe(ss.get("w", 1.0))
    rk = _safe(ss.get("rk", 0.05))
    G_c = _safe(ss.get("G_c", params["g_y"]))
    G_i = _safe(ss.get("G_i", params["gi_y"]))
    T = ss.get("T", 0.0)
    transfer_scale = _safe(abs(T) if abs(T) > 1e-8 else 1e-8)
    r = ss.get("r", 1.0 / beta - 1.0)
    tau_c = params["tau_c"]
    tau_l = params["tau_l"]
    tau_k = params["tau_k"]

    c_y = ss.get("c_y", C / Y)
    i_y = ss.get("i_y", I / Y)
    gc_y = G_c / Y
    gi_y = G_i / Y
    k_y = ss.get("k_y", K / Y)

    kappa_p = ((1.0 - theta_p) * (1.0 - beta * theta_p) / _safe(theta_p))
    kappa_w = (
        (1.0 - theta_w)
        * (1.0 - beta * theta_w)
        / _safe(theta_w * (1.0 + phi_l * eps_w))
    )
    labor_share = 1.0 - alpha
    tax_k_wedge = tau_k / _safe(1.0 - tau_k)
    invest_adj = 1.0 + params["psi_util"]
    mrs_coeff = sigma + phi_l

    Gamma_f = np.zeros((n, n))
    Gamma_0 = np.zeros((n, n))
    Gamma_l = np.zeros((n, n))
    Psi = np.zeros((n, m))
    labels: list[str] = []

    def row(label: str) -> int:
        labels.append(label)
        return len(labels) - 1

    r_ = row("euler_optimizer")
    Gamma_f[r_, vidx["c_o_hat"]] = -1.0 / _safe(1.0 + habit)
    # No direct forward pi_hat term: expected inflation already enters
    # through r_real_hat via the Fisher relation.
    Gamma_0[r_, vidx["c_o_hat"]] = 1.0
    Gamma_0[r_, vidx["r_real_hat"]] = (1.0 - habit) / _safe(sigma * (1.0 + habit))
    Gamma_l[r_, vidx["c_o_hat"]] = -habit / _safe(1.0 + habit)
    Gamma_0[r_, vidx["risk_hat"]] = (1.0 - habit) / _safe(sigma * (1.0 + habit))

    r_ = row("rot_budget")
    Gamma_0[r_, vidx["c_rot_hat"]] = 1.0
    Gamma_0[r_, vidx["w_hat"]] = -(1.0 - tau_l) * w * L / _safe(C)
    Gamma_0[r_, vidx["l_hat"]] = -(1.0 - tau_l) * w * L / _safe(C)
    Gamma_0[r_, vidx["t_hat"]] = -transfer_scale / _safe(C)
    Gamma_0[r_, vidx["tau_l_hat"]] = tau_l * w * L / _safe(C)
    Gamma_0[r_, vidx["tau_c_hat"]] = tau_c * C / _safe((1.0 + tau_c) * C)

    r_ = row("aggregate_consumption")
    Gamma_0[r_, vidx["c_hat"]] = 1.0
    Gamma_0[r_, vidx["c_o_hat"]] = -(1.0 - lambda_rot)
    Gamma_0[r_, vidx["c_rot_hat"]] = -lambda_rot

    r_ = row("fisher_relation")
    Gamma_f[r_, vidx["pi_hat"]] = -1.0
    Gamma_0[r_, vidx["r_real_hat"]] = 1.0
    Gamma_0[r_, vidx["i_nom_hat"]] = -1.0

    r_ = row("nkpc")
    Gamma_f[r_, vidx["pi_hat"]] = -beta
    Gamma_0[r_, vidx["pi_hat"]] = 1.0
    Gamma_0[r_, vidx["mc_hat"]] = -kappa_p

    r_ = row("wage_phillips")
    Gamma_f[r_, vidx["pi_w_hat"]] = -beta
    Gamma_0[r_, vidx["pi_w_hat"]] = 1.0
    Gamma_0[r_, vidx["c_hat"]] = -kappa_w * sigma
    Gamma_0[r_, vidx["l_hat"]] = -kappa_w * phi_l
    Gamma_0[r_, vidx["w_hat"]] = kappa_w

    r_ = row("real_wage_lom")
    Gamma_0[r_, vidx["w_hat"]] = 1.0
    Gamma_l[r_, vidx["w_hat"]] = -1.0
    Gamma_0[r_, vidx["pi_w_hat"]] = -1.0
    Gamma_0[r_, vidx["pi_hat"]] = 1.0

    r_ = row("marginal_cost_identity")
    Gamma_0[r_, vidx["mc_hat"]] = 1.0
    Gamma_0[r_, vidx["w_hat"]] = -labor_share
    Gamma_0[r_, vidx["rk_hat"]] = -alpha
    Gamma_0[r_, vidx["a_hat"]] = 1.0

    r_ = row("production")
    Gamma_0[r_, vidx["y_hat"]] = 1.0
    Gamma_l[r_, vidx["k_hat"]] = -alpha
    Gamma_0[r_, vidx["l_hat"]] = -labor_share
    Gamma_0[r_, vidx["a_hat"]] = -1.0

    r_ = row("capital_rental_foc")
    Gamma_0[r_, vidx["rk_hat"]] = 1.0
    Gamma_0[r_, vidx["y_hat"]] = -1.0
    Gamma_l[r_, vidx["k_hat"]] = 1.0

    r_ = row("capital_accumulation")
    Gamma_0[r_, vidx["k_hat"]] = 1.0
    Gamma_l[r_, vidx["k_hat"]] = -(1.0 - delta)
    Gamma_0[r_, vidx["i_hat"]] = -delta
    Gamma_0[r_, vidx["g_i_hat"]] = -(G_i / _safe(K))

    r_ = row("q_valuation")
    Gamma_0[r_, vidx["q_hat"]] = 1.0
    Gamma_f[r_, vidx["q_hat"]] = -beta * (1.0 - delta)
    Gamma_f[r_, vidx["rk_hat"]] = -beta * (1.0 - tau_k)
    Gamma_f[r_, vidx["tau_k_hat"]] = beta * tax_k_wedge
    Gamma_0[r_, vidx["i_nom_hat"]] = beta
    Gamma_f[r_, vidx["pi_hat"]] = -beta
    Gamma_f[r_, vidx["risk_hat"]] = beta

    r_ = row("investment_adjustment")
    Gamma_0[r_, vidx["i_hat"]] = 1.0
    Gamma_0[r_, vidx["q_hat"]] = -invest_adj
    Gamma_l[r_, vidx["i_hat"]] = -(1.0 / _safe(invest_adj))

    r_ = row("resource_constraint")
    Gamma_0[r_, vidx["y_hat"]] = 1.0
    Gamma_0[r_, vidx["c_hat"]] = -c_y
    Gamma_0[r_, vidx["i_hat"]] = -i_y
    Gamma_0[r_, vidx["g_c_hat"]] = -gc_y
    Gamma_0[r_, vidx["g_i_hat"]] = -gi_y

    r_ = row("government_budget")
    Gamma_0[r_, vidx["b_hat"]] = 1.0
    Gamma_l[r_, vidx["b_hat"]] = -(1.0 + r)
    Gamma_0[r_, vidx["g_c_hat"]] = -(G_c / B)
    Gamma_0[r_, vidx["g_i_hat"]] = -(G_i / B)
    Gamma_0[r_, vidx["t_hat"]] = -(transfer_scale / B)
    Gamma_0[r_, vidx["tau_c_hat"]] = tau_c * C / B
    Gamma_0[r_, vidx["tau_l_hat"]] = tau_l * w * L / B
    Gamma_0[r_, vidx["tau_k_hat"]] = tau_k * rk * K / B
    Gamma_0[r_, vidx["c_hat"]] = tau_c * C / B
    Gamma_0[r_, vidx["w_hat"]] = tau_l * w * L / B
    Gamma_0[r_, vidx["l_hat"]] = tau_l * w * L / B
    Gamma_0[r_, vidx["rk_hat"]] = tau_k * rk * K / B
    Gamma_0[r_, vidx["k_hat"]] = tau_k * rk * K / B
    Gamma_0[r_, vidx["r_real_hat"]] = -r

    r_ = row("taylor_rule")
    Gamma_0[r_, vidx["i_nom_hat"]] = 1.0
    Gamma_l[r_, vidx["i_nom_hat"]] = -rho_i
    Gamma_0[r_, vidx["pi_hat"]] = -(1.0 - rho_i) * phi_pi
    Gamma_0[r_, vidx["y_hat"]] = -(1.0 - rho_i) * phi_y
    Gamma_0[r_, vidx["m_policy_hat"]] = -(1.0 - rho_i)

    r_ = row("tfp_process")
    Gamma_0[r_, vidx["a_hat"]] = 1.0
    Gamma_l[r_, vidx["a_hat"]] = -rho_a
    Psi[r_, sidx["tfp"]] = -1.0

    r_ = row("gc_process")
    Gamma_0[r_, vidx["g_c_hat"]] = 1.0
    Gamma_l[r_, vidx["g_c_hat"]] = -rho_gc
    Gamma_0[r_, vidx["b_hat"]] = -coeffs["phi_b_gc"]
    Psi[r_, sidx["g_c"]] = -1.0

    r_ = row("gi_process")
    Gamma_0[r_, vidx["g_i_hat"]] = 1.0
    Gamma_l[r_, vidx["g_i_hat"]] = -rho_gi
    Psi[r_, sidx["g_i"]] = -1.0

    r_ = row("lump_sum_rule")
    Gamma_0[r_, vidx["t_hat"]] = 1.0
    Gamma_0[r_, vidx["b_hat"]] = -coeffs["phi_b_lump_sum"]
    Psi[r_, sidx["lump_sum"]] = -1.0

    r_ = row("consumption_tax_rule")
    Gamma_0[r_, vidx["tau_c_hat"]] = 1.0
    Gamma_0[r_, vidx["b_hat"]] = -coeffs["phi_b_tau_c"]
    Psi[r_, sidx["tau_c"]] = -1.0

    r_ = row("labor_tax_rule")
    Gamma_0[r_, vidx["tau_l_hat"]] = 1.0
    Gamma_0[r_, vidx["b_hat"]] = -coeffs["phi_b_tau_l"]
    Psi[r_, sidx["tau_l"]] = -1.0

    r_ = row("capital_tax_rule")
    Gamma_0[r_, vidx["tau_k_hat"]] = 1.0
    Gamma_0[r_, vidx["b_hat"]] = -coeffs["phi_b_tau_k"]
    Psi[r_, sidx["tau_k"]] = -1.0

    r_ = row("monetary_process")
    Gamma_0[r_, vidx["m_policy_hat"]] = 1.0
    Gamma_l[r_, vidx["m_policy_hat"]] = -rho_m
    Psi[r_, sidx["monetary"]] = -1.0

    r_ = row("risk_process")
    Gamma_0[r_, vidx["risk_hat"]] = 1.0
    Gamma_l[r_, vidx["risk_hat"]] = -rho_risk
    Psi[r_, sidx["risk"]] = -1.0

    return {
        "Gamma_f": Gamma_f,
        "Gamma_0": Gamma_0,
        "Gamma_l": Gamma_l,
        "Psi": Psi,
        "equation_labels": labels,
        "jump_variables": list(JUMP_VARIABLES),
        "variable_index": vidx,
        "shock_index": sidx,
        "parameters": dict(params),
        "steady_state": dict(ss),
    }

def reduced_form_from_structural(system: dict):
    from dsge_engine.solvers.rational_expectations import solve_with_qz

    solution = solve_with_qz(system)
    if not solution.get("solver_success", False):
        raise RuntimeError(solution.get("message", "Rational-expectations solve failed."))
    return solution["A_matrix"], solution["B_matrix"]


def build_state_space(
    params: dict,
    steady_state: dict | None = None,
    financing_rule: str = "lump_sum",
):
    system = build_structural_system(
        params=params,
        steady_state=steady_state,
        financing_rule=financing_rule,
    )
    A, B = reduced_form_from_structural(system)
    C = np.eye(len(system["variable_index"]))
    D = np.zeros((len(system["variable_index"]), len(system["shock_index"])))
    return A, B, C, D, system["variable_index"], system["shock_index"]
