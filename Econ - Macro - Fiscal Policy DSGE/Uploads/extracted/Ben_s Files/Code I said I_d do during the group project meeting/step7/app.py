import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import BASELINE_PARAMS, IRF_HORIZON, SIMULATION_HORIZON, SLIDER_BOUNDS
from dsge.model import check_budget_residuals, solve_model_objects
from policy.multipliers import compute_multipliers
from simulation.empirical import compute_empirical_moments, fetch_fred_levels
from policy.financing import financing_sign
from policy.shocks import shock_vector_name
from simulation.irf import compute_irf
from simulation.moments import compute_moments
from simulation.simulate import simulate_paths
from solvers.rational_expectations import likely_failure_hint, solve_with_qz
from utils.summaries import tab1_summary, tab2_summary


def load_css():
    css_path = pathlib.Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data
def cached_solve(params_tuple):
    params = dict(params_tuple)
    model = solve_model_objects(params)
    if model["diagnostics"]["solver_success"]:
        qz = solve_with_qz(model["A_matrix"], model["B_matrix"])
        model["diagnostics"].update(qz)
    return model


@st.cache_data
def cached_simulate(A, B, horizon, seed, shock_std):
    return simulate_paths(A, B, horizon=horizon, seed=seed, shock_std=shock_std, return_shocks=True)


def financing_rule_moves_with_debt(irf, idx, rule, tol=1e-10):
    debt_path = irf[:, idx["b_hat"]]
    moved = bool(np.max(np.abs(debt_path)) > tol)
    if not moved:
        return False, "Debt does not move under this calibration, so financing cannot respond."
    signed = financing_sign(rule) * debt_path
    expected = bool(np.nanmax(signed) > tol)
    if not expected:
        return False, "Financing response sign looks flipped relative to debt. Try another rule or calibration."
    return True, ""


@st.cache_data(ttl=24 * 60 * 60)
def cached_empirical():
    levels = fetch_fred_levels()
    return compute_empirical_moments(levels)


st.set_page_config(page_title="ECO 317 Fiscal Dashboard", layout="wide")
load_css()
st.title("AI-Assisted Fiscal Policy Dashboard")

params = dict(BASELINE_PARAMS)
with st.sidebar:
    st.header("Structural Parameters")
    params["habit"] = st.slider("Habit (h)", *SLIDER_BOUNDS["habit"], value=float(params["habit"]))
    params["psi"] = st.slider("Utilization Convexity (psi)", *SLIDER_BOUNDS["psi"], value=float(params["psi"]))
    params["theta_p"] = st.slider("Calvo Price (theta_p)", *SLIDER_BOUNDS["theta_p"], value=float(params["theta_p"]))
    params["theta_w"] = st.slider("Calvo Wage (theta_w)", *SLIDER_BOUNDS["theta_w"], value=float(params["theta_w"]))
    params["phi_b"] = st.slider("Debt Feedback (phi_b)", *SLIDER_BOUNDS["phi_b"], value=float(params["phi_b"]))
    params["lambda_rot"] = st.slider("Rule-of-Thumb Share (lambda)", *SLIDER_BOUNDS["lambda_rot"], value=float(params["lambda_rot"]))
    params["rho_x"] = st.slider("Shock Persistence (rho_x)", *SLIDER_BOUNDS["rho_x"], value=float(params["rho_x"]))
    params["sigma_x"] = st.slider("Shock Volatility (sigma_x)", *SLIDER_BOUNDS["sigma_x"], value=float(params["sigma_x"]))
    seed = st.number_input("Simulation Seed", min_value=0, value=42)

if params["phi_pi"] <= 1.0:
    st.error("Determinacy warning: keep phi_pi > 1 to satisfy the Taylor principle.")

solve_key = tuple(sorted((k, v) for k, v in params.items() if k != "sigma_x"))
model = cached_solve(solve_key)
diag = model["diagnostics"]
A = model["A_matrix"]
B = model["B_matrix"]
idx = model["variable_index"]
sidx = model["shock_index"]

sim_paths, sim_shocks = cached_simulate(A, B, SIMULATION_HORIZON, int(seed), float(params["sigma_x"]))
budget_check = check_budget_residuals(sim_paths, sim_shocks, A, B, idx, tol=1e-8)

model_moments = compute_moments(sim_paths, idx)
empirical_moments = cached_empirical()


def build_comparison_table(model_block, empirical_block):
    left = model_block.rename(columns={"value": "model"}).copy()
    right = empirical_block.rename(columns={"value": "empirical"}).copy()
    merged = left.merge(right, on=["series", "stat"], how="inner")
    merged["ratio_model_to_data"] = np.where(np.abs(merged["empirical"]) > 1e-12, merged["model"] / merged["empirical"], np.nan)
    merged["abs_diff"] = (merged["model"] - merged["empirical"]).abs()
    return merged

tab1, tab2 = st.tabs(["Model Fit (Unconditional Dynamics)", "Fiscal Exercises (Conditional Policy)"])

with tab1:
    if not diag.get("solver_success", False):
        suggestion = likely_failure_hint(params, diag.get("flag", "no_solution"))
        st.error(f"Solver status: {diag.get('flag', 'no_solution')}. {suggestion}")
    if not budget_check["ok"]:
        st.error(
            f"Government budget residual check failed (max abs={budget_check['max_abs_residual']:.2e}). "
            "Try less extreme parameters."
        )

    st.subheader("Business-Cycle Moments (T=1000)")
    tab1_can_plot = diag.get("solver_success", False) and budget_check["ok"]
    if tab1_can_plot:
        comparison = build_comparison_table(model_moments["hp_table"], empirical_moments["hp_table"])
        st.dataframe(comparison, use_container_width=True)
        st.caption("Comparison uses HP-filtered moments (lambda=1600). Raw model and data moments are also computed and stored.")
        st.markdown(tab1_summary(params, model_moments["raw"], comparison))

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(model_moments["frame"]["y_hat"].to_numpy(), label="Output", color="#2f80ed")
        ax.plot(model_moments["frame"]["c_hat"].to_numpy(), label="Consumption", color="#27ae60", alpha=0.8)
        ax.set_title("Simulated Output and Consumption")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    else:
        st.empty()

with tab2:
    if not diag.get("solver_success", False):
        suggestion = likely_failure_hint(params, diag.get("flag", "no_solution"))
        st.error(f"Solver status: {diag.get('flag', 'no_solution')}. {suggestion}")
    if not budget_check["ok"]:
        st.error(
            f"Government budget residual check failed (max abs={budget_check['max_abs_residual']:.2e}). "
            "Try less extreme parameters."
        )

    st.subheader("40-Quarter Fiscal IRFs")
    shock_choice = st.selectbox("Shock", ["Gc Shock", "GI Shock", "Labor-Tax Cut", "Capital-Tax Cut"])
    rule_choice = st.selectbox(
        "Financing Rule",
        ["Lump-Sum transfers", "Consumption Tax Hikes", "Labor Tax Hikes", "Capital Tax Hikes", "Government Spending Cuts"],
    )

    shock_key = shock_vector_name(shock_choice)
    tab2_can_plot = diag.get("solver_success", False) and budget_check["ok"]
    if tab2_can_plot:
        irf = compute_irf(A, B, idx, sidx, shock_key, rule_choice, horizon=IRF_HORIZON, shock_size=1.0)
        rule_ok, rule_msg = financing_rule_moves_with_debt(irf, idx, rule_choice)
        if not rule_ok:
            st.error(rule_msg)
            tab2_can_plot = False

    if tab2_can_plot:
        mult = compute_multipliers(
            irf,
            idx,
            beta=params["beta"],
            g_ss=model["steady_state"]["G"],
            y_ss=model["steady_state"]["Y"],
        )
        st.markdown(tab2_summary(shock_choice, rule_choice, mult))

        t = np.arange(IRF_HORIZON)
        fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        axs = axs.ravel()
        series = [
            ("Output", "y_hat", "#2f80ed"),
            ("Consumption", "c_hat", "#27ae60"),
            ("Investment", "i_hat", "#f2994a"),
            ("Government Debt", "b_hat", "#eb5757"),
        ]
        for ax, (label, var, color) in zip(axs, series):
            ax.plot(t, irf[:, idx[var]], color=color, linewidth=2)
            ax.axhline(0, color="black", linewidth=0.8)
            if mult["fiscal_drag_horizon"] is not None:
                ax.axvline(mult["fiscal_drag_horizon"], color="gray", linestyle="--", linewidth=1)
            ax.set_title(label)
            ax.set_xlabel("Quarter")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.empty()
