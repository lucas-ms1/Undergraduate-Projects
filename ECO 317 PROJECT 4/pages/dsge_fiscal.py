"""
pages/dsge_fiscal.py
====================
Module IV: DSGE & Fiscal Policy (from Assignment 3)

Medium-scale Smets-Wouters style DSGE model with:
  - Global friction sliders (habit, utilization, Calvo stickiness, etc.)
  - Sub-Tab A: Model Fit (1000-period simulation, unconditional moments)
  - Sub-Tab B: Fiscal Exercises (shock selection, financing rules, IRFs, multipliers)
  - AI "intelligent" summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from dsge_engine.config import BASELINE_PARAMS, IRF_HORIZON, SIMULATION_HORIZON, SLIDER_BOUNDS
from dsge_engine.dsge.calibration import override_parameters
from dsge_engine.dsge.model import solve_model_objects
from dsge_engine.policy.financing import FINANCING_RULES, ui_label_to_canonical
from dsge_engine.policy.multipliers import (
    cumulative_multiplier, fiscal_drag_horizon, impact_multiplier,
)
from dsge_engine.policy.shocks import (
    build_unit_impulse_shock,
    ui_label_to_canonical as shock_label_to_canonical,
)
from dsge_engine.simulation.econometrics import run_all_econometrics
from dsge_engine.simulation.empirical import build_comparison_table, load_empirical_moments
from dsge_engine.simulation.irf import simulate_irf
from dsge_engine.simulation.moments import compute_all_moments
from dsge_engine.simulation.simulate import run_simulation
from dsge_engine.solvers.rational_expectations import likely_failure_hint, solve_with_qz
from dsge_engine.utils.summaries import generate_tab1_commentary, generate_tab2_briefing


# ---------------------------------------------------------------------------
# Cached wrappers
# ---------------------------------------------------------------------------
@st.cache_data
def cached_solve(params_tuple, financing_rule="lump_sum"):
    p = dict(params_tuple)
    return solve_model_objects(p, financing_rule=financing_rule)


@st.cache_data
def cached_simulate(A_bytes, B_bytes, A_shape, B_shape, T, seed, obs_items, shock_stds):
    A = np.frombuffer(A_bytes).reshape(A_shape)
    B = np.frombuffer(B_bytes).reshape(B_shape)
    obs_index = dict(obs_items)
    shock_stds_arr = np.asarray(shock_stds, dtype=float)
    return run_simulation(
        A, B, T=T, seed=seed, obs_index=obs_index, shock_stds=shock_stds_arr
    )


@st.cache_data(ttl=24 * 60 * 60)
def cached_empirical_moments():
    return load_empirical_moments()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slider(label, bounds_key, param_key):
    lo, hi, step = SLIDER_BOUNDS[bounds_key]
    return st.slider(label, min_value=lo, max_value=hi, step=step,
                     value=float(BASELINE_PARAMS[param_key]),
                     key=f"dsge_{bounds_key}")


def observable_index_from_model(variable_index):
    names = {
        "y": "y_hat", "c": "c_hat", "i": "i_hat",
        "l": "l_hat", "pi": "pi_hat", "r": "i_nom_hat",
        "m_policy": "m_policy_hat",
    }
    return {obs: variable_index[var] for obs, var in names.items() if var in variable_index}


def shock_std_vector(params, shock_index):
    mapping = {
        "tfp": "sigma_a", "g_c": "sigma_g", "g_i": "sigma_gi",
        "monetary": "sigma_m", "risk": "sigma_rp",
        "lump_sum": None, "tau_c": None, "tau_l": None, "tau_k": None,
    }
    stds = np.zeros(len(shock_index))
    for shock, param_name in mapping.items():
        if shock in shock_index and param_name is not None:
            stds[shock_index[shock]] = float(params.get(param_name, 0.0))
    return stds


def impact_aligned_series(irf_result):
    return {name: values[1:] for name, values in irf_result.series.items()}


def simulate_fiscal_irf(params_key, financing_rule, shock_name, tracked_variables):
    rule_model = cached_solve(params_key)
    rule_diag = rule_model["diagnostics"]
    if not rule_diag.get("solver_success", False):
        return rule_model, None, None
    fiscal_shock = build_unit_impulse_shock(
        shock_name, rule_model["shock_index"], impulse_size=1.0
    )
    irf_result = simulate_irf(
        rule_model["A_matrix"],
        rule_model["B_matrix"],
        fiscal_shock.vector,
        rule_model["variable_index"],
        horizon=IRF_HORIZON,
        tracked_variables=tracked_variables,
        financing_rule=financing_rule,
        shock_index=rule_model["shock_index"],
        phi_b=dict(params_key).get("phi_b", 0.05),
    )
    return rule_model, irf_result, impact_aligned_series(irf_result)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render():
    """Render the complete DSGE & Fiscal Policy module."""
    st.title("Module IV: DSGE & Fiscal Policy")
    st.markdown(
        "A medium-scale **Smets-Wouters style DSGE** model with rule-of-thumb "
        "households, habit formation, Calvo pricing, and comprehensive fiscal "
        "policy analysis."
    )

    # ── Sidebar: structural parameters ───────────────────────────────────
    with st.sidebar:
        st.header("DSGE Parameters")
        habit      = _slider(r"Habit h",                     "habit",      "habit")
        psi_util   = _slider(r"Utilization ψ",               "psi_util",   "psi_util")
        theta_p    = _slider(r"Calvo Price θ_p",             "theta_p",    "theta_p")
        theta_w    = _slider(r"Calvo Wage θ_w",              "theta_w",    "theta_w")
        phi_b      = _slider(r"Debt Feedback φ_b",           "phi_b",      "phi_b")
        lambda_rot = _slider(r"Rule-of-Thumb λ",             "lambda_rot", "lambda_rot")
        st.markdown("---")
        st.subheader("Shock Parameters")
        rho_a   = _slider(r"TFP ρ_a",      "rho_a",   "rho_a")
        sigma_a = _slider(r"TFP σ_a",      "sigma_a", "sigma_a")
        seed = st.number_input("Simulation Seed", min_value=0, value=42, key="dsge_seed")

    # Build parameter dict + solve
    slider_overrides = {
        "habit": habit, "psi_util": psi_util, "theta_p": theta_p,
        "theta_w": theta_w, "phi_b": phi_b, "lambda_rot": lambda_rot,
        "rho_a": rho_a, "sigma_a": sigma_a,
    }
    params = override_parameters(slider_overrides)

    solve_key = tuple(sorted((k, v) for k, v in params.items() if not k.startswith("sigma_")))
    model = cached_solve(solve_key)
    diag = model["diagnostics"]
    ss = model["steady_state"]
    A = model["A_matrix"]
    B = model["B_matrix"]
    idx = model["variable_index"]
    sidx = model["shock_index"]
    solver_ok = diag.get("solver_success", False)

    # Simulate if solver succeeded
    series = {}
    raw_moments = {}
    hp_moments = {}
    econ_results = {}
    if solver_ok and A is not None and B is not None:
        shock_stds = shock_std_vector(params, sidx)
        series = cached_simulate(
            A.tobytes(), B.tobytes(), A.shape, B.shape,
            SIMULATION_HORIZON, int(seed),
            tuple(sorted(observable_index_from_model(idx).items())),
            tuple(shock_stds.tolist()),
        )
        raw_moments, hp_moments = compute_all_moments(series, output_key="y")
        econ_results = run_all_econometrics(series, structural_params=params)
    else:
        solver_ok = False

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_spec, tab_fit, tab_fiscal = st.tabs([
        "Model Specification",
        "Model Fit (Unconditional)",
        "Fiscal Exercises (Conditional)",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Model Specification
    # ══════════════════════════════════════════════════════════════════════
    with tab_spec:
        st.subheader("Linearised Smets-Wouters DSGE Model")

        st.markdown("#### Household Block")
        st.latex(r"\Lambda_t^o = (C_t^o - hC_{t-1}^o)^{-\sigma} - \beta h\,\mathbb{E}_t[(C_{t+1}^o - hC_t^o)^{-\sigma}]")
        st.latex(r"\text{Euler: } \Lambda_t^o = \beta\,\mathbb{E}_t[\Lambda_{t+1}^o\,(1+r_{t+1})]")
        st.latex(r"\text{Aggregate: } C_t = (1-\lambda)\,C_t^o + \lambda\,C_t^{rot}")

        st.markdown("#### Price & Wage Phillips Curves")
        st.latex(r"\hat{\pi}_t = \beta\,\mathbb{E}_t[\hat{\pi}_{t+1}] + \kappa_p\,\widehat{mc}_t")
        st.latex(r"\hat{\pi}_t^w = \beta\,\mathbb{E}_t[\hat{\pi}_{t+1}^w] + \kappa_w\,(\widehat{mrs}_t - \hat{w}_t)")

        st.markdown("#### Government Budget Constraint")
        st.latex(r"B_{t+1} = (1+r_t)\,B_t + G_t + T_t - \tau_C C_t - \tau_L W_t L_t - \tau_K r^k_t K_t")

        st.markdown("#### Taylor Rule")
        st.latex(r"\hat{\imath}_t = \rho_i\,\hat{\imath}_{t-1} + (1-\rho_i)[\phi_\pi\,\hat{\pi}_t + \phi_y\,\hat{y}_t] + \varepsilon_{i,t}")

        # Calibration table
        st.markdown("---")
        st.subheader("Current Calibration")
        cal_col1, cal_col2 = st.columns(2)
        with cal_col1:
            st.markdown("**Structural Parameters**")
            cal_rows = [
                {"Param": "β", "Value": f"{params['beta']:.3f}", "Desc": "Discount factor"},
                {"Param": "α", "Value": f"{params['alpha']:.2f}", "Desc": "Capital share"},
                {"Param": "h", "Value": f"{params['habit']:.2f}", "Desc": "Habit"},
                {"Param": "θ_p", "Value": f"{params['theta_p']:.2f}", "Desc": "Price stickiness"},
                {"Param": "θ_w", "Value": f"{params['theta_w']:.2f}", "Desc": "Wage stickiness"},
                {"Param": "λ", "Value": f"{params['lambda_rot']:.2f}", "Desc": "Rule-of-thumb share"},
                {"Param": "φ_b", "Value": f"{params['phi_b']:.3f}", "Desc": "Debt feedback"},
            ]
            st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)
        with cal_col2:
            st.markdown("**Steady-State Levels**")
            ss_rows = [
                {"Var": "Y", "Value": f"{ss['Y']:.4f}", "Desc": "Output"},
                {"Var": "C", "Value": f"{ss['C']:.4f}", "Desc": "Consumption"},
                {"Var": "I", "Value": f"{ss['I']:.4f}", "Desc": "Investment"},
                {"Var": "K", "Value": f"{ss['K']:.4f}", "Desc": "Capital"},
                {"Var": "L", "Value": f"{ss['L']:.4f}", "Desc": "Hours"},
                {"Var": "C/Y", "Value": f"{ss['c_y']:.3f}", "Desc": "Consumption share"},
            ]
            st.dataframe(pd.DataFrame(ss_rows), use_container_width=True, hide_index=True)

        with st.expander("Solver Diagnostics"):
            st.write(f"**Status:** {diag.get('flag', 'N/A')}")
            st.write(f"**BK condition satisfied:** {diag.get('bk_ok', 'N/A')}")
            st.write(f"**A matrix shape:** {A.shape if A is not None else 'N/A'}")
            st.write(f"**B matrix shape:** {B.shape if B is not None else 'N/A'}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Model Fit
    # ══════════════════════════════════════════════════════════════════════
    with tab_fit:
        if not solver_ok:
            hint = likely_failure_hint(params, diag.get("flag", "no_solution"))
            st.error(f"Solver failed: {diag.get('flag', 'no_solution')}. {hint}")
            return

        st.subheader(r"Business-Cycle Moments (T = 1,000)")
        var_labels = {"y": "Output", "c": "Consumption", "i": "Investment", "l": "Hours", "pi": "Inflation"}
        moment_rows = []
        for var in ["y", "c", "i", "l", "pi"]:
            moment_rows.append({
                "Series": var_labels[var],
                "Variance": f"{raw_moments.get(f'var_{var}', 0):.6f}",
                "Corr(x, y)": f"{raw_moments.get(f'corr_{var}', 0):.4f}",
                "AR(1)": f"{raw_moments.get(f'ac1_{var}', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(moment_rows), use_container_width=True, hide_index=True)

        # Model vs empirical
        st.subheader("Model-vs-Empirical HP-Filtered Moments")
        try:
            empirical_moments, _ = cached_empirical_moments()
            comparison = build_comparison_table(hp_moments, empirical_moments)
            st.dataframe(comparison, use_container_width=True)
        except Exception as exc:
            st.warning(f"Empirical FRED moments unavailable: {exc}")
            empirical_moments = None

        # Simulation plot
        st.subheader("Simulated Business Cycle")
        fig_sim, ax_sim = plt.subplots(1, 1, figsize=(12, 3.5))
        t_axis = np.arange(len(series["y"]))
        ax_sim.plot(t_axis, series["y"], label=r"$\hat{y}_t$", color="#2f80ed", linewidth=1.2)
        ax_sim.plot(t_axis, series["c"], label=r"$\hat{c}_t$", color="#27ae60", linewidth=1.0, alpha=0.85)
        ax_sim.plot(t_axis, series["i"], label=r"$\hat{\imath}_t$", color="#f2994a", linewidth=1.0, alpha=0.85)
        ax_sim.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax_sim.set_xlabel("Quarter")
        ax_sim.set_ylabel("Log deviation from SS")
        ax_sim.legend(loc="upper right", fontsize=9)
        fig_sim.tight_layout()
        st.pyplot(fig_sim, clear_figure=True)

        # Econometrics
        st.subheader("Econometric Analysis")
        col_t, col_c = st.columns(2)
        with col_t:
            st.markdown(r"**Taylor-Rule Recovery**")
            if "taylor_rule" in econ_results:
                tr = econ_results["taylor_rule"]
                st.dataframe(pd.DataFrame(tr.summary_rows()), use_container_width=True, hide_index=True)
        with col_c:
            st.markdown(r"**Consumption Smoothness**")
            if "consumption_smoothness" in econ_results:
                cs = econ_results["consumption_smoothness"]
                st.dataframe(pd.DataFrame(cs.summary_rows()), use_container_width=True, hide_index=True)

        # Commentary
        st.subheader("Economic Commentary")
        commentary = generate_tab1_commentary(
            moments=raw_moments, h=habit, utilisation=psi_util,
            price_stickiness=theta_p, wage_stickiness=theta_w, debt_feedback=phi_b,
            empirical_moments=empirical_moments if 'empirical_moments' in dir() else None,
            econometrics=econ_results,
        )
        st.markdown(commentary)

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Fiscal Exercises
    # ══════════════════════════════════════════════════════════════════════
    with tab_fiscal:
        if not solver_ok:
            hint = likely_failure_hint(params, diag.get("flag", "no_solution"))
            st.error(f"Solver failed: {diag.get('flag', 'no_solution')}. {hint}")
            return

        st.subheader("40-Quarter Fiscal IRFs")
        col_shock, col_rule = st.columns(2)
        with col_shock:
            shock_choice = st.selectbox(
                "Shock",
                ["Gc Shock", "GI Shock", "Labor-Tax Cut", "Capital-Tax Cut"],
                key="dsge_shock",
            )
        with col_rule:
            rule_choice = st.selectbox(
                "Financing Rule",
                ["Lump-Sum transfers", "Consumption Tax Hikes", "Labor Tax Hikes",
                 "Capital Tax Hikes", "Government Spending Cuts"],
                key="dsge_rule",
            )

        shock_canonical = shock_label_to_canonical(shock_choice)
        rule_canonical = ui_label_to_canonical(rule_choice)
        tracked = (
            "y_hat", "c_hat", "i_hat", "b_hat", "pi_hat",
            "l_hat", "i_nom_hat", "g_c_hat", "g_i_hat",
        )
        rule_model, irf_result, irf_aligned = simulate_fiscal_irf(
            solve_key, rule_canonical, shock_canonical, tracked
        )
        rule_diag = rule_model["diagnostics"]
        if irf_result is None:
            hint = likely_failure_hint(params, rule_diag.get("flag", "no_solution"))
            st.error(f"Rule solve failed: {hint}")
            return

        y_irf = irf_aligned["y_hat"]
        if shock_canonical == "gc":
            g_irf = irf_aligned["g_c_hat"]
            g_ss = rule_model["steady_state"]["G_c"]
        elif shock_canonical == "gi":
            g_irf = irf_aligned["g_i_hat"]
            g_ss = rule_model["steady_state"]["G_i"]
        else:
            g_irf = y_irf.copy()
            g_ss = rule_model["steady_state"]["Y"]

        y_ss = rule_model["steady_state"]["Y"]
        im = impact_multiplier(y_irf, g_irf, y_ss, g_ss)
        cm = cumulative_multiplier(y_irf, g_irf, y_ss, g_ss, beta=params["beta"])
        drag = fiscal_drag_horizon(y_irf)

        col_im, col_cm, col_drag = st.columns(3)
        col_im.metric("Impact Multiplier", f"{im:.3f}" if np.isfinite(im) else "N/A")
        col_cm.metric("Cumulative Multiplier", f"{cm:.3f}" if np.isfinite(cm) else "N/A")
        col_drag.metric("Fiscal Drag Horizon", f"Q{drag}" if drag is not None else "None (40Q)")

        # IRF plots
        t_irf = np.arange(IRF_HORIZON)
        fig_irf, axs = plt.subplots(2, 2, figsize=(12, 7))
        axs = axs.ravel()
        irf_series = [
            (r"Output $\hat{y}_t$", "y_hat", "#2f80ed"),
            (r"Consumption $\hat{c}_t$", "c_hat", "#27ae60"),
            (r"Investment $\hat{\imath}_t$", "i_hat", "#f2994a"),
            (r"Gov. Debt $\hat{B}_t$", "b_hat", "#eb5757"),
        ]
        for ax, (label, var, color) in zip(axs, irf_series):
            ax.plot(t_irf, irf_aligned[var], color=color, linewidth=2)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            if drag is not None:
                ax.axvline(drag, color="gray", linestyle=":", linewidth=1,
                           label=f"Drag t*={drag}")
                ax.legend(fontsize=8)
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Quarter")
            ax.set_ylabel("Log deviation")
        fig_irf.suptitle(f"{shock_choice} under {rule_choice}",
                         fontsize=13, fontweight="bold", y=1.01)
        fig_irf.tight_layout()
        st.pyplot(fig_irf, clear_figure=True)

        # Composite overlay
        st.subheader("Output IRF: All Financing Rules")
        rule_labels = ["Lump-Sum transfers", "Consumption Tax Hikes", "Labor Tax Hikes",
                       "Capital Tax Hikes", "Government Spending Cuts"]
        rule_colors = ["#2f80ed", "#27ae60", "#f2994a", "#eb5757", "#9b59b6"]

        fig_comp, ax_comp = plt.subplots(1, 1, figsize=(12, 5))
        cum_mults = {}
        for rlabel, rcolor in zip(rule_labels, rule_colors):
            canonical_rule = ui_label_to_canonical(rlabel)
            model_r, irf_r, aligned_r = simulate_fiscal_irf(
                solve_key, canonical_rule, shock_canonical, ("y_hat", "g_c_hat", "g_i_hat")
            )
            if irf_r is None:
                continue
            y_r = aligned_r["y_hat"]
            ax_comp.plot(t_irf, y_r, color=rcolor, linewidth=1.8, label=rlabel)
            ss_r = model_r["steady_state"]
            if shock_canonical == "gc":
                cm_r = cumulative_multiplier(y_r, aligned_r["g_c_hat"], ss_r["Y"], ss_r["G_c"], beta=params["beta"])
            elif shock_canonical == "gi":
                cm_r = cumulative_multiplier(y_r, aligned_r["g_i_hat"], ss_r["Y"], ss_r["G_i"], beta=params["beta"])
            else:
                cm_r = cumulative_multiplier(y_r, y_r, ss_r["Y"], ss_r["Y"], beta=params["beta"])
            cum_mults[rlabel] = cm_r

        ax_comp.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax_comp.set_xlabel("Quarter")
        ax_comp.set_ylabel(r"$\hat{y}_t$")
        ax_comp.set_title(f"Output IRF: {shock_choice} - All Rules")
        ax_comp.legend(fontsize=9)
        fig_comp.tight_layout()
        st.pyplot(fig_comp, clear_figure=True)

        cm_rows = [{"Rule": k, "CM": f"{v:.4f}" if np.isfinite(v) else "N/A"}
                    for k, v in cum_mults.items()]
        st.dataframe(pd.DataFrame(cm_rows), use_container_width=True, hide_index=True)

        # Policy briefing
        st.subheader("Automated Policy Briefing")
        briefing = generate_tab2_briefing(
            shock_name=shock_choice, financing_rule=rule_choice,
            impact_mult=im, cumulative_mult=cm, drag_horizon=drag,
        )
        st.markdown(briefing)
