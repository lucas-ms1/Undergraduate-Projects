"""
app.py
ECO 317 Project 3 - AI-Assisted Fiscal Policy Dashboard
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import BASELINE_PARAMS, IRF_HORIZON, SIMULATION_HORIZON, SLIDER_BOUNDS
from dsge.calibration import override_parameters
from dsge.model import solve_model_objects
from policy.financing import FINANCING_RULES, ui_label_to_canonical
from policy.multipliers import (
    cumulative_multiplier,
    fiscal_drag_horizon,
    impact_multiplier,
)
from policy.shocks import build_unit_impulse_shock, ui_label_to_canonical as shock_label_to_canonical
from simulation.econometrics import run_all_econometrics
from simulation.empirical import build_comparison_table, load_empirical_moments
from simulation.irf import simulate_irf
from simulation.moments import compute_all_moments
from simulation.simulate import run_simulation
from solvers.rational_expectations import likely_failure_hint, solve_with_qz
from utils.summaries import generate_tab1_commentary, generate_tab2_briefing


# Page config
st.set_page_config(page_title="ECO 317 Fiscal Dashboard", layout="wide")


# Sidebar
def _slider(label, bounds_key, param_key):
    lo, hi, step = SLIDER_BOUNDS[bounds_key]
    return st.slider(label, min_value=lo, max_value=hi, step=step,
                     value=float(BASELINE_PARAMS[param_key]))

with st.sidebar:
    dark_mode = st.toggle("Dark Mode", value=True)
    st.markdown("---")
    st.header("Structural Parameters")
    habit      = _slider(r"Habit $h$",                       "habit",      "habit")
    psi_util   = _slider(r"Utilization Convexity $\psi$",    "psi_util",   "psi_util")
    theta_p    = _slider(r"Calvo Price $\theta_p$",          "theta_p",    "theta_p")
    theta_w    = _slider(r"Calvo Wage $\theta_w$",           "theta_w",    "theta_w")
    phi_b      = _slider(r"Debt Feedback $\phi_b$",          "phi_b",      "phi_b")
    lambda_rot = _slider(r"Rule-of-Thumb Share $\lambda$",   "lambda_rot", "lambda_rot")
    st.markdown("---")
    st.subheader("Shock Parameters")
    rho_a   = _slider(r"TFP Persistence $\rho_a$",  "rho_a",   "rho_a")
    sigma_a = _slider(r"TFP Shock Std $\sigma_a$",  "sigma_a", "sigma_a")
    seed = st.number_input("Simulation Seed", min_value=0, value=42)
    if BASELINE_PARAMS["phi_pi"] <= 1.0:
        st.error(r"$\phi_\pi \leq 1$: Taylor principle violated.")


# Dynamic CSS
if dark_mode:
    theme_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');
    html, body, [class*="css"] { font-family: "EB Garamond", Garamond, "Times New Roman", serif; }
    .stApp { background: #0e1a2b; color: #f0f2f6; }
    [data-testid="stSidebar"] { background: #0a1420; }
    [data-testid="stSidebar"] .stMarkdown { color: #d0d8e8; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: #d0d8e8 !important; }
    h1, h2, h3 { font-family: "EB Garamond", Garamond, serif; color: #e8ecf4; }
    .stTabs [data-baseweb="tab"] { font-family: "EB Garamond", Garamond, serif; font-size: 1.1rem; color: #c0c8d8; }
    .stTabs [aria-selected="true"] { color: #ffffff; border-bottom-color: #4a90d9; }
    [data-testid="stMetric"] {
        background: #162236; color: #f0f2f6;
        border: 1px solid #2a3a52; border-radius: 8px; padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] { color: #a0b0c8 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #f0f2f6 !important; }
    .katex, .katex * { color: #f0f2f6 !important; }
    .stSelectbox label, .stNumberInput label { color: #f0f2f6 !important; }
    </style>
    """
else:
    theme_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');
    html, body, [class*="css"] { font-family: "EB Garamond", Garamond, "Times New Roman", serif; }
    .stApp { background: #fafafa; color: #1a1a1a; }
    [data-testid="stSidebar"] { background: #f0f2f5; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: #1a1a1a !important; }
    h1, h2, h3 { font-family: "EB Garamond", Garamond, serif; color: #1a1a1a; }
    .stTabs [data-baseweb="tab"] { font-family: "EB Garamond", Garamond, serif; font-size: 1.1rem; color: #444; }
    .stTabs [aria-selected="true"] { color: #111; border-bottom-color: #2f80ed; }
    [data-testid="stMetric"] {
        background: #ffffff; color: #1a1a1a;
        border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] { color: #555 !important; }
    [data-testid="stMetricValue"] { color: #111 !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #1a1a1a !important; }
    .katex, .katex * { color: #1a1a1a !important; }
    .stSelectbox label, .stNumberInput label { color: #1a1a1a !important; }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)
st.title("AI-Assisted Fiscal Policy Dashboard")


# Build parameter dict + solve + simulate
slider_overrides = {
    "habit": habit, "psi_util": psi_util, "theta_p": theta_p,
    "theta_w": theta_w, "phi_b": phi_b, "lambda_rot": lambda_rot,
    "rho_a": rho_a, "sigma_a": sigma_a,
}
params = override_parameters(slider_overrides)

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

solve_key = tuple(sorted((k, v) for k, v in params.items() if not k.startswith("sigma_")))
model = cached_solve(solve_key)
diag = model["diagnostics"]
ss = model["steady_state"]
A = model["A_matrix"]
B = model["B_matrix"]
idx = model["variable_index"]
sidx = model["shock_index"]
solver_ok = diag.get("solver_success", False)

series = {}
raw_moments = {}
hp_moments = {}
econ_results = {}
if solver_ok and A is not None and B is not None:
    shock_stds = shock_std_vector(params, sidx)
    series = cached_simulate(
        A.tobytes(), B.tobytes(), A.shape, B.shape,
        SIMULATION_HORIZON, int(seed), tuple(sorted(observable_index_from_model(idx).items())),
        tuple(shock_stds.tolist()),
    )
    raw_moments, hp_moments = compute_all_moments(series, output_key="y")
    econ_results = run_all_econometrics(series, structural_params=params)
else:
    solver_ok = False


# Tabs
tab_model, tab1, tab2 = st.tabs([
    "Model Specification",
    "Model Fit (Unconditional Dynamics)",
    "Fiscal Exercises (Conditional Policy)",
])

# MODEL TAB
with tab_model:
    st.subheader("Linearised Smets-Wouters DSGE Model")

    st.markdown("#### Household Block")
    st.latex(r"""
    \Lambda_t^o = \left(C_t^o - h C_{t-1}^o\right)^{-\sigma}
    - \beta h\,\mathbb{E}_t!\left[\left(C_{t+1}^o - h C_t^o\right)^{-\sigma}\right]
    """)
    st.latex(r"""
    \text{Euler: } \quad \Lambda_t^o = \beta\,\mathbb{E}_t!\left[\Lambda_{t+1}^o\,(1+r_{t+1})\right]
    """)
    st.latex(r"""
    \text{Rule-of-thumb: } \quad (1+\tau_{C,t})\,C_t^{rot} = (1-\tau_{L,t})\,W_t L_t^{rot} + T_t^{rot}
    """)
    st.latex(r"""
    \text{Aggregate: } \quad C_t = (1-\lambda)\,C_t^o + \lambda\,C_t^{rot}
    """)

    st.markdown("#### Firms and Price/Wage Stickiness")
    st.latex(r"""
    \hat{\pi}_t = \beta\,\mathbb{E}_t[\hat{\pi}_{t+1}] + \kappa_p\,\widehat{mc}_t,
    \qquad \kappa_p = \frac{(1-\theta_p)(1-\beta\theta_p)}{\theta_p}
    """)
    st.latex(r"""
    \hat{\pi}_t^w = \beta\,\mathbb{E}_t[\hat{\pi}_{t+1}^w]
    + \kappa_w\,(\widehat{mrs}_t - \hat{w}_t),
    \qquad \kappa_w = \frac{(1-\theta_w)(1-\beta\theta_w)}{\theta_w(1+\varphi\,\varepsilon_w)}
    """)
    st.latex(r"""
    \widehat{mc}_t = (1-\alpha)\,\hat{w}_t + \alpha\,\hat{r}^k_t - \hat{A}_t
    """)

    st.markdown("#### Government and Fiscal Block")
    st.latex(r"""
    B_{t+1} = (1+r_t)\,B_t + G_{c,t} + G_{I,t} + T_t
    - \tau_{C,t}\,C_t - \tau_{L,t}\,W_t L_t - \tau_{K,t}\,r^k_t K_t
    """)
    st.latex(r"""
    \text{Fiscal rule: } \quad \hat{\tau}_{X,t} = \rho_X\,\hat{\tau}_{X,t-1}
    + \phi_{b,X}\,\hat{B}_t + \varepsilon_{X,t}
    """)

    st.markdown("#### Monetary Policy (Taylor Rule)")
    st.latex(r"""
    \hat{\imath}_t = \rho_i\,\hat{\imath}_{t-1}
    + (1-\rho_i)!\left[\phi_\pi\,\hat{\pi}_t + \phi_y\,\hat{y}_t\right]
    + \varepsilon_{i,t}
    """)
    st.latex(r"""
    \text{Fisher: } \quad \hat{r}_t = \hat{\imath}_t - \mathbb{E}_t\hat{\pi}_{t+1}
    """)

    st.markdown("#### State-Space Representation")
    st.latex(r"""
    x_{t+1} = A\,x_t + B\,\varepsilon_{t+1},
    \qquad x = [\hat{y},\;\hat{c},\;\hat{\imath},\;\hat{B},\;\hat{\pi},\;\hat{\ell},\;\hat{r}]'
    """)

    st.markdown("---")
    st.subheader("Current Calibration")
    cal_col1, cal_col2 = st.columns(2)

    with cal_col1:
        st.markdown("**Structural Parameters**")
        cal_rows = [
            {"Parameter": "\u03b2", "Value": f"{params['beta']:.3f}", "Description": "Discount factor"},
            {"Parameter": "\u03c3", "Value": f"{params['sigma']:.2f}", "Description": "CRRA coefficient"},
            {"Parameter": "\u03b1", "Value": f"{params['alpha']:.2f}", "Description": "Capital share"},
            {"Parameter": "\u03b4", "Value": f"{params['delta']:.3f}", "Description": "Depreciation rate"},
            {"Parameter": "h", "Value": f"{params['habit']:.2f}", "Description": "Habit persistence"},
            {"Parameter": "\u03c8", "Value": f"{params['psi_util']:.2f}", "Description": "Utilization convexity"},
            {"Parameter": "\u03b8\u209a", "Value": f"{params['theta_p']:.2f}", "Description": "Calvo price stickiness"},
            {"Parameter": "\u03b8_w", "Value": f"{params['theta_w']:.2f}", "Description": "Calvo wage stickiness"},
            {"Parameter": "\u03bb", "Value": f"{params['lambda_rot']:.2f}", "Description": "Rule-of-thumb share"},
            {"Parameter": "\u03d5_b", "Value": f"{params['phi_b']:.3f}", "Description": "Debt feedback"},
        ]
        st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)

    with cal_col2:
        st.markdown("**Steady-State Levels**")
        ss_rows = [
            {"Variable": "Y", "Value": f"{ss['Y']:.4f}", "Description": "Output"},
            {"Variable": "C", "Value": f"{ss['C']:.4f}", "Description": "Consumption"},
            {"Variable": "I", "Value": f"{ss['I']:.4f}", "Description": "Investment"},
            {"Variable": "K", "Value": f"{ss['K']:.4f}", "Description": "Capital stock"},
            {"Variable": "L", "Value": f"{ss['L']:.4f}", "Description": "Hours worked"},
            {"Variable": "G", "Value": f"{ss['G']:.4f}", "Description": "Government spending"},
            {"Variable": "B", "Value": f"{ss['B']:.4f}", "Description": "Government debt"},
            {"Variable": "w", "Value": f"{ss['w']:.4f}", "Description": "Real wage"},
            {"Variable": "r\u1d4f", "Value": f"{ss['rk']:.4f}", "Description": "Capital rental rate"},
            {"Variable": "C/Y", "Value": f"{ss['c_y']:.3f}", "Description": "Consumption share"},
        ]
        st.dataframe(pd.DataFrame(ss_rows), use_container_width=True, hide_index=True)

    st.markdown("**Monetary Policy (Taylor Rule)**")
    mp_rows = [
        {"Parameter": "rho_i", "Value": f"{params['rho_i']:.2f}", "Description": "Interest-rate smoothing"},
        {"Parameter": "phi_pi", "Value": f"{params['phi_pi']:.3f}", "Description": "Inflation response"},
        {"Parameter": "phi_y", "Value": f"{params['phi_y']:.3f}", "Description": "Output-gap response"},
    ]
    st.dataframe(pd.DataFrame(mp_rows), use_container_width=True, hide_index=True)

    with st.expander("Solver Diagnostics"):
        st.write(f"**Status:** {diag.get('flag', 'N/A')}")
        st.write(f"**Equation coverage OK:** {diag.get('equation_coverage_ok', 'N/A')}")
        if "eigenvalues" in diag:
            eigs = diag["eigenvalues"]
            st.write(f"**Eigenvalues:** {len(eigs)} total, "
                     f"{diag.get('explosive_roots', '?')} explosive")
            st.write(f"**BK condition satisfied:** {diag.get('bk_ok', 'N/A')}")
        st.write(f"**A matrix shape:** {A.shape if A is not None else 'not available'}")
        st.write(f"**B matrix shape:** {B.shape if B is not None else 'not available'}")


# TAB 1: Unconditional Dynamics
with tab1:
    if not solver_ok:
        hint = likely_failure_hint(params, diag.get("flag", "no_solution"))
        st.error(f"Solver status: {diag.get('flag', 'no_solution')}. {hint}")
        st.info("Simulation, moments, and econometrics are skipped until BK conditions are satisfied.")
    else:
        st.subheader(r"Business-Cycle Moments ($T = 1{,}000$)")
        var_labels = {"y": "Output", "c": "Consumption", "i": "Investment", "l": "Hours", "pi": "Inflation"}
        moment_names = ["y", "c", "i", "l", "pi"]
        moment_rows = []
        for var in moment_names:
            moment_rows.append({
                "Series": var_labels[var],
                "Variance": f"{raw_moments.get(f'var_{var}', 0):.6f}",
                "Corr(x, y)": f"{raw_moments.get(f'corr_{var}', 0):.4f}",
                "AR(1)": f"{raw_moments.get(f'ac1_{var}', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(moment_rows), use_container_width=True, hide_index=True)

        empirical_moments = None
        st.subheader("Model-vs-Empirical HP-Filtered Moments")
        try:
            empirical_moments, _ = cached_empirical_moments()
            comparison = build_comparison_table(hp_moments, empirical_moments)
            st.dataframe(comparison, use_container_width=True)
        except Exception as exc:
            st.warning(f"Empirical FRED moments unavailable: {exc}")

        st.subheader(r"Simulated $\hat{y}_t$, $\hat{c}_t$, $\hat{\imath}_t$")
        fig_sim, ax_sim = plt.subplots(1, 1, figsize=(12, 3.5))
        t_axis = np.arange(len(series["y"]))
        ax_sim.plot(t_axis, series["y"], label=r"$\hat{y}_t$ (Output)", color="#2f80ed", linewidth=1.2)
        ax_sim.plot(t_axis, series["c"], label=r"$\hat{c}_t$ (Consumption)", color="#27ae60", linewidth=1.0, alpha=0.85)
        ax_sim.plot(t_axis, series["i"], label=r"$\hat{\imath}_t$ (Investment)", color="#f2994a", linewidth=1.0, alpha=0.85)
        ax_sim.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax_sim.set_xlabel("Quarter")
        ax_sim.set_ylabel("Log deviation from SS")
        ax_sim.legend(loc="upper right", fontsize=9)
        ax_sim.set_title("Unconditional Simulation (log-deviations from steady state)")
        fig_sim.tight_layout()
        st.pyplot(fig_sim, clear_figure=True)

        st.subheader("Econometric Analysis")
        col_taylor, col_consumption = st.columns(2)
        with col_taylor:
            st.markdown(r"**Taylor-Rule Recovery**: $\hat{\imath}_t = c + \hat{\rho}_i \hat{\imath}_{t-1} + \hat{\phi}_\pi \hat{\pi}_t + \hat{\phi}_y \hat{y}_t$")
            if "taylor_rule" in econ_results:
                tr = econ_results["taylor_rule"]
                st.dataframe(pd.DataFrame(tr.summary_rows()), use_container_width=True, hide_index=True)
        with col_consumption:
            st.markdown(r"**Consumption Smoothness**: $\hat{c}_t = c + \alpha_1 \hat{y}_t + \alpha_2 \hat{y}_{t-1} + \alpha_3 \hat{c}_{t-1}$")
            if "consumption_smoothness" in econ_results:
                cs = econ_results["consumption_smoothness"]
                st.dataframe(pd.DataFrame(cs.summary_rows()), use_container_width=True, hide_index=True)

        if "variance_ratios" in econ_results:
            st.markdown(r"**Variance Ratios** $\mathrm{Var}(\hat{x})/\mathrm{Var}(\hat{y})$ **(bootstrapped 95% CI)**")
            vr_rows = []
            for vr in econ_results["variance_ratios"]:
                vr_rows.append({
                    "Ratio": vr.name,
                    "Estimate": f"{vr.ratio:.4f}",
                    "SE": f"{vr.se:.4f}",
                    "95% CI": f"[{vr.ci_low:.4f}, {vr.ci_high:.4f}]",
                })
            st.dataframe(pd.DataFrame(vr_rows), use_container_width=True, hide_index=True)

        st.subheader("Economic Commentary")
        commentary = generate_tab1_commentary(
            moments=raw_moments, h=habit, utilisation=psi_util,
            price_stickiness=theta_p, wage_stickiness=theta_w, debt_feedback=phi_b,
            empirical_moments=empirical_moments, econometrics=econ_results,
        )
        st.markdown(commentary)


# TAB 2: Fiscal Exercises
with tab2:
    if not solver_ok:
        hint = likely_failure_hint(params, diag.get("flag", "no_solution"))
        st.error(f"Solver status: {diag.get('flag', 'no_solution')}. {hint}")

    st.subheader("40-Quarter Fiscal IRFs")
    col_shock, col_rule = st.columns(2)
    with col_shock:
        shock_choice = st.selectbox(
            "Shock",
            ["Gc Shock", "GI Shock", "Labor-Tax Cut", "Capital-Tax Cut"],
        )
    with col_rule:
        rule_choice = st.selectbox(
            "Financing Rule",
            ["Lump-Sum transfers", "Consumption Tax Hikes", "Labor Tax Hikes",
             "Capital Tax Hikes", "Government Spending Cuts"],
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
        st.error(f"Selected rule solve failed: {rule_diag.get('flag', 'no_solution')}. {hint}")
        st.stop()

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
    with col_im:
        st.metric("Impact Multiplier", f"{im:.3f}" if np.isfinite(im) else "N/A")
    with col_cm:
        st.metric("Cumulative Multiplier", f"{cm:.3f}" if np.isfinite(cm) else "N/A")
    with col_drag:
        st.metric("Fiscal Drag Horizon", f"Q{drag}" if drag is not None else "None (40Q)")

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
                       label=r"Drag $t^*=$" + f"{drag}")
            ax.legend(fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Log deviation")
    fig_irf.suptitle(f"{shock_choice} under {rule_choice}",
                     fontsize=13, fontweight="bold", y=1.01)
    fig_irf.tight_layout()
    st.pyplot(fig_irf, clear_figure=True)

    # Composite overlay
    st.subheader(r"Composite: $\hat{y}_t$ IRF across Financing Rules")
    rule_labels = ["Lump-Sum transfers", "Consumption Tax Hikes", "Labor Tax Hikes",
                   "Capital Tax Hikes", "Government Spending Cuts"]
    rule_colors = ["#2f80ed", "#27ae60", "#f2994a", "#eb5757", "#9b59b6"]

    fig_comp, ax_comp = plt.subplots(1, 1, figsize=(12, 5))
    cum_mults = {}
    skipped_rules = []
    for rlabel, rcolor in zip(rule_labels, rule_colors):
        canonical_rule = ui_label_to_canonical(rlabel)
        model_r, irf_r, aligned_r = simulate_fiscal_irf(
            solve_key, canonical_rule, shock_canonical, ("y_hat", "g_c_hat", "g_i_hat")
        )
        if irf_r is None:
            skipped_rules.append(rlabel)
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

    if skipped_rules:
        st.warning("Skipped failed rule solves: " + ", ".join(skipped_rules))
    ax_comp.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_comp.set_xlabel("Quarter")
    ax_comp.set_ylabel(r"$\hat{y}_t$ (log deviation)")
    ax_comp.set_title(f"Output IRF: {shock_choice} - All Financing Rules")
    ax_comp.legend(fontsize=9, loc="upper right")
    fig_comp.tight_layout()
    st.pyplot(fig_comp, clear_figure=True)

    st.markdown(r"**Cumulative Multipliers $\mathrm{CM}(H)$ by Financing Rule**")
    cm_rows = [{"Rule": k, "Cumulative Multiplier": f"{v:.4f}" if np.isfinite(v) else "N/A"}
               for k, v in cum_mults.items()]
    st.dataframe(pd.DataFrame(cm_rows), use_container_width=True, hide_index=True)

    st.subheader("Automated Policy Briefing")
    briefing = generate_tab2_briefing(
        shock_name=shock_choice, financing_rule=rule_choice,
        impact_mult=im, cumulative_mult=cm, drag_horizon=drag,
    )
    st.markdown(briefing)
