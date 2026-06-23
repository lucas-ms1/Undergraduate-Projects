import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import BASELINE_PARAMS, IRF_HORIZON, SIMULATION_HORIZON, SLIDER_BOUNDS
from dsge.model import solve_model_objects
from policy.multipliers import compute_multipliers
from policy.shocks import shock_vector_name
from simulation.irf import compute_irf
from simulation.moments import compute_moments
from simulation.simulate import simulate_paths
from utils.summaries import tab1_summary, tab2_summary


# -- helpers --

def load_css():
    css_path = pathlib.Path("assets/style.css")
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


@st.cache_data
def cached_solve(params_tuple):
    params = dict(params_tuple)
    return solve_model_objects(params)


@st.cache_data
def cached_simulate(A_tuple, B_tuple, horizon, seed):
    A = np.array(A_tuple)
    B = np.array(B_tuple)
    return simulate_paths(A, B, horizon=horizon, seed=seed)


# -- page config --

st.set_page_config(page_title="ECO 317 Fiscal Dashboard", layout="wide")
load_css()
st.title("AI-Assisted Fiscal Policy Dashboard")

# -- sidebar sliders --

params = dict(BASELINE_PARAMS)
with st.sidebar:
    st.header("Structural Parameters")
    params["habit"] = st.slider(
        "Habit (h)", *SLIDER_BOUNDS["habit"], value=float(params["habit"])
    )
    params["psi"] = st.slider(
        "Utilization Convexity (psi)", *SLIDER_BOUNDS["psi"], value=float(params["psi"])
    )
    params["theta_p"] = st.slider(
        "Calvo Price (theta_p)", *SLIDER_BOUNDS["theta_p"], value=float(params["theta_p"])
    )
    params["theta_w"] = st.slider(
        "Calvo Wage (theta_w)", *SLIDER_BOUNDS["theta_w"], value=float(params["theta_w"])
    )
    params["phi_b"] = st.slider(
        "Debt Feedback (phi_b)", *SLIDER_BOUNDS["phi_b"], value=float(params["phi_b"])
    )
    params["lambda_rot"] = st.slider(
        "Rule-of-Thumb Share (lambda)",
        *SLIDER_BOUNDS["lambda_rot"],
        value=float(params["lambda_rot"]),
    )
    params["rho_x"] = st.slider(
        "Shock Persistence (rho_x)", *SLIDER_BOUNDS["rho_x"], value=float(params["rho_x"])
    )
    params["sigma_x"] = st.slider(
        "Shock Volatility (sigma_x)",
        *SLIDER_BOUNDS["sigma_x"],
        value=float(params["sigma_x"]),
    )
    seed = st.number_input("Simulation Seed", min_value=0, value=42)

    st.markdown("---")
    dark_mode = st.toggle("Dark Mode", value=True)

# -- inject dark/light theme colours --
if dark_mode:
    st.markdown("""<style>
        .stApp { background: #0e1a2b; color: #f5f7fb; }
        [data-testid="stSidebar"] { background: #152238; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: #f5f7fb !important; }
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
        [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: #f5f7fb !important; }
        h1, h2, h3, h4, .stMarkdown, .stMarkdown p, .stMarkdown li { color: #f5f7fb !important; }
        button[data-baseweb="tab"] { color: #f5f7fb !important; }
        .stSelectbox label, .stNumberInput label { color: #f5f7fb !important; }
        .katex, .katex * { color: #f5f7fb !important; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        .stApp { background: #ffffff; color: #111111; }
        [data-testid="stSidebar"] { background: #f0f2f6; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: #111111 !important; }
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
        [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: #111111 !important; }
        h1, h2, h3, h4, .stMarkdown, .stMarkdown p, .stMarkdown li { color: #111111 !important; }
        button[data-baseweb="tab"] { color: #111111 !important; }
        .stSelectbox label, .stNumberInput label { color: #111111 !important; }
        .katex, .katex * { color: #111111 !important; }
    </style>""", unsafe_allow_html=True)

# -- determinacy warning --

if params["phi_pi"] <= 1.0:
    st.error("Determinacy warning: keep phi_pi > 1 to satisfy the Taylor principle.")

# -- solve & simulate --

model = cached_solve(tuple(sorted(params.items())))
diag = model["diagnostics"]

if not diag["solver_success"]:
    if diag["flag"] == "indeterminacy":
        st.error(
            f"Solver status: indeterminacy. The model has "
            f"{diag['explosive_roots']} explosive root(s) but needs "
            f"{diag['jump_count']} (one per jump variable). "
            f"Try adjusting theta_p, phi_b, or keeping phi_pi well above 1."
        )
    else:
        st.error(
            f"Solver status: {diag['flag']}. The model has too many "
            f"explosive roots ({diag['explosive_roots']}). Try lowering phi_pi "
            f"closer to 1.5 or raising theta_p."
        )

A = model["A_matrix"]
B_ = model["B_matrix"]
idx = model["variable_index"]
sidx = model["shock_index"]

# Make A and B hashable for st.cache_data
sim = cached_simulate(
    tuple(map(tuple, A)), tuple(map(tuple, B_)), SIMULATION_HORIZON, int(seed)
)
moments, frame = compute_moments(sim, idx)

# -- tabs --

tab_spec, tab_fit, tab_fiscal = st.tabs(
    ["Model Specification",
     "Model Fit (Unconditional Dynamics)",
     "Fiscal Exercises (Conditional Policy)"]
)


# =====================================================================
# TAB 0 - MODEL SPECIFICATION
# =====================================================================
with tab_spec:
    st.subheader("Model Overview")
    st.markdown(
        "This dashboard simulates a **medium-scale New-Keynesian DSGE model** "
        "inspired by Smets & Wouters (2007). The economy is populated by "
        "households that choose how much to consume and save, firms that set "
        "prices subject to Calvo-style nominal rigidities, and a government "
        "that finances spending through taxes and debt."
    )
    st.markdown(
        "The model captures several real-world frictions that shape how fiscal "
        "policy propagates through the economy:"
    )
    st.markdown(
        "**Habit formation** (*h*) makes households partly anchor their "
        "consumption to past levels, smoothing the response of aggregate "
        "demand.  **Price stickiness** (*theta_p*) and **wage stickiness** "
        "(*theta_w*) prevent instant market clearing, giving monetary and "
        "fiscal shocks real effects.  A fraction **lambda** of rule-of-thumb "
        "(non-Ricardian) households simply spend their after-tax income each "
        "period, amplifying the fiscal multiplier.  **Investment adjustment "
        "costs** (*psi*) slow capital reallocation, and a **debt-feedback "
        "rule** (*phi_b*) ensures long-run fiscal sustainability."
    )
    st.markdown(
        "The model is solved in state-space form using the **Blanchard-Kahn "
        "(QZ/Schur) decomposition**, which splits the system into "
        "predetermined (backward-looking) and jump (forward-looking) "
        "variables and finds the unique saddle-path equilibrium."
    )

    st.subheader("Endogenous Variables")
    var_df = pd.DataFrame(
        [
            ["y_hat", "Output gap",
             "Log-deviation of GDP from steady state. Predetermined (backward-looking)."],
            ["c_hat", "Consumption gap",
             "Log-deviation of aggregate consumption. Forward-looking via the Euler equation with habit persistence."],
            ["i_hat", "Investment gap",
             "Log-deviation of investment. Forward-looking via Tobin-Q / adjustment-cost dynamics."],
            ["b_hat", "Government debt",
             "Log-deviation of real government debt from steady state. Predetermined (budget constraint)."],
            ["pi_hat", "Inflation",
             "Log-deviation of price inflation. Forward-looking via the New-Keynesian Phillips curve."],
            ["l_hat", "Hours worked",
             "Log-deviation of labour supply. Predetermined (wage-setting block)."],
            ["r_hat", "Nominal interest rate",
             "Log-deviation of the policy rate. Predetermined (Taylor rule with smoothing)."],
        ],
        columns=["Code name", "Variable", "Description"],
    )
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    st.subheader("Structural Shocks")
    shock_df = pd.DataFrame(
        [
            ["tfp", "Total factor productivity shock"],
            ["g", "Government spending / demand shock"],
            ["monetary", "Monetary policy (interest-rate) shock"],
            ["tau_l", "Labour income tax shock"],
            ["tau_k", "Capital income tax shock"],
            ["risk", "Risk / labour-supply shock"],
        ],
        columns=["Code name", "Description"],
    )
    st.dataframe(shock_df, use_container_width=True, hide_index=True)

    st.subheader("Key Equations (Linearised)")
    st.latex(
        r"\textbf{Consumption Euler (with habit):}\quad "
        r"\hat{c}_t - h\,\hat{c}_{t-1} "
        r"= \mathbb{E}_t[\hat{c}_{t+1} - h\,\hat{c}_t] "
        r"- \frac{1-h}{\sigma}\bigl(\hat{r}_t - \mathbb{E}_t\hat{\pi}_{t+1}\bigr)"
    )
    st.latex(
        r"\textbf{NK Phillips Curve:}\quad "
        r"\hat{\pi}_t = \beta\,\mathbb{E}_t\hat{\pi}_{t+1} "
        r"+ \kappa_p\,\widehat{mc}_t, \qquad "
        r"\kappa_p = \frac{(1-\theta_p)(1-\beta\theta_p)}{\theta_p}"
    )
    st.latex(
        r"\textbf{Taylor Rule:}\quad "
        r"\hat{\imath}_t = \rho_i\,\hat{\imath}_{t-1} "
        r"+ (1-\rho_i)\bigl[\phi_\pi\,\hat{\pi}_t "
        r"+ \phi_y\,\hat{y}_t\bigr] + \varepsilon_{i,t}"
    )
    st.latex(
        r"\textbf{Government Budget:}\quad "
        r"\hat{B}_{t+1} = (1+r)\,\hat{B}_t + \hat{G}_t - \hat{T}_t"
    )
    st.latex(
        r"\textbf{State-Space Form:}\quad "
        r"\mathbb{E}_t[x_{t+1}] = A\,x_t + B\,\varepsilon_{t+1}, \qquad "
        r"x = [\hat{y},\;\hat{c},\;\hat{\imath},\;\hat{B},"
        r"\;\hat{\pi},\;\hat{\ell},\;\hat{r}]^\prime"
    )

    st.subheader("Blanchard-Kahn Diagnostics")
    if diag["solver_success"]:
        st.success(
            f"BK satisfied. {diag['explosive_roots']} explosive root(s) = "
            f"{diag['jump_count']} jump variable(s). "
            f"Unique saddle-path solution found."
        )
    else:
        st.error(
            f"BK failed - flag: {diag['flag']}. "
            f"Explosive roots: {diag['explosive_roots']}, "
            f"Jump variables: {diag['jump_count']}."
        )

    eig_rows = []
    for e in diag["eigenvalues"]:
        if abs(e.imag) > 1e-10:
            ev_str = f"{e.real:+.4f}{e.imag:+.4f}j"
        else:
            ev_str = f"{e.real:+.6f}"
        eig_rows.append({
            "Eigenvalue": ev_str,
            "|lambda|": f"{abs(e):.6f}",
            "Type": "Explosive" if abs(e) > 1 else "Stable",
        })
    st.dataframe(pd.DataFrame(eig_rows), use_container_width=True, hide_index=True)

    st.subheader("Current Calibration")
    col_par, col_ss = st.columns(2)
    with col_par:
        st.markdown("**Structural Parameters**")
        par_df = pd.DataFrame(
            [
                ["beta", f"{params['beta']:.3f}", "Discount factor"],
                ["sigma", f"{params['sigma']:.2f}", "CRRA coefficient"],
                ["alpha", f"{params['alpha']:.2f}", "Capital share"],
                ["delta", f"{params['delta']:.3f}", "Depreciation rate"],
                ["h", f"{params['habit']:.2f}", "Habit formation"],
                ["psi", f"{params['psi']:.2f}", "Utilization convexity"],
                ["theta_p", f"{params['theta_p']:.2f}", "Calvo price stickiness"],
                ["theta_w", f"{params['theta_w']:.2f}", "Calvo wage stickiness"],
                ["phi_b", f"{params['phi_b']:.3f}", "Debt-feedback coefficient"],
                ["phi_pi", f"{params['phi_pi']:.2f}", "Taylor rule: inflation"],
                ["phi_y", f"{params['phi_y']:.3f}", "Taylor rule: output"],
                ["rho_i", f"{params['rho_i']:.2f}", "Interest-rate smoothing"],
                ["lambda", f"{params['lambda_rot']:.2f}", "Rule-of-thumb share"],
            ],
            columns=["Parameter", "Value", "Description"],
        )
        st.dataframe(par_df, use_container_width=True, hide_index=True)

    with col_ss:
        st.markdown("**Steady-State Levels**")
        ss = model["steady_state"]
        ss_df = pd.DataFrame(
            [[k, f"{v:.4f}", desc] for k, v, desc in [
                ("Y", ss["Y"], "Output"),
                ("C", ss["C"], "Consumption"),
                ("I", ss["I"], "Investment"),
                ("G", ss["G"], "Government spending"),
                ("K", ss["K"], "Capital stock"),
                ("L", ss["L"], "Hours worked"),
                ("W", ss["W"], "Real wage"),
                ("Rk", ss["Rk"], "Rental rate of capital"),
                ("B", ss["B"], "Government debt"),
            ]],
            columns=["Variable", "Value", "Description"],
        )
        st.dataframe(ss_df, use_container_width=True, hide_index=True)


# =====================================================================
# TAB 1 - MODEL FIT (UNCONDITIONAL DYNAMICS)
# =====================================================================
with tab_fit:
    st.subheader(f"Business-Cycle Moments (T = {SIMULATION_HORIZON:,})")

    display_names = {
        "y_hat": "y_hat (Output)",
        "c_hat": "c_hat (Consumption)",
        "i_hat": "i_hat (Investment)",
        "l_hat": "l_hat (Hours)",
        "pi_hat": "pi_hat (Inflation)",
    }
    rows = []
    for key, label in display_names.items():
        rows.append({
            "Series": label,
            "Variance": f"{moments[f'var_{key}']:.6f}",
            "Corr(x, y)": f"{moments[f'corr_{key}_y']:.4f}",
            "AR(1)": f"{moments[f'acf1_{key}']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(tab1_summary(params, moments))

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(frame["y_hat"].to_numpy(), label="Output", color="#2f80ed")
    ax.plot(frame["c_hat"].to_numpy(), label="Consumption", color="#27ae60", alpha=0.8)
    ax.set_title("Simulated Output and Consumption")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    st.pyplot(fig, clear_figure=True)


# =====================================================================
# TAB 2 - FISCAL EXERCISES (CONDITIONAL POLICY)
# =====================================================================
with tab_fiscal:
    st.subheader("40-Quarter Fiscal IRFs")
    shock_choice = st.selectbox(
        "Shock", ["Gc Shock", "GI Shock", "Labor-Tax Cut", "Capital-Tax Cut"]
    )
    rule_choice = st.selectbox(
        "Financing Rule",
        [
            "Lump-Sum transfers",
            "Consumption Tax Hikes",
            "Labor Tax Hikes",
            "Capital Tax Hikes",
            "Government Spending Cuts",
        ],
    )

    shock_key = shock_vector_name(shock_choice)
    irf = compute_irf(
        A, B_, idx, sidx, shock_key, rule_choice,
        horizon=IRF_HORIZON, shock_size=1.0,
    )
    mult = compute_multipliers(
        irf, idx,
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
        ax.axhline(0, color="gray", linewidth=0.8)
        if mult["fiscal_drag_horizon"] is not None:
            ax.axvline(
                mult["fiscal_drag_horizon"],
                color="gray", linestyle="--", linewidth=1,
            )
        ax.set_title(label)
        ax.set_xlabel("Quarter")
        ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
