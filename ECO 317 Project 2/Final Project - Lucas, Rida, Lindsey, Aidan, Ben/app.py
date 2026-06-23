"""
ECO 317 Project 2 -- AI-Assisted Macroeconomic Modeling Dashboard
================================================================
Launch with:  streamlit run app.py

Solves three infinite-horizon stochastic macro models via Value Function
Iteration (VFI), simulates them, computes moments, and produces exact
non-linear forecasts -- all within an interactive Streamlit UI.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

from config import MODEL1_DEFAULTS, MODEL2_DEFAULTS, MODEL3_DEFAULTS, SIM_DEFAULTS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ECO 317 Macro Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------
THEMES = {
    "Dark": dict(
        bg="#0e1a2b", sidebar_bg="#12223d", text="#e8e8e8",
        heading="#ffffff", card_bg="#1a2d4a", card_text="#e8e8e8",
        divider="#2a3f5c", caption="#8899aa",
        plot_paper="#162338", plot_bg="#0e1a2b",
        plot_grid="rgba(255,255,255,0.1)", plot_text="#c9d6e3",
        line1="#5b9bd5", line2="#ed7d31", line3="#70ad47", line4="#ffc000",
        plotly_template="plotly_dark",
    ),
    "Light": dict(
        bg="#f5f7fa", sidebar_bg="#e8ecf1", text="#1a1a1a",
        heading="#111111", card_bg="#ffffff", card_text="#1a1a1a",
        divider="#cccccc", caption="#666666",
        plot_paper="#ffffff", plot_bg="#ffffff",
        plot_grid="rgba(0,0,0,0.1)", plot_text="#333333",
        line1="#2b5c8a", line2="#d35400", line3="#27ae60", line4="#8e44ad",
        plotly_template="plotly_white",
    ),
}

# ---------------------------------------------------------------------------
# Sidebar -- theme toggle & model selector
# ---------------------------------------------------------------------------
st.sidebar.title("ECO 317 Dashboard")
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)
TH = THEMES[theme_choice]

st.markdown(f"""<style>
:root {{
    --bg-color: {TH['bg']}; --sidebar-bg: {TH['sidebar_bg']};
    --text-color: {TH['text']}; --heading-color: {TH['heading']};
    --card-bg: {TH['card_bg']}; --card-text: {TH['card_text']};
    --divider-color: {TH['divider']}; --caption-color: {TH['caption']};
}}
</style>""", unsafe_allow_html=True)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly / download helpers
# ---------------------------------------------------------------------------
_FONT = "EB Garamond, Garamond, Georgia, serif"
_PLOTLY_CFG = {"displayModeBar": True, "scrollZoom": True}


def _layout(**kw):
    plot_font = dict(family=_FONT, color=TH["plot_text"])
    base = dict(
        template=TH["plotly_template"],
        paper_bgcolor=TH["plot_paper"], plot_bgcolor=TH["plot_bg"],
        font=plot_font,
        legend=dict(font=plot_font),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    base.update(kw)
    # Force title and axis labels/ticks to use theme text color (visible in light and dark)
    title_text = base.get("title", "")
    if isinstance(title_text, dict):
        title_text = title_text.get("text", "")
    base["title"] = dict(text=title_text or "", font=plot_font)
    xaxis = base.get("xaxis") or {}
    x_title = base.pop("xaxis_title", None)
    if x_title is not None:
        xaxis["title"] = dict(text=x_title, font=plot_font)
    else:
        existing = xaxis.get("title")
        cur_text = existing.get("text", "") if isinstance(existing, dict) else (existing or "")
        xaxis["title"] = dict(text=cur_text, font=plot_font)
    xaxis["tickfont"] = plot_font
    base["xaxis"] = xaxis
    yaxis = base.get("yaxis") or {}
    y_title = base.pop("yaxis_title", None)
    if y_title is not None:
        yaxis["title"] = dict(text=y_title, font=plot_font)
    else:
        existing = yaxis.get("title")
        cur_text = existing.get("text", "") if isinstance(existing, dict) else (existing or "")
        yaxis["title"] = dict(text=cur_text, font=plot_font)
    yaxis["tickfont"] = plot_font
    base["yaxis"] = yaxis
    return base


def _style_axes(fig):
    plot_font = dict(family=_FONT, color=TH["plot_text"])
    fig.update_xaxes(gridcolor=TH["plot_grid"], title_font=plot_font, tickfont=plot_font)
    fig.update_yaxes(gridcolor=TH["plot_grid"], title_font=plot_font, tickfont=plot_font)
    # Ensure main title font is visible (e.g. in light mode)
    fig.update_layout(title_font=plot_font)
    return fig


def _colors():
    return [TH["line1"], TH["line2"], TH["line3"], TH["line4"]]


def _download_btns(fig, data_dict, label, key_prefix):
    c1, c2, _ = st.columns([1, 1, 4])
    csv_bytes = pd.DataFrame(data_dict).to_csv(index=False).encode()
    c1.download_button(f"CSV: {label}", csv_bytes, f"{label}.csv",
                       "text/csv", key=f"csv_{key_prefix}")
    try:
        h = fig.layout.height or 700
        png = fig.to_image(format="png", width=1200, height=h, scale=2)
        c2.download_button(f"PNG: {label}", png, f"{label}.png",
                           "image/png", key=f"png_{key_prefix}")
    except Exception:
        c2.caption("Install *kaleido* for PNG export")


# ---------------------------------------------------------------------------
# Cached solver wrappers -- @st.cache_data prevents re-solving when only
# the forecast dropdown changes (rubric: "Wrap the solver function in
# Streamlit's @st.cache_data decorator so the app does not freeze.")
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Solving Model 1 via VFI...")
def _solve_model1(beta, sigma, r, y_L, y_H, p00, p11, a_min, a_max, n_a):
    from models.consumption_savings import solve
    params = dict(beta=beta, sigma=sigma, r=r,
                  y_vals=(y_L, y_H),
                  P=((p00, 1 - p00), (1 - p11, p11)),
                  a_min=a_min, a_max=a_max, n_a=n_a)
    return solve(params), params


@st.cache_data(show_spinner="Solving Model 2 via VFI...")
def _solve_model2(beta, sigma, alpha, delta, A, z_L, z_H, p00, p11,
                   k_min, k_max, n_k):
    from models.robinson_crusoe import solve
    params = dict(beta=beta, sigma=sigma, alpha=alpha, delta=delta, A=A,
                  z_vals=(z_L, z_H),
                  P=((p00, 1 - p00), (1 - p11, p11)),
                  k_min=k_min, k_max=k_max, n_k=n_k)
    return solve(params), params


@st.cache_data(show_spinner="Solving Model 3 via VFI...")
def _solve_model3(beta, sigma, psi, nu, r, w_L, w_H, p00, p11,
                   a_min, a_max, n_a, n_L, include_assets):
    from models.labor_supply import solve
    params = dict(beta=beta, sigma=sigma, psi=psi, nu=nu, r=r,
                  w_vals=(w_L, w_H),
                  P=((p00, 1 - p00), (1 - p11, p11)),
                  a_min=a_min, a_max=a_max, n_a=n_a,
                  n_L=n_L, include_assets=include_assets)
    return solve(params), params


# ---------------------------------------------------------------------------
# Sidebar -- model selector
# ---------------------------------------------------------------------------
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Model 1: Consumption-Savings",
     "Model 2: Robinson Crusoe",
     "Model 3: Labor Supply"],
)
MODEL_KEY = {"Model 1: Consumption-Savings": "model1",
             "Model 2: Robinson Crusoe": "model2",
             "Model 3: Labor Supply": "model3"}[model_choice]

# ---------------------------------------------------------------------------
# Sidebar -- shared parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Core Parameters")
beta = st.sidebar.slider("Discount factor (beta)", 0.80, 0.99, 0.95, 0.01)
sigma = st.sidebar.slider("Risk aversion (sigma)", 0.5, 5.0, 2.0, 0.1)

# ---------------------------------------------------------------------------
# Sidebar -- Markov transition matrix (all models share this structure)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Markov Transition Matrix")
p00 = st.sidebar.slider("P(Low | Low)", 0.50, 0.99, 0.90, 0.01)
p11 = st.sidebar.slider("P(High | High)", 0.50, 0.99, 0.90, 0.01)
st.sidebar.caption(
    f"P = [[{p00:.2f}, {1-p00:.2f}], [{1-p11:.2f}, {p11:.2f}]]"
)

# ---------------------------------------------------------------------------
# Sidebar -- model-specific parameters
# ---------------------------------------------------------------------------
if MODEL_KEY == "model1":
    st.sidebar.markdown("### Model 1 Parameters")
    r = st.sidebar.slider("Interest rate (r)", 0.00, 0.10, 0.03, 0.01)
    init_wealth = st.sidebar.slider("Initial wealth", 0.0, 15.0, 5.0, 0.5)
    n_a = st.sidebar.number_input("Grid size (n_a)", 40, 600, 200, 20)

elif MODEL_KEY == "model2":
    st.sidebar.markdown("### Model 2 Parameters")
    alpha = st.sidebar.slider("Capital share (alpha)", 0.20, 0.50, 0.36, 0.01)
    delta = st.sidebar.slider("Depreciation (delta)", 0.02, 0.20, 0.10, 0.01)
    n_k = st.sidebar.number_input("Grid size (n_k)", 40, 600, 200, 20)

else:
    st.sidebar.markdown("### Model 3 Parameters")
    r_m3 = st.sidebar.slider("Interest rate (r)", 0.00, 0.10, 0.03, 0.01)
    psi = st.sidebar.slider("Leisure weight (psi)", 0.1, 5.0, 1.0, 0.1)
    nu = st.sidebar.slider("Leisure curvature (nu)", 0.5, 5.0, 2.0, 0.1)
    include_assets = st.sidebar.checkbox("Include assets", value=True)
    if include_assets:
        init_wealth_m3 = st.sidebar.slider(
            "Initial wealth (Model 3)", 0.0, 15.0, 5.0, 0.5)
        n_a_m3 = st.sidebar.number_input("Grid size (n_a)", 40, 600, 150, 20)
    n_L = st.sidebar.number_input("Labor grid (n_L)", 20, 200, 60, 10)

# Simulation settings in sidebar
st.sidebar.markdown("### Simulation")
T_sim = st.sidebar.number_input("Periods", 100, 1000, SIM_DEFAULTS["T_sim"], 50)
seed = st.sidebar.number_input("Random seed", 0, 9999, SIM_DEFAULTS["seed"], 1)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════
st.title("ECO 317 -- Macroeconomic Modeling Dashboard")
st.markdown(f"**Active model:** {model_choice}")

# ---------------------------------------------------------------------------
# Model formulation display
# ---------------------------------------------------------------------------
if MODEL_KEY == "model1":
    st.subheader("Model Formulation")
    st.markdown("**Bellman equation** -- the household chooses next-period assets $a'$ to maximize:")
    st.latex(r"V(a,\, y) \;=\; \max_{a' \ge \underline{a}} \left\{ u(c) \;+\; \beta \sum_{y'} P(y'\!\mid y)\, V(a',\, y') \right\}")
    st.markdown("**Budget constraint** and **CRRA utility:**")
    st.latex(r"c = (1+r)\,a + y - a', \qquad u(c) = \frac{c^{1-\sigma}}{1-\sigma}")
    st.caption(
        f"Current parameters: $\\beta={beta:.2f}$, $\\sigma={sigma:.1f}$, $r={r:.2f}$"
    )

elif MODEL_KEY == "model2":
    st.subheader("Model Formulation")
    st.markdown("**Bellman equation** -- the planner chooses next-period capital $k'$ to maximize:")
    st.latex(r"V(k,\, z) \;=\; \max_{k' \ge 0} \left\{ u(c) \;+\; \beta \sum_{z'} P(z'\!\mid z)\, V(k',\, z') \right\}")
    st.markdown("**Budget constraint** and **CRRA utility:**")
    st.latex(r"c = z\,A\,k^{\alpha} + (1-\delta)\,k - k', \qquad u(c) = \frac{c^{1-\sigma}}{1-\sigma}")
    st.caption(
        f"Current parameters: $\\beta={beta:.2f}$, $\\sigma={sigma:.1f}$, "
        f"$\\alpha={alpha:.2f}$, $\\delta={delta:.2f}$"
    )

else:
    st.subheader("Model Formulation")
    if include_assets:
        st.markdown("**Bellman equation** -- the household jointly chooses savings $a'$ and labor $L$ to maximize:")
        st.latex(r"V(a,\, w) \;=\; \max_{a' \ge \underline{a},\; L \in [0,1]} \left\{ u(c) + \psi\,\frac{(1-L)^{1-\nu}}{1-\nu} \;+\; \beta \sum_{w'} P(w'\!\mid w)\, V(a',\, w') \right\}")
        st.markdown("**Budget constraint:**")
        st.latex(r"c = (1+r)\,a + w\,L - a', \qquad u(c) = \frac{c^{1-\sigma}}{1-\sigma}")
    else:
        st.markdown("**Bellman equation** -- the household chooses labor $L$ to maximize (no savings):")
        st.latex(r"V(w) \;=\; \max_{L \in [0,1]} \left\{ u(w\,L) + \psi\,\frac{(1-L)^{1-\nu}}{1-\nu} \;+\; \beta \sum_{w'} P(w'\!\mid w)\, V(w') \right\}")
    st.caption(
        f"Current parameters: $\\beta={beta:.2f}$, $\\sigma={sigma:.1f}$, "
        f"$\\psi={psi:.1f}$, $\\nu={nu:.1f}$, $r={r_m3:.2f}$"
    )

# ---------------------------------------------------------------------------
# Solve the selected model (cached -- sliders trigger re-solve only when
# the relevant parameters change)
# ---------------------------------------------------------------------------
if MODEL_KEY == "model1":
    d = MODEL1_DEFAULTS
    result, params = _solve_model1(
        beta, sigma, r,
        d["y_vals"][0], d["y_vals"][1],
        p00, p11, d["a_min"], d["a_max"], n_a)
    shock_vals = np.asarray(params["y_vals"])
    P = np.asarray(params["P"])
    grid = result["grid"]
    grid_label = "Assets"
    shock_labels = ["Low income", "High income"]
    sim_model_name = "model1"
    init_state = float(init_wealth)
    income_key = "y"

elif MODEL_KEY == "model2":
    d = MODEL2_DEFAULTS
    result, params = _solve_model2(
        beta, sigma, alpha, delta, d["A"],
        d["z_vals"][0], d["z_vals"][1],
        p00, p11, d["k_min"], d["k_max"], n_k)
    shock_vals = np.asarray(params["z_vals"])
    P = np.asarray(params["P"])
    grid = result["grid"]
    grid_label = "Capital"
    shock_labels = ["Low TFP", "High TFP"]
    sim_model_name = "model2"
    init_state = float(np.median(grid))
    income_key = "y"

else:
    d = MODEL3_DEFAULTS
    _inc = include_assets
    _n_a = n_a_m3 if _inc else d["n_a"]
    _iw = init_wealth_m3 if _inc else 0.0
    result, params = _solve_model3(
        beta, sigma, psi, nu, r_m3,
        d["w_vals"][0], d["w_vals"][1],
        p00, p11, d["a_min"], d["a_max"], _n_a, n_L, _inc)
    shock_vals = np.asarray(params["w_vals"])
    P = np.asarray(params["P"])
    shock_labels = ["Low wage", "High wage"]
    income_key = "earnings"
    if _inc:
        sim_model_name = "model3"
        grid = result["grid"]["a_grid"]
        grid_label = "Assets"
        init_state = float(_iw)
    else:
        sim_model_name = "model3_labor_only"
        grid = None
        grid_label = None
        init_state = 0.0

# Show convergence diagnostics
diag = result["diagnostics"]
st.success(
    f"VFI converged: {diag['converged']}  |  "
    f"Iterations: {diag['iterations']}  |  "
    f"Final sup-norm error: {diag['final_error']:.2e}"
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Value Function & Policy Plots
# ═══════════════════════════════════════════════════════════════════════════
st.header("1. Solved Value & Policy Functions")
colors = _colors()

if MODEL_KEY in ("model1", "model2") or (MODEL_KEY == "model3" and grid is not None):
    plot_grid = grid
    col1, col2 = st.columns(2)

    # Value function
    fig_v = go.Figure(layout=_layout(
        title="Value Function",
        xaxis_title=grid_label, yaxis_title="V(state, shock)"))
    for s in range(2):
        fig_v.add_trace(go.Scatter(
            x=plot_grid, y=result["value_function"][:, s],
            mode="lines", name=shock_labels[s],
            line=dict(width=1.5, color=colors[s])))
    _style_axes(fig_v)
    col1.plotly_chart(fig_v, width="stretch", config=_PLOTLY_CFG)

    # Consumption policy
    fig_p = go.Figure(layout=_layout(
        title="Consumption Policy  c*(state, shock)",
        xaxis_title=grid_label, yaxis_title="Consumption"))
    for s in range(2):
        fig_p.add_trace(go.Scatter(
            x=plot_grid, y=result["c_policy"][:, s],
            mode="lines", name=shock_labels[s],
            line=dict(width=1.5, color=colors[s])))
    _style_axes(fig_p)
    col2.plotly_chart(fig_p, width="stretch", config=_PLOTLY_CFG)

    sec1_data = {grid_label: plot_grid}
    for s in range(2):
        sec1_data[f"V({shock_labels[s]})"] = result["value_function"][:, s]
        sec1_data[f"c*({shock_labels[s]})"] = result["c_policy"][:, s]

    if MODEL_KEY == "model3":
        col3, col4 = st.columns(2)

        fig_s = go.Figure(layout=_layout(
            title="Savings Policy",
            xaxis_title="Assets", yaxis_title="a' (next-period assets)"))
        for s in range(2):
            fig_s.add_trace(go.Scatter(
                x=plot_grid,
                y=result["policy_levels"]["savings"][:, s],
                mode="lines", name=shock_labels[s],
                line=dict(width=1.5, color=colors[s])))
        _style_axes(fig_s)
        col3.plotly_chart(fig_s, width="stretch", config=_PLOTLY_CFG)

        fig_l = go.Figure(layout=_layout(
            title="Labor Policy",
            xaxis_title="Assets", yaxis_title="L* (labor)"))
        for s in range(2):
            fig_l.add_trace(go.Scatter(
                x=plot_grid,
                y=result["policy_levels"]["labor"][:, s],
                mode="lines", name=shock_labels[s],
                line=dict(width=1.5, color=colors[s])))
        _style_axes(fig_l)
        col4.plotly_chart(fig_l, width="stretch", config=_PLOTLY_CFG)

        for s in range(2):
            sec1_data[f"a'({shock_labels[s]})"] = \
                result["policy_levels"]["savings"][:, s]
            sec1_data[f"L*({shock_labels[s]})"] = \
                result["policy_levels"]["labor"][:, s]

    _download_btns(fig_v, sec1_data, "value_policy", "sec1")

elif sim_model_name == "model3_labor_only":
    st.subheader("Labor-Only Results (no asset dimension)")
    vf = result["value_function"]
    labor = result["policy_levels"]["labor"]
    c_pol = result["c_policy"]
    col1, col2 = st.columns(2)
    col1.metric("V(Low wage)", f"{vf[0]:.4f}")
    col1.metric("V(High wage)", f"{vf[1]:.4f}")
    col2.metric("L*(Low wage)", f"{labor[0]:.4f}")
    col2.metric("L*(High wage)", f"{labor[1]:.4f}")
    col2.metric("c*(Low wage)", f"{c_pol[0]:.4f}")
    col2.metric("c*(High wage)", f"{c_pol[1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Stochastic Simulation
# ═══════════════════════════════════════════════════════════════════════════
st.header("2. Stochastic Simulation")
from simulation.simulate import simulate_model

sim = simulate_model(
    result, shock_vals, P, init_state,
    T_sim=T_sim, seed=seed,
    model_name=sim_model_name, model_params=params,
)

series_keys = [k for k in sim if k != "shock_idx"]
n_plots = len(series_keys)

fig_sim = make_subplots(
    rows=n_plots, cols=1, shared_xaxes=True,
    subplot_titles=[f"Simulated {k.capitalize()}" for k in series_keys])
for i, key in enumerate(series_keys, 1):
    fig_sim.add_trace(
        go.Scatter(y=sim[key], mode="lines",
                   line=dict(width=0.8, color=colors[0]),
                   name=key.capitalize(), showlegend=False),
        row=i, col=1)
    fig_sim.update_yaxes(title_text=key.capitalize(), row=i, col=1)
fig_sim.update_xaxes(title_text="Period", row=n_plots, col=1)
fig_sim.update_layout(**_layout(
    title=f"Simulated Paths ({T_sim} periods)",
    height=280 * n_plots))
_style_axes(fig_sim)
st.plotly_chart(fig_sim, width="stretch", config=_PLOTLY_CFG)

_download_btns(fig_sim, {k: sim[k] for k in series_keys},
               "simulation", "sec2")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Summary Statistics (Moments)
# ═══════════════════════════════════════════════════════════════════════════
st.header("3. Summary Statistics")
from simulation.moments import compute_moments

moments = compute_moments(
    {k: v for k, v in sim.items() if k != "shock_idx"},
    income_key=income_key,
)
df_moments = pd.DataFrame(moments).T.round(4)
df_moments.index = [idx.capitalize() for idx in df_moments.index]
df_moments.columns = ["Mean", "Variance", "Autocorrelation (lag-1)",
                       f"Corr with {income_key}"]
st.dataframe(df_moments, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: AI "Intelligent" Summaries
# ═══════════════════════════════════════════════════════════════════════════
st.header("4. Economic Analysis")

if MODEL_KEY == "model1":
    m_c = moments["c"]
    m_y = moments["y"]
    m_a = moments.get("a", {})
    var_ratio = m_c["variance"] / m_y["variance"] if m_y["variance"] > 0 else float("nan")

    st.subheader("Technical")
    st.markdown(f"""
Over the {T_sim}-period simulation the key moments are:

| Variable | Mean | Variance | Autocorr (lag-1) | Corr w/ income |
|----------|------|----------|-----------------|----------------|
| Consumption | {m_c['mean']:.3f} | {m_c['variance']:.4f} | {m_c['autocorrelation']:.3f} | {m_c['corr_with_income']:.3f} |
| Income | {m_y['mean']:.3f} | {m_y['variance']:.4f} | {m_y['autocorrelation']:.3f} | 1.000 |

The consumption-to-income variance ratio is $\\text{{Var}}(c)/\\text{{Var}}(y) = {var_ratio:.3f}$,
{'which is well below one, indicating significant consumption smoothing.' if var_ratio < 0.5 else 'indicating partial but incomplete consumption smoothing.' if var_ratio < 0.9 else 'close to unity, indicating near hand-to-mouth behavior.'}
The Euler equation for this model is $u'(c_t) = \\beta (1+r) \\, \\mathbb{{E}}_t \\left[ u'(c_{{t+1}}) \\right]$.
With $\\beta = {beta:.2f}$, $\\sigma = {sigma:.1f}$, and $r = {r:.2f}$, the gross effective
discount rate is $\\beta(1+r) = {beta*(1+r):.3f}$.
{'Since this exceeds 1, the household has a motive to accumulate assets on average, generating a strong savings buffer.' if beta*(1+r) > 1.0 else 'Since this is below 1, the household is effectively impatient and will tend to run down assets toward the borrowing constraint.' if beta*(1+r) < 0.99 else 'This is close to unity, so the household is roughly indifferent between consuming today and saving for tomorrow in expectation.'}
""")

    st.subheader("Intuition")
    st.markdown(f"""
Imagine a household that earns an income that randomly switches between a low state and a high state each period.
The household can save or borrow (up to a limit) at interest rate $r = {r:.2f}$ per period.
Its goal is to keep its consumption -- how much it spends on goods -- as smooth and stable as possible over time,
because large swings in spending are painful (a big dinner followed by going hungry is worse than two moderate meals).

The **variance ratio of {var_ratio:.3f}** tells us how well the household succeeds at this.
{'A ratio well below 1 means consumption bounces around far less than income does. The household builds up a savings buffer during good times and draws it down during bad times, successfully shielding its day-to-day spending from the ups and downs of its paycheck. This is the core prediction of the Permanent Income Hypothesis: rational, forward-looking people plan ahead and use savings as a shock absorber.' if var_ratio < 0.5 else 'A ratio near but below 1 means the household partially smooths its spending, but quite a bit of income volatility still bleeds through. This can happen when the household is relatively impatient (low $\\beta$), when interest rates are low, or when the borrowing limit is tight -- all of which limit the ability to self-insure.' if var_ratio < 0.9 else 'A ratio near 1 means the household is essentially living paycheck to paycheck. Almost every fluctuation in income shows up as a fluctuation in consumption. This can happen when the household is very impatient, when borrowing constraints are tight, or when the interest rate is very low.'}

{'The high autocorrelation of consumption (' + f"{m_c['autocorrelation']:.3f}" + ') means spending this period is a strong predictor of spending next period -- the household adjusts its lifestyle gradually rather than in sudden jumps, which is a hallmark of an effective savings buffer.' if m_c['autocorrelation'] > 0.8 else 'The moderate autocorrelation of consumption (' + f"{m_c['autocorrelation']:.3f}" + ') tells us that spending is somewhat persistent but still responds noticeably to new income shocks each period.' if m_c['autocorrelation'] > 0.5 else 'The low autocorrelation of consumption (' + f"{m_c['autocorrelation']:.3f}" + ') suggests spending reacts quickly to each new income draw, without much of a smoothing buffer carrying over from previous periods.'}

In short, {'this household is a disciplined planner who saves patiently and rides out income swings with barely a ripple in its consumption.' if var_ratio < 0.5 and beta > 0.93 else 'this household makes a meaningful effort to smooth its spending, but faces constraints that prevent perfect insurance against income risk.' if var_ratio < 0.9 else 'this household lacks the patience or means to smooth effectively and largely consumes what it earns each period.'}
""")

elif MODEL_KEY == "model2":
    m_c = moments["c"]
    m_y = moments["y"]
    m_k = moments["k"]
    m_inv = moments["investment"]
    k_ss = ((beta * alpha * d["A"]) / (1 - beta * (1 - delta))) ** (1 / (1 - alpha))
    inv_share = m_inv["mean"] / m_y["mean"] if m_y["mean"] > 0 else float("nan")

    st.subheader("Technical")
    st.markdown(f"""
Over the {T_sim}-period simulation the key moments are:

| Variable | Mean | Variance | Autocorr (lag-1) | Corr w/ output |
|----------|------|----------|-----------------|----------------|
| Output | {m_y['mean']:.3f} | {m_y['variance']:.4f} | {m_y['autocorrelation']:.3f} | 1.000 |
| Consumption | {m_c['mean']:.3f} | {m_c['variance']:.4f} | {m_c['autocorrelation']:.3f} | {m_c['corr_with_income']:.3f} |
| Capital | {m_k['mean']:.3f} | {m_k['variance']:.4f} | {m_k['autocorrelation']:.3f} | {m_k['corr_with_income']:.3f} |
| Investment | {m_inv['mean']:.3f} | {m_inv['variance']:.4f} | {m_inv['autocorrelation']:.3f} | {m_inv['corr_with_income']:.3f} |

The deterministic steady-state capital stock is $k^* = {k_ss:.3f}$ (from $\\alpha A k^{{\\alpha-1}} = 1/\\beta - (1-\\delta)$).
The mean simulated capital of **{m_k['mean']:.3f}** {'is close to' if abs(m_k['mean'] - k_ss) / k_ss < 0.1 else 'deviates from'} this level due to precautionary motives and Jensen's inequality under stochastic TFP.
The investment-to-output ratio is **{inv_share:.3f}**, and capital autocorrelation is very high (**{m_k['autocorrelation']:.3f}**) because the law of motion $k' = z A k^\\alpha + (1-\\delta)k - c$ makes capital inherently persistent.
With $\\alpha = {alpha:.2f}$ and $\\delta = {delta:.2f}$, {'the economy has a high capital share, amplifying the role of investment in output.' if alpha > 0.40 else 'the calibration is near standard RBC values (capital share ~1/3).' if 0.30 <= alpha <= 0.40 else 'the low capital share means labor (absent here) would normally dominate.'}
""")

    st.subheader("Intuition")
    st.markdown(f"""
Think of a Robinson Crusoe economy: a single person stranded on an island who must decide each period how much of
their harvest to eat now versus how much to set aside as seed corn (investment) that will grow the next harvest.
Productivity fluctuates randomly -- some seasons the soil is fertile (high TFP), others it is poor (low TFP).

The key tension is **eating today versus planting for tomorrow**. Every unit of output consumed now is a unit that
cannot grow into more output next period. The household must constantly weigh the immediate benefit of eating
against the future benefit of a larger capital stock.

The mean investment share of **{inv_share:.3f}** tells us that roughly **{inv_share*100:.0f}%** of total output is being
reinvested each period.
{'This relatively high reinvestment rate means the household is forward-looking and willing to sacrifice current consumption to build productive capacity.' if inv_share > 0.25 else 'This moderate reinvestment rate reflects a balance between current enjoyment and future productivity.' if inv_share > 0.15 else 'This low reinvestment rate means the household is consuming most of what it produces, leaving little for capital accumulation.'}

Capital is extremely persistent (autocorrelation of **{m_k['autocorrelation']:.3f}**) because the capital stock
is like a large reservoir -- it changes slowly. Building it up takes many periods of saving, and it depreciates
only gradually at rate $\\delta = {delta:.2f}$ per period. This means that even temporary productivity shocks
have long-lasting effects on the economy: a good harvest leads to more investment, which raises the capital stock,
which raises future output, which allows more investment, and so on.

{'The high correlation between consumption and output (' + f"{m_c['corr_with_income']:.3f}" + ') means the household has limited ability to decouple its eating from its harvesting -- when times are good it eats more, when times are bad it eats less. Capital adjustment provides some buffer, but not enough to fully insulate consumption from production shocks.' if m_c['corr_with_income'] > 0.8 else 'The moderate consumption-output correlation (' + f"{m_c['corr_with_income']:.3f}" + ') shows the household has some ability to use capital adjustment to smooth consumption relative to output, absorbing part of the production shocks through changes in investment.' if m_c['corr_with_income'] > 0.5 else 'The relatively low consumption-output correlation (' + f"{m_c['corr_with_income']:.3f}" + ') suggests the household successfully uses investment as a buffer, partially decoupling its spending from output fluctuations.'}

{'In this economy, patience pays off: the high $\\beta = ' + f"{beta:.2f}" + '$ means the household genuinely cares about the future, so it saves diligently and maintains a healthy capital stock.' if beta > 0.93 else 'With a moderate discount factor of $\\beta = ' + f"{beta:.2f}" + '$, the household balances present and future reasonably, maintaining a workable capital stock without extreme frugality.' if beta > 0.90 else 'The relatively low $\\beta = ' + f"{beta:.2f}" + '$ means the household is impatient and heavily discounts the future, leading to lower savings and a smaller capital stock.'}
""")

else:
    m_c = moments["c"]
    m_e = moments.get("earnings", {})
    m_l = moments.get("labor", {})
    mean_L = m_l.get("mean", 0)
    mean_leisure = 1 - mean_L

    st.subheader("Technical")
    st.markdown(f"""
Over the {T_sim}-period simulation the key moments are:

| Variable | Mean | Variance | Autocorr (lag-1) |
|----------|------|----------|-----------------|
| Labor | {mean_L:.3f} | {m_l.get('variance', 0):.4f} | {m_l.get('autocorrelation', 0):.3f} |
| Consumption | {m_c['mean']:.3f} | {m_c['variance']:.4f} | {m_c['autocorrelation']:.3f} |
| Earnings | {m_e.get('mean', 0):.3f} | {m_e.get('variance', 0):.4f} | {m_e.get('autocorrelation', 0):.3f} |

The intra-temporal optimality condition (FOC) equates the marginal value of an extra hour worked
to the marginal cost in foregone leisure:
""")
    st.latex(r"w \, u'(c) = \psi \, v'(1 - L)")
    st.markdown(f"""
With $\\psi = {psi:.1f}$ and $\\nu = {nu:.1f}$, the Frisch elasticity of labor supply
is approximately $1/\\nu = {1/nu:.2f}$.
{'This low elasticity means labor supply is quite rigid -- hours worked respond little to wage changes.' if nu > 2.0 else 'This moderate elasticity means the household adjusts hours meaningfully but not drastically in response to wage changes.' if nu > 1.0 else 'This high elasticity means hours worked are very responsive to wage changes -- small wage increases lead to noticeably more work.'}
Mean labor is **{mean_L:.3f}** out of a unit time endowment, leaving mean leisure of **{mean_leisure:.3f}**.
""")

    st.subheader("Intuition")
    st.markdown(f"""
This model captures a fundamental daily decision: **how much to work versus how much to relax.** The household
has a fixed amount of time (normalized to 1) and must split it between labor (which earns wages and funds
consumption) and leisure (which provides direct enjoyment).

The wage randomly switches between a low state and a high state, mimicking the reality that job opportunities
and pay rates fluctuate over time. Each period, the household observes its current wage and decides how many
hours to work.

**The core trade-off** is straightforward: working more means more income and therefore more consumption (food,
housing, goods), but it also means less free time for rest, hobbies, and family. The household weighs these
two benefits against each other.

The parameter $\\psi = {psi:.1f}$ controls **how much the household values its leisure time**.
{'With this high value, the household places great importance on its free time and will resist working long hours even when wages are attractive. Think of someone who highly values work-life balance.' if psi > 2.0 else 'At this moderate value, the household has a balanced view -- it enjoys leisure but is willing to work reasonable hours for good pay. This is a typical calibration for a household that neither overworks nor underworks.' if psi > 0.8 else 'With this low value, the household does not particularly mind working long hours. Leisure is nice but not essential, so the household tends to work more and earn more.'}

The parameter $\\nu = {nu:.1f}$ controls **how rapidly the household tires of additional work**
(technically, the curvature of the leisure utility function).
{'A high $\\nu$ means the marginal value of each additional hour of leisure rises steeply -- the household strongly resists giving up its last few hours of free time. Hours worked respond sluggishly to wage changes because the pain of losing scarce leisure time outweighs the gain from higher earnings.' if nu > 2.0 else 'A moderate $\\nu$ means the household experiences diminishing returns to leisure at a steady pace. It will adjust its hours meaningfully when wages change, but not dramatically -- a sensible middle ground.' if nu > 1.0 else 'A low $\\nu$ means leisure utility is nearly linear, so the household can easily substitute between work and rest without strong discomfort. This makes labor supply highly responsive to wage changes.'}

With the current parameters the household works an average of **{mean_L:.1%} of its available time**
and spends the remaining **{mean_leisure:.1%}** on leisure.
{'This is a heavy workload, suggesting wages are attractive relative to the cost of lost leisure, or the household does not value free time very highly.' if mean_L > 0.5 else 'This is a moderate workload, reflecting a balanced trade-off between the benefits of extra income and the enjoyment of free time.' if mean_L > 0.3 else 'This is a light workload -- the household strongly prefers leisure, or wages are too low to justify long hours.'}

{'When wages rise, the household works more -- the higher pay makes each hour of work more rewarding, pulling hours away from leisure. Economists call this the **substitution effect** dominating: the opportunity cost of relaxing (forgoing high wages) outweighs the benefit of time off.' if m_l.get('corr_with_income', 0) > 0 else 'Interestingly, when wages rise the household may not increase hours much (or may even reduce them) because the extra pay per hour means it can afford the same consumption with fewer hours. This is the **income effect** at work: the household is already earning enough and prefers to enjoy more free time rather than chase additional income.'}
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Exact Non-Linear Forecasting
# ═══════════════════════════════════════════════════════════════════════════
st.header("5. Exact Non-Linear Forecast")
st.markdown(
    "Select a shock path for the next periods. The forecast uses the **exact "
    "solved policy functions** from VFI -- no trend fitting."
)

from simulation.forecast import forecast_model

fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    shock_1 = st.selectbox("Period 1 shock", ["Low", "High"], index=1)
with fcol2:
    shock_2 = st.selectbox("Period 2 shock", ["Low", "High"], index=1)
with fcol3:
    shock_3 = st.selectbox("Period 3 shock", ["Low", "High"], index=0)

shock_map = {"Low": 0, "High": 1}
user_shocks = [shock_map[shock_1], shock_map[shock_2], shock_map[shock_3]]

T_fcast = st.slider("Forecast horizon (periods after chosen shocks)",
                     3, 30, SIM_DEFAULTS["T_fcast"], 1)

full_shock_path = user_shocks + [user_shocks[-1]] * (T_fcast - len(user_shocks))

fcast_state = init_state
if sim_model_name == "model3_labor_only":
    fcast_state = 0.0

fcast = forecast_model(
    result, shock_vals, full_shock_path, fcast_state,
    model_name=sim_model_name, model_params=params,
)

fcast_keys = [k for k in fcast if k != "shock_idx"]
n_fp = len(fcast_keys)

fig_fc = make_subplots(
    rows=n_fp, cols=1, shared_xaxes=True,
    subplot_titles=[f"Forecast: {k.capitalize()}" for k in fcast_keys])
for i, key in enumerate(fcast_keys, 1):
    fig_fc.add_trace(
        go.Scatter(y=fcast[key], mode="lines+markers",
                   line=dict(width=1.5, color=TH["line2"]),
                   marker=dict(size=5),
                   name=key.capitalize(), showlegend=False),
        row=i, col=1)
    fig_fc.update_yaxes(title_text=key.capitalize(), row=i, col=1)

n_fcast_pts = len(fcast[fcast_keys[0]])
for t, s_idx in enumerate(full_shock_path[:n_fcast_pts]):
    clr = "rgba(255,0,0,0.08)" if s_idx == 0 else "rgba(0,128,0,0.08)"
    for r in range(1, n_fp + 1):
        ax_suffix = "" if r == 1 else str(r)
        fig_fc.add_shape(
            type="rect", x0=t - 0.4, x1=t + 0.4, y0=0, y1=1,
            xref=f"x{ax_suffix}", yref=f"y{ax_suffix} domain",
            fillcolor=clr, line_width=0, layer="below")

fig_fc.update_xaxes(title_text="Forecast Period", row=n_fp, col=1)

_shock_labels_list = [shock_1, shock_2, shock_3] + \
    [("Low" if user_shocks[-1] == 0 else "High")] * (T_fcast - 3)
_colored_shocks = []
for _sl in _shock_labels_list:
    if _sl == "Low":
        _colored_shocks.append('<span style="color:#e74c3c;font-weight:600">Low</span>')
    else:
        _colored_shocks.append('<span style="color:#27ae60;font-weight:600">High</span>')
shock_str_colored = " &#8594; ".join(_colored_shocks)

st.markdown(
    f'<p style="text-align:center;font-size:0.95rem;margin-bottom:0">'
    f'<b>Shock path:</b> {shock_str_colored}</p>',
    unsafe_allow_html=True,
)

fig_fc.update_layout(**_layout(
    title="Forecast Path",
    height=280 * n_fp))
_style_axes(fig_fc)
st.plotly_chart(fig_fc, width="stretch", config=_PLOTLY_CFG)

_download_btns(fig_fc, {k: fcast[k] for k in fcast_keys},
               "forecast", "sec5")

st.markdown(f"""
**Forecast interpretation:** Starting from {'assets' if MODEL_KEY != 'model2' else 'capital'} = {fcast_state:.2f},
the first three shock realisations are **{shock_1}, {shock_2}, {shock_3}**.
{'The forecast then extends the final shock state for the remaining periods.' if T_fcast > 3 else ''}
Because the forecast uses the exact VFI policy functions, these paths are consistent
with the household's optimal decision rules -- not extrapolations.
""")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Static Intuition & Comparative Statics
# ═══════════════════════════════════════════════════════════════════════════
st.header("6. Static Intuition & Comparative Statics")

if MODEL_KEY == "model1":
    st.subheader("Consumption vs. Savings Policy Across Shock States")

    fig_cs = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Savings Policy vs. 45-degree Line",
                        "MPC (numerical gradient of consumption policy)"])
    for s in range(2):
        fig_cs.add_trace(
            go.Scatter(x=grid, y=result["policy_levels"][:, s],
                       mode="lines", name=shock_labels[s],
                       line=dict(width=1.5, color=colors[s])),
            row=1, col=1)
    fig_cs.add_trace(
        go.Scatter(x=grid, y=grid, mode="lines", name="45-degree line",
                   line=dict(width=0.8, color="gray", dash="dash")),
        row=1, col=1)

    c_lo = result["c_policy"][:, 0]
    c_hi = result["c_policy"][:, 1]
    mpc_lo = np.gradient(c_lo, grid)
    mpc_hi = np.gradient(c_hi, grid)
    fig_cs.add_trace(
        go.Scatter(x=grid, y=mpc_lo, mode="lines",
                   name=f"MPC ({shock_labels[0]})",
                   line=dict(width=1.5, color=colors[0])),
        row=1, col=2)
    fig_cs.add_trace(
        go.Scatter(x=grid, y=mpc_hi, mode="lines",
                   name=f"MPC ({shock_labels[1]})",
                   line=dict(width=1.5, color=colors[1])),
        row=1, col=2)

    fig_cs.update_xaxes(title_text="Current assets (a)", row=1, col=1)
    fig_cs.update_yaxes(title_text="Next-period assets (a')", row=1, col=1)
    fig_cs.update_xaxes(title_text="Current assets (a)", row=1, col=2)
    fig_cs.update_yaxes(title_text="Marginal Propensity to Consume",
                        row=1, col=2)
    fig_cs.update_layout(**_layout(height=500))
    _style_axes(fig_cs)
    st.plotly_chart(fig_cs, width="stretch", config=_PLOTLY_CFG)

    _download_btns(fig_cs,
                   {"Assets": grid,
                    f"a'({shock_labels[0]})": result["policy_levels"][:, 0],
                    f"a'({shock_labels[1]})": result["policy_levels"][:, 1],
                    f"MPC({shock_labels[0]})": mpc_lo,
                    f"MPC({shock_labels[1]})": mpc_hi},
                   "comparative_statics", "sec6")

    st.markdown(f"""
**Intuition:** The savings policy lies below the 45-degree line at high wealth levels,
meaning the household dis-saves when wealthy. The MPC declines with wealth --
wealthier households consume a smaller fraction of additional income, consistent with
the concavity of the value function under CRRA preferences with $\\sigma = {sigma:.1f}$.
    """)

elif MODEL_KEY == "model2":
    st.subheader("Steady-State Capital vs. Patience (beta)")
    beta_range = np.linspace(0.80, 0.99, 30)
    ss_k = [((b * alpha * d["A"]) / (1 - b * (1 - delta))) ** (1 / (1 - alpha))
            for b in beta_range]

    fig_ss = go.Figure(layout=_layout(
        title="Steady-State Capital vs. Patience",
        xaxis_title="Discount factor (beta)",
        yaxis_title="Steady-state capital (k*)"))
    fig_ss.add_trace(go.Scatter(
        x=beta_range, y=ss_k, mode="lines",
        line=dict(width=2, color=colors[0]), name="k*(beta)"))
    fig_ss.add_vline(x=beta, line_dash="dash", line_color="red",
                     annotation_text=f"Current beta = {beta:.2f}")
    _style_axes(fig_ss)
    st.plotly_chart(fig_ss, width="stretch", config=_PLOTLY_CFG)

    _download_btns(fig_ss,
                   {"beta": beta_range, "k_star": ss_k},
                   "steady_state", "sec6")

    k_ss_cur = ((beta * alpha * d["A"]) / (1 - beta * (1 - delta))) ** (1 / (1 - alpha))
    st.markdown(f"""
**Intuition:** More patient households (higher $\\beta$) accumulate more capital in the
long run. At the current $\\beta = {beta:.2f}$, the deterministic steady-state capital is
approximately **{k_ss_cur:.2f}**. This is the level where the marginal product of
capital equals the effective discount rate:
""")
    st.latex(r"\alpha \, A \, k^{\alpha - 1} = \frac{1}{\beta} - (1 - \delta)")

else:
    st.subheader("Labor vs. Wage (Intra-temporal Trade-off)")
    from models.labor_supply import labor_vs_wage_for_plot
    w_arr, l_arr = labor_vs_wage_for_plot(result, shock_vals)

    fig_lw = go.Figure(layout=_layout(
        title="Labor Supply at Each Wage State",
        yaxis_title="Optimal Labor L*",
        yaxis_range=[0, 1]))
    fig_lw.add_trace(go.Bar(
        x=[f"w = {w:.2f}" for w in w_arr], y=list(l_arr),
        marker_color=[colors[0], colors[1]],
        text=[f"{v:.3f}" for v in l_arr], textposition="outside"))
    _style_axes(fig_lw)
    st.plotly_chart(fig_lw, width="stretch", config=_PLOTLY_CFG)

    _download_btns(fig_lw,
                   {"wage": w_arr, "labor": l_arr},
                   "labor_vs_wage", "sec6")

    st.markdown(f"""
**Intuition:** When wages rise from {w_arr[0]:.2f} to {w_arr[1]:.2f}, optimal labor
{'increases' if l_arr[1] > l_arr[0] else 'decreases'} from {l_arr[0]:.3f} to {l_arr[1]:.3f}.
{'This substitution effect dominates: higher wages make work more attractive relative to leisure.' if l_arr[1] > l_arr[0] else 'The income effect dominates: higher wages allow the household to maintain consumption with fewer hours.'}
With $\\psi = {psi:.1f}$ (leisure weight) and $\\nu = {nu:.1f}$ (leisure curvature), the
intra-temporal FOC balances:
""")
    st.latex(r"w \, u'(c) = \psi \, v'(1 - L)")


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "ECO 317 -- Intermediate Macroeconomic Theory | Spring 2026 | "
    "AI-Assisted Macroeconomic Modeling Dashboard"
)
