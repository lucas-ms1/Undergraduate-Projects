"""
pages/vfi_models.py
===================
Module III: Micro-Founded Dynamic Models (from Assignment 2)

Infinite-horizon dynamic programming with Value Function Iteration (VFI).
Sub-navigation between three models:
  1. Stochastic Consumption-Savings (CES + income shocks)
  2. Stochastic Robinson Crusoe (capital accumulation + TFP shocks)
  3. Endogenous Labor Supply

Features: solved value/policy functions, stochastic simulation,
summary statistics (moments), exact non-linear forecasting, AI summaries.
"""

import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.theme import (
    get_theme, colors, plotly_layout, style_axes,
    download_btns, PLOTLY_CFG,
)
from vfi.config import MODEL1_DEFAULTS, MODEL2_DEFAULTS, MODEL3_DEFAULTS, SIM_DEFAULTS


COMPARISON_SERIES = {
    "Output / Income": {
        "model_keys": ["y", "earnings"],
        "fred_id": "GDPC1",
        "label": "Real GDP",
        "start": "1960-01-01",
    },
    "Consumption": {
        "model_keys": ["c"],
        "fred_id": "PCECC96",
        "label": "Real PCE",
        "start": "1960-01-01",
    },
    "Investment": {
        "model_keys": ["investment"],
        "fred_id": "GPDIC1",
        "label": "Real Private Domestic Investment",
        "start": "1960-01-01",
    },
    "Labor": {
        "model_keys": ["labor"],
        "fred_id": "HOANBS",
        "label": "Nonfarm Business Hours",
        "start": "1960-01-01",
    },
}


def _first_available_model_key(sim: dict, keys: list[str]) -> str | None:
    for key in keys:
        if key in sim:
            return key
    return None


def _index_to_100(series: pd.Series) -> pd.Series:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return clean
    base = clean.iloc[0]
    if base == 0 or pd.isna(base):
        return clean
    return clean / base * 100


def _comparison_moments(series: pd.Series) -> dict[str, float]:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"Mean": np.nan, "Std. dev.": np.nan, "Autocorrelation": np.nan}
    return {
        "Mean": float(clean.mean()),
        "Std. dev.": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "Autocorrelation": float(clean.autocorr(lag=1)) if len(clean) > 2 else np.nan,
    }


def _pct_growth(series: pd.Series) -> pd.Series:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    positive = clean[clean > 0]
    if len(positive) < 2:
        return pd.Series(dtype=float)
    return (np.log(positive) - np.log(positive.shift(1))).dropna() * 100


@st.cache_data(show_spinner="Fetching comparison data from FRED...", ttl=3600)
def _fetch_comparison_series(series_id, start, key):
    from empirical.data_fetch import fetch_fred_series
    return fetch_fred_series([series_id], start=start, api_key=key)


# ---------------------------------------------------------------------------
# Cached VFI solvers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Solving Model 1 via VFI...")
def _solve_model1(beta, sigma, r, y_L, y_H, p00, p11, a_min, a_max, n_a):
    from vfi.models.consumption_savings import solve
    params = dict(beta=beta, sigma=sigma, r=r,
                  y_vals=(y_L, y_H),
                  P=((p00, 1 - p00), (1 - p11, p11)),
                  a_min=a_min, a_max=a_max, n_a=n_a)
    return solve(params), params


@st.cache_data(show_spinner="Solving Model 2 via VFI...")
def _solve_model2(beta, sigma, alpha, delta, A, z_L, z_H, p00, p11,
                   k_min, k_max, n_k):
    from vfi.models.robinson_crusoe import solve
    params = dict(beta=beta, sigma=sigma, alpha=alpha, delta=delta, A=A,
                  z_vals=(z_L, z_H),
                  P=((p00, 1 - p00), (1 - p11, p11)),
                  k_min=k_min, k_max=k_max, n_k=n_k)
    return solve(params), params


@st.cache_data(show_spinner="Solving Model 3 via VFI...")
def _solve_model3(beta, sigma, psi, nu, r, w_L, w_H, p00, p11,
                   a_min, a_max, n_a, n_L, include_assets):
    from vfi.models.labor_supply import solve
    params = dict(beta=beta, sigma=sigma, psi=psi, nu=nu, r=r,
                  w_vals=(w_L, w_H),
                  P=((p00, 1 - p00), (1 - p11, p11)),
                  a_min=a_min, a_max=a_max, n_a=n_a,
                  n_L=n_L, include_assets=include_assets)
    return solve(params), params


def render():
    """Render the complete VFI Dynamic Models module."""
    TH = get_theme()
    C = colors(TH)

    st.title("Module III: Micro-Founded Dynamic Models")
    st.markdown(
        "Solve infinite-horizon stochastic macro models via **Value Function "
        "Iteration (VFI)**, simulate them, compute moments, and produce "
        "exact non-linear forecasts."
    )

    # ── Sidebar: model selector + parameters ─────────────────────────────
    with st.sidebar:
        st.header("VFI Parameters")
        model_choice = st.selectbox(
            "Select Model",
            ["Model 1: Consumption-Savings",
             "Model 2: Robinson Crusoe",
             "Model 3: Labor Supply"],
            key="vfi_model_choice",
        )
        MODEL_KEY = {
            "Model 1: Consumption-Savings": "model1",
            "Model 2: Robinson Crusoe": "model2",
            "Model 3: Labor Supply": "model3",
        }[model_choice]

        st.markdown("### Core Parameters")
        beta = st.slider("Discount factor (β)", 0.80, 0.99, 0.95, 0.01, key="vfi_beta")
        sigma = st.slider("Risk aversion (σ)", 0.5, 5.0, 2.0, 0.1, key="vfi_sigma")

        st.markdown("### Markov Transition Matrix")
        p00 = st.slider("P(Low | Low)", 0.50, 0.99, 0.90, 0.01, key="vfi_p00")
        p11 = st.slider("P(High | High)", 0.50, 0.99, 0.90, 0.01, key="vfi_p11")
        st.caption(f"P = [[{p00:.2f}, {1-p00:.2f}], [{1-p11:.2f}, {p11:.2f}]]")

        # Model-specific parameters
        if MODEL_KEY == "model1":
            st.markdown("### Model 1 Parameters")
            r = st.slider("Interest rate (r)", 0.00, 0.10, 0.03, 0.01, key="vfi_r")
            init_wealth = st.slider("Initial wealth", 0.0, 15.0, 5.0, 0.5, key="vfi_init_w")
            n_a = st.number_input("Grid size (n_a)", 40, 600, 200, 20, key="vfi_na")
        elif MODEL_KEY == "model2":
            st.markdown("### Model 2 Parameters")
            alpha = st.slider("Capital share (α)", 0.20, 0.50, 0.36, 0.01, key="vfi_alpha")
            delta = st.slider("Depreciation (δ)", 0.02, 0.20, 0.10, 0.01, key="vfi_delta")
            n_k = st.number_input("Grid size (n_k)", 40, 600, 200, 20, key="vfi_nk")
        else:
            st.markdown("### Model 3 Parameters")
            r_m3 = st.slider("Interest rate (r)", 0.00, 0.10, 0.03, 0.01, key="vfi_r3")
            psi = st.slider("Leisure weight (ψ)", 0.1, 5.0, 1.0, 0.1, key="vfi_psi")
            nu = st.slider("Leisure curvature (ν)", 0.5, 5.0, 2.0, 0.1, key="vfi_nu")
            include_assets = st.checkbox("Include assets", value=True, key="vfi_assets")
            if include_assets:
                init_wealth_m3 = st.slider("Initial wealth", 0.0, 15.0, 5.0, 0.5, key="vfi_init3")
                n_a_m3 = st.number_input("Grid size (n_a)", 40, 600, 150, 20, key="vfi_na3")
            n_L = st.number_input("Labor grid (n_L)", 20, 200, 60, 10, key="vfi_nL")

        st.markdown("### Simulation")
        T_sim = st.number_input("Periods", 100, 1000, SIM_DEFAULTS["T_sim"], 50, key="vfi_Tsim")
        seed = st.number_input("Random seed", 0, 9999, SIM_DEFAULTS["seed"], 1, key="vfi_seed")

    # ── Model formulation display ────────────────────────────────────────
    with st.expander("Model Formulation", expanded=False):
        st.markdown("#### Model 1: Stochastic Consumption-Savings")
        st.latex(r"""
        \begin{aligned}
        V(a_t,y_t)
        &= \max_{a_{t+1}\ge \underline{a}}
        \left\{
        u(c_t)
        + \beta \sum_{y_{t+1}\in\{y_L,y_H\}}
        \Pi(y_{t+1}\mid y_t)V(a_{t+1},y_{t+1})
        \right\} \\
        c_t &= (1+r)a_t + y_t - a_{t+1} \\
        u(c_t) &= \frac{c_t^{1-\sigma}}{1-\sigma}, \qquad c_t>0 \\
        y_t &\in \{y_L,y_H\}, \qquad
        \Pi =
        \begin{bmatrix}
        p_{LL} & 1-p_{LL} \\
        1-p_{HH} & p_{HH}
        \end{bmatrix}
        \end{aligned}
        """)

        st.markdown("#### Model 2: Stochastic Robinson Crusoe Production Economy")
        st.latex(r"""
        \begin{aligned}
        V(k_t,z_t)
        &= \max_{k_{t+1}\ge 0}
        \left\{
        u(c_t)
        + \beta \sum_{z_{t+1}\in\{z_L,z_H\}}
        \Pi(z_{t+1}\mid z_t)V(k_{t+1},z_{t+1})
        \right\} \\
        y_t &= z_t A k_t^{\alpha} \\
        c_t &= z_t A k_t^{\alpha} + (1-\delta)k_t - k_{t+1} \\
        i_t &= k_{t+1} - (1-\delta)k_t \\
        u(c_t) &= \frac{c_t^{1-\sigma}}{1-\sigma}, \qquad c_t>0 \\
        z_t &\in \{z_L,z_H\}
        \end{aligned}
        """)

        st.markdown("#### Model 3: Stochastic Labor Supply with Optional Savings")
        st.latex(r"""
        \begin{aligned}
        V(a_t,w_t)
        &= \max_{a_{t+1}\ge \underline{a},\,L_t\in[0,1]}
        \left\{
        u(c_t)
        + \psi\,v(1-L_t)
        + \beta \sum_{w_{t+1}\in\{w_L,w_H\}}
        \Pi(w_{t+1}\mid w_t)V(a_{t+1},w_{t+1})
        \right\} \\
        c_t &= (1+r)a_t + w_tL_t - a_{t+1} \\
        u(c_t) &= \frac{c_t^{1-\sigma}}{1-\sigma}, \qquad
        v(1-L_t)=\frac{(1-L_t)^{1-\nu}}{1-\nu} \\
        e_t &= w_tL_t, \qquad w_t\in\{w_L,w_H\}
        \end{aligned}
        """)

        st.markdown("#### Labor-Only Special Case")
        st.latex(r"""
        \begin{aligned}
        V(w_t)
        &= \max_{L_t\in[0,1]}
        \left\{
        u(w_tL_t) + \psi\,v(1-L_t)
        + \beta \sum_{w_{t+1}\in\{w_L,w_H\}}
        \Pi(w_{t+1}\mid w_t)V(w_{t+1})
        \right\} \\
        w_tu'(w_tL_t) &= \psi(1-L_t)^{-\nu}
        \end{aligned}
        """)

    # ── Solve ────────────────────────────────────────────────────────────
    if MODEL_KEY == "model1":
        d = MODEL1_DEFAULTS
        result, params = _solve_model1(
            beta, sigma, r, d["y_vals"][0], d["y_vals"][1],
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

    # Convergence diagnostic
    diag = result["diagnostics"]
    st.success(
        f"VFI converged: {diag['converged']}  |  "
        f"Iterations: {diag['iterations']}  |  "
        f"Final sup-norm error: {diag['final_error']:.2e}"
    )

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_vp, tab_sim, tab_moments, tab_forecast = st.tabs([
        "Value & Policy Functions",
        "Stochastic Simulation",
        "Summary Statistics",
        "Forecasting",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Value Function & Policy Plots
    # ══════════════════════════════════════════════════════════════════════
    with tab_vp:
        st.subheader("Solved Value & Policy Functions")

        if grid is not None:
            col1, col2 = st.columns(2)

            fig_v = go.Figure(layout=plotly_layout(TH,
                title="Value Function", xaxis_title=grid_label, yaxis_title="V"))
            for s_idx in range(2):
                fig_v.add_trace(go.Scatter(
                    x=grid, y=result["value_function"][:, s_idx],
                    mode="lines", name=shock_labels[s_idx],
                    line=dict(width=1.5, color=C[s_idx])))
            style_axes(fig_v, TH)
            col1.plotly_chart(fig_v, use_container_width=True, config=PLOTLY_CFG)

            fig_p = go.Figure(layout=plotly_layout(TH,
                title="Consumption Policy c*(state, shock)",
                xaxis_title=grid_label, yaxis_title="Consumption"))
            for s_idx in range(2):
                fig_p.add_trace(go.Scatter(
                    x=grid, y=result["c_policy"][:, s_idx],
                    mode="lines", name=shock_labels[s_idx],
                    line=dict(width=1.5, color=C[s_idx])))
            style_axes(fig_p, TH)
            col2.plotly_chart(fig_p, use_container_width=True, config=PLOTLY_CFG)

        elif sim_model_name == "model3_labor_only":
            st.subheader("Labor-Only Results (no asset dimension)")
            vf = result["value_function"]
            labor = result["policy_levels"]["labor"]
            c_pol = result["c_policy"]
            c1, c2 = st.columns(2)
            c1.metric("V(Low wage)", f"{vf[0]:.4f}")
            c1.metric("V(High wage)", f"{vf[1]:.4f}")
            c2.metric("L*(Low wage)", f"{labor[0]:.4f}")
            c2.metric("L*(High wage)", f"{labor[1]:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Stochastic Simulation
    # ══════════════════════════════════════════════════════════════════════
    with tab_sim:
        st.subheader("Stochastic Simulation")
        from vfi.simulation.simulate import simulate_model

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
                           line=dict(width=0.8, color=C[0]),
                           name=key.capitalize(), showlegend=False),
                row=i, col=1)
            fig_sim.update_yaxes(title_text=key.capitalize(), row=i, col=1)
        fig_sim.update_xaxes(title_text="Period", row=n_plots, col=1)
        fig_sim.update_layout(**plotly_layout(TH,
            title=f"Simulated Paths ({T_sim} periods)", height=280 * n_plots))
        style_axes(fig_sim, TH)
        st.plotly_chart(fig_sim, use_container_width=True, config=PLOTLY_CFG)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Summary Statistics
    # ══════════════════════════════════════════════════════════════════════
    with tab_moments:
        st.subheader("Summary Statistics (Moments)")
        from vfi.simulation.moments import compute_moments

        moments = compute_moments(
            {k: v for k, v in sim.items() if k != "shock_idx"},
            income_key=income_key,
        )
        df_moments = pd.DataFrame(moments).T.round(4)
        df_moments.index = [idx.capitalize() for idx in df_moments.index]
        df_moments.columns = ["Mean", "Variance", "Autocorrelation (lag-1)",
                              f"Corr with {income_key}"]
        st.dataframe(df_moments, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: Exact Non-Linear Forecasting
    # ══════════════════════════════════════════════════════════════════════
    with tab_forecast:
        st.subheader("Exact Non-Linear Forecast")
        st.markdown(
            "Select a shock path for the next periods. Uses the **exact "
            "solved policy functions** from VFI."
        )
        from vfi.simulation.forecast import forecast_model

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            shock_1 = st.selectbox("Period 1 shock", ["Low", "High"], index=1, key="vfi_fc1")
        with fc2:
            shock_2 = st.selectbox("Period 2 shock", ["Low", "High"], index=1, key="vfi_fc2")
        with fc3:
            shock_3 = st.selectbox("Period 3 shock", ["Low", "High"], index=0, key="vfi_fc3")

        shock_map = {"Low": 0, "High": 1}
        user_shocks = [shock_map[shock_1], shock_map[shock_2], shock_map[shock_3]]
        T_fcast = st.slider("Forecast horizon", 3, 30, SIM_DEFAULTS["T_fcast"], 1, key="vfi_Tfcast")
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
                           line=dict(width=1.5, color=C[1]),
                           marker=dict(size=5),
                           name=key.capitalize(), showlegend=False),
                row=i, col=1)
            fig_fc.update_yaxes(title_text=key.capitalize(), row=i, col=1)
        fig_fc.update_xaxes(title_text="Forecast Period", row=n_fp, col=1)
        fig_fc.update_layout(**plotly_layout(TH,
            title="Forecast Path", height=280 * n_fp))
        style_axes(fig_fc, TH)
        st.plotly_chart(fig_fc, use_container_width=True, config=PLOTLY_CFG)

    # ══════════════════════════════════════════════════════════════════════
    # AI "Intelligent" Summary
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Economic Analysis")

    if MODEL_KEY == "model1":
        m_c = moments["c"]
        m_y = moments["y"]
        var_ratio = m_c["variance"] / m_y["variance"] if m_y["variance"] > 0 else float("nan")

        st.markdown(f"""
**Consumption Smoothing Analysis:**
The consumption-to-income variance ratio is $\\text{{Var}}(c)/\\text{{Var}}(y) = {var_ratio:.3f}$.
{'This is well below one, indicating **significant consumption smoothing** — the household uses savings as a buffer against income shocks, consistent with the Permanent Income Hypothesis.' if var_ratio < 0.5 else 'This indicates **partial but incomplete** consumption smoothing. The household uses its savings buffer, but income volatility still bleeds through.' if var_ratio < 0.9 else 'This is near unity, indicating **hand-to-mouth behavior** — consumption tracks income almost one-for-one.'}

The Euler equation is $u'(c_t) = \\beta (1+r) \\, \\mathbb{{E}}_t [u'(c_{{t+1}})]$.
With $\\beta(1+r) = {beta*(1+r):.3f}$,
{'the household has a motive to accumulate assets (patient saver).' if beta*(1+r) > 1.0 else 'the household is effectively impatient and tends toward the borrowing constraint.' if beta*(1+r) < 0.99 else 'the household is roughly indifferent between consuming today and saving.'}

**Key Moments:**
Mean consumption = {m_c['mean']:.3f}, Consumption autocorrelation = {m_c['autocorrelation']:.3f}.
{'High autocorrelation confirms the savings buffer smooths consumption over time.' if m_c['autocorrelation'] > 0.8 else 'Moderate autocorrelation suggests partial smoothing.'}
""")

    elif MODEL_KEY == "model2":
        m_c = moments["c"]
        m_y = moments["y"]
        m_k = moments["k"]
        k_ss = ((beta * alpha * d["A"]) / (1 - beta * (1 - delta))) ** (1 / (1 - alpha))
        inv_share = moments["investment"]["mean"] / m_y["mean"] if m_y["mean"] > 0 else 0

        st.markdown(f"""
**Robinson Crusoe Economy:**
The deterministic steady-state capital is $k^* = {k_ss:.3f}$. The simulated mean capital
of **{m_k['mean']:.3f}** {'is close to' if abs(m_k['mean'] - k_ss) / k_ss < 0.1 else 'deviates from'}
this level due to precautionary saving under stochastic TFP.

The investment-to-output ratio is **{inv_share:.3f}** ({inv_share*100:.0f}% of output reinvested).
Capital autocorrelation is **{m_k['autocorrelation']:.3f}** — capital is inherently persistent
because the capital accumulation law changes slowly relative to output.

{'The high consumption-output correlation indicates limited ability to decouple spending from production shocks.' if m_c['corr_with_income'] > 0.8 else 'The moderate correlation shows the household partially buffers consumption through investment adjustment.'}
""")
        st.latex(r"k_{t+1}=z_tAk_t^\alpha+(1-\delta)k_t-c_t")

    else:
        m_c = moments["c"]
        m_l = moments.get("labor", {})
        mean_L = m_l.get("mean", 0)

        st.markdown(f"""
**Labor-Leisure Trade-Off:**
The household works an average of **{mean_L:.1%}** of its time endowment.
With Frisch elasticity $1/\\nu = {1/nu:.2f}$,
{'labor supply is quite rigid — hours barely respond to wage changes.' if nu > 2.0 else 'labor supply adjusts meaningfully to wage changes.' if nu > 1.0 else 'labor supply is highly elastic — small wage changes produce large hour adjustments.'}

The leisure weight $\\psi = {psi:.1f}$ controls the household's valuation of free time.
The intra-temporal FOC balances the marginal value of work against leisure:
""")
        st.latex(r"w \, u'(c) = \psi \, (1-L)^{-\nu}")

    # ═══════════════════════════════════════════════════════════════════════
    # Empirical comparison with Module I data
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Model vs. Real Data Comparison")
    st.markdown(
        "Compare the simulated VFI path with the matching FRED series from "
        "Module I. Both series are indexed to 100 at their first observation "
        "so the comparison emphasizes dynamics rather than units."
    )

    available_comparisons = {
        name: spec
        for name, spec in COMPARISON_SERIES.items()
        if _first_available_model_key(sim, spec["model_keys"]) is not None
    }

    if not available_comparisons:
        st.info("No comparable simulated series is available for this model.")
        return

    comp_name = st.selectbox(
        "Comparison variable",
        list(available_comparisons.keys()),
        key="vfi_empirical_comparison_var",
    )
    comp_spec = available_comparisons[comp_name]
    model_key = _first_available_model_key(sim, comp_spec["model_keys"])

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        st.warning(
            "Set the `FRED_API_KEY` environment variable to pull the real-data "
            "series for this comparison."
        )
        return

    try:
        real_df = _fetch_comparison_series(comp_spec["fred_id"], comp_spec["start"], api_key)
    except Exception as exc:
        st.error(f"Comparison data fetch failed: {exc}")
        return

    if real_df.empty or comp_spec["fred_id"] not in real_df:
        st.warning("No FRED data returned for the selected comparison series.")
        return

    real_series = real_df[comp_spec["fred_id"]].dropna()
    model_series = pd.Series(sim[model_key], name=f"Model {model_key}")

    n_compare = min(len(real_series), len(model_series))
    if n_compare < 5:
        st.info("Not enough overlapping observations to build the comparison.")
        return

    real_compare = real_series.iloc[-n_compare:]
    model_compare = model_series.iloc[-n_compare:]
    model_compare.index = real_compare.index

    real_indexed = _index_to_100(real_compare)
    model_indexed = _index_to_100(model_compare)

    c_real, c_model = st.columns(2)
    fig_real = go.Figure(layout=plotly_layout(
        TH,
        title=f"Real Data: {comp_spec['label']} ({comp_spec['fred_id']})",
        xaxis_title="Date",
        yaxis_title="Index, first obs. = 100",
        height=360,
    ))
    fig_real.add_trace(go.Scatter(
        x=real_indexed.index,
        y=real_indexed.values,
        mode="lines",
        name=comp_spec["label"],
        line=dict(width=1.5, color=C[0]),
    ))
    style_axes(fig_real, TH)
    c_real.plotly_chart(fig_real, use_container_width=True, config=PLOTLY_CFG)

    fig_model = go.Figure(layout=plotly_layout(
        TH,
        title=f"VFI Model Prediction: {model_key.capitalize()}",
        xaxis_title="Date-aligned simulation period",
        yaxis_title="Index, first obs. = 100",
        height=360,
    ))
    fig_model.add_trace(go.Scatter(
        x=model_indexed.index,
        y=model_indexed.values,
        mode="lines",
        name=f"Model {model_key}",
        line=dict(width=1.5, color=C[1]),
    ))
    style_axes(fig_model, TH)
    c_model.plotly_chart(fig_model, use_container_width=True, config=PLOTLY_CFG)

    fig_overlay = go.Figure(layout=plotly_layout(
        TH,
        title=f"Overlay: Real {comp_name} vs. VFI Prediction",
        xaxis_title="Date-aligned observation",
        yaxis_title="Index, first obs. = 100",
        height=460,
    ))
    fig_overlay.add_trace(go.Scatter(
        x=real_indexed.index,
        y=real_indexed.values,
        mode="lines",
        name=f"Real: {comp_spec['label']}",
        line=dict(width=1.7, color=C[0]),
    ))
    fig_overlay.add_trace(go.Scatter(
        x=model_indexed.index,
        y=model_indexed.values,
        mode="lines",
        name=f"Model: {model_key}",
        line=dict(width=1.7, color=C[1], dash="dash"),
    ))
    style_axes(fig_overlay, TH)
    st.plotly_chart(fig_overlay, use_container_width=True, config=PLOTLY_CFG)

    comparison_df = pd.DataFrame(
        {
            f"Real: {comp_spec['fred_id']}": _comparison_moments(real_indexed),
            f"Model: {model_key}": _comparison_moments(model_indexed),
        }
    ).T.round(4)
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("### Growth Rate Comparison")
    real_growth = _pct_growth(real_compare)
    model_growth = _pct_growth(model_compare)
    n_growth = min(len(real_growth), len(model_growth))

    if n_growth >= 5:
        real_growth = real_growth.iloc[-n_growth:]
        model_growth = model_growth.iloc[-n_growth:]
        model_growth.index = real_growth.index

        fig_growth = go.Figure(layout=plotly_layout(
            TH,
            title=f"Growth Rates: Real {comp_name} vs. VFI Prediction",
            xaxis_title="Date-aligned observation",
            yaxis_title="Log growth rate (%)",
            height=430,
        ))
        fig_growth.add_trace(go.Scatter(
            x=real_growth.index,
            y=real_growth.values,
            mode="lines",
            name=f"Real: {comp_spec['label']}",
            line=dict(width=1.6, color=C[0]),
        ))
        fig_growth.add_trace(go.Scatter(
            x=model_growth.index,
            y=model_growth.values,
            mode="lines",
            name=f"Model: {model_key}",
            line=dict(width=1.6, color=C[1], dash="dash"),
        ))
        style_axes(fig_growth, TH)
        st.plotly_chart(fig_growth, use_container_width=True, config=PLOTLY_CFG)
    else:
        st.info("Growth-rate comparison is unavailable for this selected model series.")

    st.markdown("### Cyclical Component Comparison")
    try:
        from empirical.transforms import hp_filter

        _, real_cycle = hp_filter(np.log(real_compare[real_compare > 0]), lamb=1600)
        _, model_cycle = hp_filter(np.log(model_compare[model_compare > 0]), lamb=1600)
        n_cycle = min(len(real_cycle), len(model_cycle))

        if n_cycle >= 10:
            real_cycle = real_cycle.iloc[-n_cycle:] * 100
            model_cycle = model_cycle.iloc[-n_cycle:] * 100
            model_cycle.index = real_cycle.index

            fig_cycle = go.Figure(layout=plotly_layout(
                TH,
                title=f"HP-Filtered Cycles: Real {comp_name} vs. VFI Prediction",
                xaxis_title="Date-aligned observation",
                yaxis_title="Deviation from trend (%)",
                height=430,
            ))
            fig_cycle.add_trace(go.Scatter(
                x=real_cycle.index,
                y=real_cycle.values,
                mode="lines",
                name=f"Real cycle: {comp_spec['label']}",
                line=dict(width=1.6, color=C[0]),
            ))
            fig_cycle.add_trace(go.Scatter(
                x=model_cycle.index,
                y=model_cycle.values,
                mode="lines",
                name=f"Model cycle: {model_key}",
                line=dict(width=1.6, color=C[1], dash="dash"),
            ))
            style_axes(fig_cycle, TH)
            st.plotly_chart(fig_cycle, use_container_width=True, config=PLOTLY_CFG)
        else:
            st.info("HP-cycle comparison needs more observations for this selected series.")
    except Exception as exc:
        st.info(f"HP-cycle comparison unavailable: {exc}")

    st.markdown("### Moments-Only Comparison")
    moment_sources = {
        f"Real level index: {comp_spec['fred_id']}": real_indexed,
        f"Model level index: {model_key}": model_indexed,
    }
    if n_growth >= 5:
        moment_sources[f"Real growth: {comp_spec['fred_id']}"] = real_growth
        moment_sources[f"Model growth: {model_key}"] = model_growth
    if "real_cycle" in locals() and "model_cycle" in locals() and len(real_cycle) >= 10 and len(model_cycle) >= 10:
        moment_sources[f"Real HP cycle: {comp_spec['fred_id']}"] = real_cycle
        moment_sources[f"Model HP cycle: {model_key}"] = model_cycle

    moments_only = pd.DataFrame(
        {name: _comparison_moments(series) for name, series in moment_sources.items()}
    ).T.round(4)
    st.dataframe(moments_only, use_container_width=True)
