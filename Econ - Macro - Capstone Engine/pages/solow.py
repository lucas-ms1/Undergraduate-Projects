"""
pages/solow.py
==============
Module II: Long-Run Growth -- The Solow Model

Interactive Streamlit page for the Solow Growth Model with Harrod-neutral
technological progress.  Features:
  - Sidebar sliders for s, δ, n, g, α
  - Transition dynamics (k_t, y_t, c_t time series)
  - Classic Solow diagram (sf(k) vs. (n+g+δ)k)
  - Golden Rule calculation & visual marker
  - Capital destruction simulation
  - AI "intelligent" summary
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.theme import (
    get_theme, colors, plotly_layout, style_axes,
    download_btns, PLOTLY_CFG,
)
from solow.model import (
    steady_state_k,
    golden_rule,
    simulate_transition,
    solow_diagram_curves,
    simulate_capital_destruction,
)


def render():
    """Render the complete Solow Growth Model module."""
    TH = get_theme()
    C = colors(TH)

    st.title("Module II: Long-Run Growth -- The Solow Model")
    st.markdown(
        "Simulate the Solow Growth Model with **Harrod-neutral technological "
        "progress** to explore long-run steady-state growth, transition "
        "dynamics, and the Golden Rule of capital accumulation."
    )

    # ── Model formulation ────────────────────────────────────────────────
    with st.expander("Model Formulation", expanded=False):
        st.markdown("**Cobb-Douglas production function:**")
        st.latex(r"Y_t = K_t^{\alpha} \left(A_t L_t\right)^{1-\alpha}")
        st.markdown(
            "where $A_t$ grows at rate $g$ (technological progress) and "
            "$L_t$ grows at rate $n$ (population growth)."
        )
        st.markdown("**In effective-worker units** ($k \\equiv K/(AL)$, $y \\equiv Y/(AL)$):")
        st.latex(r"y = k^{\alpha}")
        st.markdown("**Law of motion:**")
        st.latex(r"k_{t+1} = \frac{s \cdot k_t^{\alpha} + (1-\delta)\,k_t}{(1+n)(1+g)}")
        st.markdown("**Steady state (analytical):**")
        st.latex(r"k^* = \left[\frac{s}{n + g + \delta}\right]^{\frac{1}{1-\alpha}}")

    # ── Sidebar parameters ───────────────────────────────────────────────
    with st.sidebar:
        st.header("Solow Parameters")
        s = st.slider("Saving rate (s)", 0.05, 0.80, 0.30, 0.01, key="solow_s")
        alpha = st.slider("Capital share (α)", 0.10, 0.70, 0.33, 0.01, key="solow_alpha")
        delta = st.slider("Depreciation (δ)", 0.01, 0.20, 0.05, 0.01, key="solow_delta")
        n = st.slider("Population growth (n)", 0.00, 0.05, 0.01, 0.005, key="solow_n")
        g = st.slider("Tech. progress (g)", 0.00, 0.05, 0.02, 0.005, key="solow_g")

        st.markdown("---")
        st.subheader("Simulation Settings")
        T_sim = st.number_input("Periods", 50, 500, 200, 25, key="solow_T")
        k0_mult = st.slider(
            "Initial k₀ (× k*)", 0.1, 3.0, 0.5, 0.1, key="solow_k0",
            help="Set initial capital relative to steady state"
        )

    # ── Compute steady state + golden rule ───────────────────────────────
    k_star = steady_state_k(s, n, g, delta, alpha)
    y_star = k_star ** alpha
    c_star = (1 - s) * y_star
    breakeven = n + g + delta

    gr = golden_rule(n, g, delta, alpha)

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("k* (steady state)", f"{k_star:.4f}")
    col2.metric("y* (output/eff. worker)", f"{y_star:.4f}")
    col3.metric("c* (consumption/eff. worker)", f"{c_star:.4f}")
    col4.metric("Golden Rule s", f"{gr['s_gold']:.4f}")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_trans, tab_diagram, tab_golden, tab_destroy = st.tabs([
        "Transition Dynamics",
        "Solow Diagram",
        "Golden Rule",
        "Capital Destruction",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Transition Dynamics
    # ══════════════════════════════════════════════════════════════════════
    with tab_trans:
        st.subheader("Transition from Initial State to Steady State")

        k0 = k0_mult * k_star
        sim = simulate_transition(s, n, g, delta, alpha, k0=k0, T=T_sim)

        fig_trans = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=[
                "Capital per Effective Worker (k_t)",
                "Output per Effective Worker (y_t)",
                "Consumption per Effective Worker (c_t)",
            ],
        )

        # k_t
        fig_trans.add_trace(go.Scatter(
            x=sim["t"], y=sim["k"], mode="lines",
            name="k_t", line=dict(width=2, color=C[0]),
        ), row=1, col=1)
        fig_trans.add_hline(
            y=k_star, line_dash="dash", line_color="gray",
            annotation_text=f"k* = {k_star:.3f}", row=1, col=1,
        )

        # y_t
        fig_trans.add_trace(go.Scatter(
            x=sim["t"], y=sim["y"], mode="lines",
            name="y_t", line=dict(width=2, color=C[1]),
        ), row=2, col=1)
        fig_trans.add_hline(
            y=y_star, line_dash="dash", line_color="gray",
            annotation_text=f"y* = {y_star:.3f}", row=2, col=1,
        )

        # c_t
        fig_trans.add_trace(go.Scatter(
            x=sim["t"], y=sim["c"], mode="lines",
            name="c_t", line=dict(width=2, color=C[2]),
        ), row=3, col=1)
        fig_trans.add_hline(
            y=c_star, line_dash="dash", line_color="gray",
            annotation_text=f"c* = {c_star:.3f}", row=3, col=1,
        )

        fig_trans.update_xaxes(title_text="Period", row=3, col=1)
        fig_trans.update_layout(**plotly_layout(TH, height=750,
            title="Solow Transition Dynamics"))
        style_axes(fig_trans, TH)
        st.plotly_chart(fig_trans, use_container_width=True, config=PLOTLY_CFG)

        download_btns(fig_trans,
                      {"t": sim["t"], "k": sim["k"], "y": sim["y"], "c": sim["c"]},
                      "solow_transition", "solow_trans")

        # Convergence check
        pct_of_ss = sim["k"][-1] / k_star * 100 if k_star > 0 else 0
        converge_t = np.argmax(np.abs(sim["k"] - k_star) / k_star < 0.01)
        if converge_t == 0 and np.abs(sim["k"][0] - k_star) / k_star >= 0.01:
            converge_t = T_sim  # didn't converge within horizon

        st.info(
            f"After {T_sim} periods, capital is at **{pct_of_ss:.1f}%** of "
            f"its steady-state value. "
            f"{'The economy reached within 1% of k* at period ' + str(converge_t) + '.' if converge_t < T_sim else 'The economy has not yet converged within 1% of k* — try increasing the simulation horizon.'}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Classic Solow Diagram
    # ══════════════════════════════════════════════════════════════════════
    with tab_diagram:
        st.subheader("The Classic Solow Diagram")

        curves = solow_diagram_curves(s, n, g, delta, alpha)

        fig_solow = go.Figure(layout=plotly_layout(TH,
            title="Actual vs. Break-Even Investment",
            xaxis_title="Capital per Effective Worker (k)",
            yaxis_title="Investment per Effective Worker"))

        # Actual investment: s·f(k)
        fig_solow.add_trace(go.Scatter(
            x=curves["k_grid"], y=curves["sf_k"],
            mode="lines", name="s·f(k) (actual investment)",
            line=dict(width=2.5, color=C[0]),
        ))

        # Break-even: (n+g+δ)k
        fig_solow.add_trace(go.Scatter(
            x=curves["k_grid"], y=curves["breakeven_k"],
            mode="lines", name="(n+g+δ)k (break-even)",
            line=dict(width=2.5, color=C[1]),
        ))

        # Steady-state marker
        fig_solow.add_trace(go.Scatter(
            x=[k_star], y=[s * k_star ** alpha],
            mode="markers", name=f"k* = {k_star:.3f}",
            marker=dict(size=12, color=C[3], symbol="star"),
        ))

        # Golden Rule marker
        fig_solow.add_trace(go.Scatter(
            x=[gr["k_gold"]], y=[gr["s_gold"] * gr["k_gold"] ** alpha],
            mode="markers", name=f"k_gold = {gr['k_gold']:.3f}",
            marker=dict(size=10, color=C[4], symbol="diamond"),
        ))

        # Vertical line at k*
        fig_solow.add_vline(
            x=k_star, line_dash="dot", line_color="gray",
            annotation_text=f"k* = {k_star:.3f}",
        )

        style_axes(fig_solow, TH)
        st.plotly_chart(fig_solow, use_container_width=True, config=PLOTLY_CFG)

        st.markdown(
            f"At the current parameters ($s = {s:.2f}$, $\\alpha = {alpha:.2f}$, "
            f"$\\delta = {delta:.2f}$, $n = {n:.3f}$, $g = {g:.3f}$), the "
            f"**steady-state capital** per effective worker is $k^* = {k_star:.4f}$ "
            f"and output is $y^* = {y_star:.4f}$."
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3: Golden Rule
    # ══════════════════════════════════════════════════════════════════════
    with tab_golden:
        st.subheader("The Golden Rule of Capital Accumulation")
        st.markdown(
            "The Golden Rule saving rate **maximizes steady-state consumption**. "
            "At the Golden Rule, the marginal product of capital equals the "
            "break-even rate:"
        )
        st.latex(r"MPK = \alpha \, k^{\alpha - 1} = n + g + \delta \implies s_{gold} = \alpha")

        col_gr1, col_gr2, col_gr3 = st.columns(3)
        col_gr1.metric("Golden Rule s", f"{gr['s_gold']:.4f}")
        col_gr2.metric("Current s", f"{s:.4f}")
        col_gr3.metric(
            "Gap",
            f"{s - gr['s_gold']:+.4f}",
            delta=f"{'Over-saving' if s > gr['s_gold'] else 'Under-saving' if s < gr['s_gold'] else 'At Golden Rule'}",
            delta_color="inverse" if s > gr['s_gold'] else "normal",
        )

        # Consumption as function of s
        s_range = np.linspace(0.01, 0.95, 200)
        c_star_range = np.array([
            (1 - si) * steady_state_k(si, n, g, delta, alpha) ** alpha
            for si in s_range
        ])

        fig_golden = go.Figure(layout=plotly_layout(TH,
            title="Steady-State Consumption vs. Saving Rate",
            xaxis_title="Saving Rate (s)",
            yaxis_title="c* (consumption per eff. worker)"))

        fig_golden.add_trace(go.Scatter(
            x=s_range, y=c_star_range, mode="lines",
            name="c*(s)", line=dict(width=2.5, color=C[0]),
        ))

        # Golden Rule marker
        fig_golden.add_vline(
            x=gr["s_gold"], line_dash="dash", line_color=C[3],
            annotation_text=f"s_gold = {gr['s_gold']:.3f}",
        )

        # Current s marker
        fig_golden.add_vline(
            x=s, line_dash="dot", line_color=C[1],
            annotation_text=f"Current s = {s:.3f}",
        )

        fig_golden.add_trace(go.Scatter(
            x=[gr["s_gold"]], y=[gr["c_gold"]],
            mode="markers", name=f"Max c* = {gr['c_gold']:.4f}",
            marker=dict(size=12, color=C[3], symbol="star"),
        ))

        style_axes(fig_golden, TH)
        st.plotly_chart(fig_golden, use_container_width=True, config=PLOTLY_CFG)

        # Interpretation
        if abs(s - gr["s_gold"]) < 0.01:
            verdict = "The economy is **at the Golden Rule** — consumption is maximized."
        elif s > gr["s_gold"]:
            verdict = (
                f"The economy is **dynamically inefficient** — the saving rate "
                f"($s = {s:.2f}$) exceeds the Golden Rule ($s_{{gold}} = {gr['s_gold']:.2f}$). "
                f"Reducing savings would raise consumption in **both** the short and long run."
            )
        else:
            verdict = (
                f"The economy is **below the Golden Rule** — the saving rate "
                f"($s = {s:.2f}$) is below $s_{{gold}} = {gr['s_gold']:.2f}$. "
                f"Raising savings would increase long-run consumption at the cost "
                f"of a temporary dip in current consumption."
            )
        st.markdown(verdict)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4: Capital Destruction Simulation
    # ══════════════════════════════════════════════════════════════════════
    with tab_destroy:
        st.subheader("Sudden Destruction of Capital")
        st.markdown(
            "Simulate a catastrophic event (war, natural disaster) that "
            "**destroys a fraction of the capital stock**. Observe how the "
            "economy rapidly catches up to its steady state via temporarily "
            "elevated growth rates — a key prediction of the Solow model."
        )

        with st.sidebar:
            st.markdown("---")
            st.subheader("Destruction Event")
            destruction_frac = st.slider(
                "Fraction destroyed", 0.10, 0.90, 0.50, 0.05,
                key="solow_destruct_frac",
            )

        dest = simulate_capital_destruction(
            s, n, g, delta, alpha,
            destruction_frac=destruction_frac,
            T_before=50, T_after=150,
        )

        fig_dest = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=[
                "Output per Effective Worker (y_t)",
                "Growth Rate of y_t (%)",
            ],
        )

        # Output path
        fig_dest.add_trace(go.Scatter(
            x=dest["t"], y=dest["y"], mode="lines",
            name="y_t", line=dict(width=2, color=C[0]),
        ), row=1, col=1)
        fig_dest.add_hline(
            y=y_star, line_dash="dash", line_color="gray",
            annotation_text=f"y* = {y_star:.3f}", row=1, col=1,
        )

        # Growth rate
        fig_dest.add_trace(go.Scatter(
            x=dest["t"], y=dest["growth_rate"], mode="lines",
            name="Growth %", line=dict(width=2, color=C[1]),
        ), row=2, col=1)
        fig_dest.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Destruction event marker
        fig_dest.add_vline(
            x=dest["destruction_period"], line_dash="dash",
            line_color=C[4], annotation_text="Destruction Event",
        )

        fig_dest.update_xaxes(title_text="Period", row=2, col=1)
        fig_dest.update_layout(**plotly_layout(TH, height=600,
            title=f"Capital Destruction ({destruction_frac:.0%} destroyed)"))
        style_axes(fig_dest, TH)
        st.plotly_chart(fig_dest, use_container_width=True, config=PLOTLY_CFG)

        # Recovery analysis
        post_shock_y = dest["y"][dest["destruction_period"]:]
        if len(post_shock_y) > 1:
            recovery_pct = post_shock_y[-1] / y_star * 100
            peak_growth = np.max(dest["growth_rate"][dest["destruction_period"]:])
            st.info(
                f"After losing **{destruction_frac:.0%}** of capital, "
                f"the economy recovers to **{recovery_pct:.1f}%** of steady-state output "
                f"within {len(post_shock_y)} periods. "
                f"Peak growth rate immediately after destruction: **{peak_growth:.2f}%**."
            )

    # ══════════════════════════════════════════════════════════════════════
    # AI "Intelligent" Summary
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Economic Analysis")

    half_life = None
    if k0_mult != 1.0:
        sim_check = simulate_transition(s, n, g, delta, alpha, k0=k0_mult * k_star, T=T_sim)
        gap = np.abs(sim_check["k"] - k_star)
        initial_gap = gap[0]
        if initial_gap > 0:
            half_idx = np.argmax(gap < initial_gap / 2)
            if half_idx > 0:
                half_life = half_idx

    st.markdown(f"""
**Steady-State Analysis:**
The economy's steady-state capital per effective worker is $k^* = {k_star:.4f}$,
yielding output $y^* = {y_star:.4f}$ and consumption $c^* = {c_star:.4f}$ per effective
worker. The break-even investment rate is $(n + g + \\delta) = ({n:.3f} + {g:.3f} + {delta:.2f}) = {breakeven:.4f}$.

**Golden Rule Assessment:**
The Golden Rule saving rate is $s_{{gold}} = \\alpha = {alpha:.2f}$, yielding maximum
steady-state consumption of $c^*_{{gold}} = {gr['c_gold']:.4f}$.
{'The current saving rate $s = ' + f'{s:.2f}' + '$ exceeds the Golden Rule — the economy is **dynamically inefficient**. A benevolent social planner could raise consumption in every period by lowering the saving rate.' if s > gr['s_gold'] + 0.01 else 'The current saving rate $s = ' + f'{s:.2f}' + '$ is below the Golden Rule. Increasing savings would raise long-run consumption but requires a short-run sacrifice — a classic intergenerational trade-off.' if s < gr['s_gold'] - 0.01 else 'The current saving rate is approximately at the Golden Rule — steady-state consumption is near its maximum.'}

**Convergence Speed:**
{'The half-life of convergence (time for the gap to k* to halve) is approximately **' + str(half_life) + ' periods**. This reflects the neoclassical growth model prediction that poorer economies (further from k*) grow faster — the engine of conditional convergence.' if half_life else 'Starting at steady state, there is no convergence dynamics to measure. Adjust the initial k₀ slider to observe transition behavior.'}

**Growth Decomposition:**
In the Solow model, long-run per-capita growth is driven entirely by technological
progress $g = {g:.3f}$, not by capital accumulation. Capital deepening only generates
*transitional* growth during convergence to the steady state. This explains why the
growth rate of $y_t$ declines monotonically toward zero as the economy approaches $k^*$.
""")
