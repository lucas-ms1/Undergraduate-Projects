"""
pages/empirical.py
==================
Module I: The Empirical Data Suite (from Assignment 1)

Serves as the empirical anchor of the dashboard:
  - Automated retrieval from FRED and Yahoo Finance
  - Data transformations (log-diff, YoY, etc.)
  - Professional charts with recession shading, dual axes, trend lines
  - Interactive econometric models (OLS on live data)
  - AI "intelligent" summary
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os

from config.theme import (
    get_theme, colors, plotly_layout, style_axes,
    download_btns, PLOTLY_CFG, FONT,
)
from empirical.transforms import TRANSFORMS, hp_filter


# ---------------------------------------------------------------------------
# Preset data bundles
# ---------------------------------------------------------------------------
PRESET_BUNDLES = {
    "Business Cycle (Quarterly)": {
        "fred_ids": ["GDPC1", "PCECC96", "GPDIC1", "UNRATE", "FEDFUNDS"],
        "labels": {
            "GDPC1": "Real GDP",
            "PCECC96": "Real PCE",
            "GPDIC1": "Real Investment",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Fed Funds Rate",
        },
        "start": "1960-01-01",
    },
    "Inflation & Rates": {
        "fred_ids": ["CPIAUCSL", "CPILFESL", "FEDFUNDS", "DGS10", "DGS2"],
        "labels": {
            "CPIAUCSL": "CPI (All Items)",
            "CPILFESL": "Core CPI",
            "FEDFUNDS": "Fed Funds Rate",
            "DGS10": "10-Year Treasury",
            "DGS2": "2-Year Treasury",
        },
        "start": "1970-01-01",
    },
    "Labor Market": {
        "fred_ids": ["PAYEMS", "UNRATE", "ICSA", "AWHAETP", "CES0500000003"],
        "labels": {
            "PAYEMS": "Nonfarm Payrolls",
            "UNRATE": "Unemployment Rate",
            "ICSA": "Initial Claims",
            "AWHAETP": "Avg Weekly Hours",
            "CES0500000003": "Avg Hourly Earnings",
        },
        "start": "1970-01-01",
    },
    "Custom": {
        "fred_ids": [],
        "labels": {},
        "start": "1970-01-01",
    },
}


def render():
    """Render the complete Empirical Data Suite module."""
    TH = get_theme()
    C = colors(TH)

    st.title("Module I: The Empirical Data Suite")
    st.markdown(
        "Explore live macroeconomic data from **FRED** and **Yahoo Finance** "
        "with automated retrieval, professional visualizations, and "
        "interactive econometric analysis."
    )

    # ── FRED API key check ───────────────────────────────────────────────
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        st.warning(
            "No FRED API key detected. Set the `FRED_API_KEY` environment "
            "variable before running. You can get a free key at "
            "[fred.stlouisfed.org/docs/api/api_key.html]"
            "(https://fred.stlouisfed.org/docs/api/api_key.html)."
        )

    # ── Sidebar controls ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Empirical Settings")

        bundle_name = st.selectbox(
            "Data Bundle",
            list(PRESET_BUNDLES.keys()),
            key="emp_bundle",
        )
        bundle = PRESET_BUNDLES[bundle_name]

        if bundle_name == "Custom":
            custom_ids = st.text_input(
                "FRED Series IDs (comma-separated)",
                value="GDPC1, CPIAUCSL, UNRATE",
                key="emp_custom_ids",
            )
            fred_ids = [s.strip().upper() for s in custom_ids.split(",") if s.strip()]
        else:
            fred_ids = bundle["fred_ids"]

        transform_name = st.selectbox(
            "Transformation",
            list(TRANSFORMS.keys()),
            key="emp_transform",
        )

        start_date = st.text_input("Start date", bundle.get("start", "1970-01-01"), key="emp_start")
        show_recession = st.checkbox("Recession shading", value=True, key="emp_recession")
        show_trend = st.checkbox("HP trend line", value=False, key="emp_trend")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_viz, tab_econ = st.tabs(["Visualization", "Econometrics"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1: Data Retrieval & Visualization
    # ══════════════════════════════════════════════════════════════════════
    with tab_viz:
        if not fred_ids:
            st.info("Select a data bundle or enter custom FRED series IDs.")
            return

        if not api_key:
            st.error("Cannot fetch data without a FRED API key.")
            return

        # Fetch data (cached)
        @st.cache_data(show_spinner="Fetching FRED data...", ttl=3600)
        def _fetch(ids, start, key):
            from empirical.data_fetch import fetch_fred_series
            return fetch_fred_series(ids, start=start, api_key=key)

        try:
            raw_df = _fetch(fred_ids, start_date, api_key)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

        if raw_df.empty:
            st.warning("No data returned. Check your series IDs and API key.")
            return

        # Apply transformation
        transform_fn = TRANSFORMS[transform_name]
        transformed = {}
        for col in raw_df.columns:
            try:
                transformed[col] = transform_fn(raw_df[col].dropna())
            except Exception:
                transformed[col] = raw_df[col].dropna()

        # Build labels
        labels = bundle.get("labels", {})

        # Determine if we need dual axes (different scales)
        st.subheader("Macroeconomic Time Series")

        fig = go.Figure(layout=plotly_layout(TH,
            title=f"{bundle_name} — {transform_name}",
            xaxis_title="Date",
            yaxis_title=transform_name,
            height=500))

        for i, (sid, series) in enumerate(transformed.items()):
            label = labels.get(sid, sid)
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                mode="lines", name=label,
                line=dict(width=1.5, color=C[i % len(C)]),
            ))

            # Optional HP trend
            if show_trend and len(series.dropna()) > 20:
                try:
                    trend, _ = hp_filter(series.dropna())
                    fig.add_trace(go.Scatter(
                        x=trend.index, y=trend.values,
                        mode="lines", name=f"{label} (HP trend)",
                        line=dict(width=1, color=C[i % len(C)], dash="dash"),
                        showlegend=True,
                    ))
                except Exception:
                    pass

        # Recession shading
        if show_recession:
            @st.cache_data(ttl=86400)
            def _fetch_recessions(key):
                from empirical.data_fetch import fetch_recession_dates
                return fetch_recession_dates(api_key=key)

            try:
                recessions = _fetch_recessions(api_key)
                for (rs, re) in recessions:
                    fig.add_vrect(
                        x0=rs, x1=re,
                        fillcolor="rgba(128,128,128,0.15)",
                        line_width=0, layer="below",
                    )
            except Exception:
                pass

        style_axes(fig, TH)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

        # Data preview
        with st.expander("Raw Data Preview"):
            st.dataframe(raw_df.tail(20), use_container_width=True)
            csv = raw_df.to_csv().encode()
            st.download_button("Download CSV", csv, "empirical_data.csv", "text/csv",
                               key="emp_csv_download")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2: Econometrics
    # ══════════════════════════════════════════════════════════════════════
    with tab_econ:
        st.subheader("Interactive OLS Regression")
        st.markdown(
            "Run a flexible OLS regression on the live data. "
            "Select your dependent and independent variables below."
        )

        if not api_key or raw_df is None or raw_df.empty:
            st.info("Fetch data first (ensure FRED API key is set).")
            return

        available_cols = list(raw_df.columns)
        if len(available_cols) < 2:
            st.info("Need at least 2 series for regression.")
            return

        col_y = st.selectbox(
            "Dependent variable (y)",
            available_cols,
            index=0,
            key="econ_y",
        )
        col_x = st.multiselect(
            "Independent variables (X)",
            [c for c in available_cols if c != col_y],
            default=[c for c in available_cols if c != col_y][:1],
            key="econ_x",
        )

        n_lags = st.number_input("Lags to include", 0, 8, 0, key="econ_lags")
        use_transform = st.checkbox(
            f"Apply '{transform_name}' before regression", value=True, key="econ_use_transform"
        )

        if col_x and st.button("Run OLS", key="econ_run"):
            from empirical.econometrics import run_ols, build_regression_data

            # Prepare data — align indices FIRST, then transform
            aligned = raw_df[[col_y] + col_x].dropna()
            if use_transform and transform_name != "Raw Level":
                transform_fn = TRANSFORMS[transform_name]
                reg_df = aligned.apply(lambda c: transform_fn(c)).dropna()
            else:
                reg_df = aligned

            try:
                y, X = build_regression_data(reg_df, col_y, col_x, lags=n_lags)
                result = run_ols(y, X)

                st.markdown(f"**{result.dependent_name}** ~ {' + '.join(result.variable_names)}")
                st.dataframe(result.summary_df(), use_container_width=True, hide_index=True)

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("R²", f"{result.r_squared:.4f}")
                col_r2.metric("Adj. R²", f"{result.adj_r_squared:.4f}")
                col_r3.metric("N observations", str(result.n_obs))

                # Fitted vs actual plot
                fig_fit = go.Figure(layout=plotly_layout(TH,
                    title="Fitted vs. Actual",
                    xaxis_title="Observation",
                    yaxis_title=col_y))
                fig_fit.add_trace(go.Scatter(
                    y=y.values, mode="lines", name="Actual",
                    line=dict(width=1.5, color=C[0])))
                fig_fit.add_trace(go.Scatter(
                    y=result.fitted, mode="lines", name="Fitted",
                    line=dict(width=1.5, color=C[1], dash="dash")))
                style_axes(fig_fit, TH)
                st.plotly_chart(fig_fit, use_container_width=True, config=PLOTLY_CFG)

            except Exception as e:
                st.error(f"Regression failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # AI "Intelligent" Summary
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("Economic Analysis")

    if raw_df is not None and not raw_df.empty:
        latest = raw_df.iloc[-1]
        series_summaries = []
        for col in raw_df.columns:
            label = labels.get(col, col)
            val = latest.get(col)
            if pd.notna(val):
                series_summaries.append(f"- **{label}** ({col}): latest value = {val:.2f}")

        summaries_text = "\n".join(series_summaries) if series_summaries else "No data available."

        n_obs = len(raw_df)
        date_range = f"{raw_df.index[0].strftime('%Y-%m') if hasattr(raw_df.index[0], 'strftime') else raw_df.index[0]} to {raw_df.index[-1].strftime('%Y-%m') if hasattr(raw_df.index[-1], 'strftime') else raw_df.index[-1]}"

        st.markdown(f"""
**Data Overview:**
The current bundle contains **{len(fred_ids)} series** spanning {date_range}
({n_obs} observations). All data is fetched live from the FRED API and
reflects the most recent available releases.

**Latest Values:**
{summaries_text}

**Transformation Applied:** {transform_name}.
{'Log-differencing approximates percentage changes and is the standard transformation for trending macro series (GDP, CPI). It makes the series stationary, a prerequisite for valid regression inference.' if 'Log' in transform_name else 'Year-over-year growth rates remove seasonal patterns and show the annual pace of change, commonly used for inflation and employment reporting.' if 'YoY' in transform_name else 'Raw levels are shown without transformation. For trending series, consider applying log-differences or YoY growth for meaningful comparisons.'}
""")
