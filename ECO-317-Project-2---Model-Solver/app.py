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

from analysis.calibration import (
    classify_two_state_series,
    download_fred_series,
    estimate_transition_matrix_from_states,
    objective_from_moments,
    parameter_bounds_for_model,
    run_structural_calibration,
)
from analysis.diagnostics import (
    edge_usage_summary,
    grid_convergence_check,
    policy_shape_checks,
    solver_diagnostics_summary,
)
from analysis.distribution import (
    run_distributional_simulation,
    summarize_distribution_runs,
)
from analysis.plots import (
    make_phase_diagram,
    make_policy_surfaces,
    make_value_surface,
)
from analysis.regime_stats import (
    compute_regime_spells,
    compute_transition_summary,
    conditional_series_stats,
    summarize_spell_stats,
)
from analysis.scorecards import (
    build_targets_from_empirical_bundle,
    build_moment_scorecard,
    get_default_empirical_moment_list,
    get_default_targets,
    get_fred_series_presets,
    targets_from_dataframe,
)
from analysis.risk_preferences import (
    ChoiceQuestion,
    certainty_equivalent,
    pick_next_question,
    posterior_summary,
    probs_from_logp,
    update_posterior_logp,
)
from analysis.ui_text import get_advanced_explanations
from analysis.welfare import (
    compare_counterfactual_welfare,
    summarize_state_welfare,
)
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
# Streamlit disallows mutating a widget-backed session_state key after the widget
# has been instantiated. We use a one-run "request" flag to set the checkbox
# state before it is created.
if st.session_state.get("request_advanced_mode", False):
    st.session_state["advanced_mode_enabled"] = True
    st.session_state["request_advanced_mode"] = False
advanced_mode = st.sidebar.checkbox("Advanced mode", value=False, key="advanced_mode_enabled")
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


@st.cache_data(show_spinner=False)
def _solve_model_from_params(model_key, params):
    """Generic cached solve for optional advanced re-solves."""
    if model_key == "model1":
        from models.consumption_savings import solve
    elif model_key == "model2":
        from models.robinson_crusoe import solve
    else:
        from models.labor_supply import solve
    return solve(dict(params))


@st.cache_data(show_spinner=False)
def _download_fred_cached(series_id, start_date, end_date):
    """Cached FRED series download for advanced empirical targets."""
    return download_fred_series(series_id, start_date=start_date, end_date=end_date)


def _apply_fred_transform(df, transform):
    """Apply optional transform to a downloaded FRED series for empirical workflows."""
    out = df.copy()
    values = pd.to_numeric(out["value"], errors="coerce")
    if transform == "pct_change":
        values = values.pct_change()
    elif transform == "log_diff":
        values = np.log(values.replace(0, np.nan)).diff()
    elif transform == "diff":
        values = values.diff()
    out["value"] = values
    out = out.dropna(subset=["date", "value"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No usable observations after applying transform '{transform}'.")
    return out


def _preset_labels_and_lookup(preset_df):
    rows = preset_df.to_dict(orient="records")
    labels = []
    row_lookup = {}
    transform_lookup = {}
    for row in rows:
        label = f"{row['variable']}: {row['label']} ({row['series_id']})"
        labels.append(label)
        row_lookup[label] = row
        transform_lookup[row["series_id"]] = row.get("transform", "level")
    return labels, row_lookup, transform_lookup


def _download_preset_bundle(model_key, selected_ids, start_date, end_date):
    bundle = {}
    failed = []
    for series_id in selected_ids:
        try:
            bundle[series_id] = _download_fred_cached(
                series_id,
                str(start_date),
                str(end_date),
            )
        except Exception:
            failed.append(series_id)

    st.session_state[f"fred_preset_bundle_{model_key}"] = bundle
    st.session_state[f"fred_preset_bundle_ids_{model_key}"] = selected_ids
    st.session_state[f"fred_preset_bundle_dates_{model_key}"] = (
        str(start_date),
        str(end_date),
    )
    return bundle, failed


def _render_fred_preset_controls(model_key, preset_df, key_prefix):
    if preset_df.empty:
        st.caption("No preset FRED series are defined for this model.")
        return {}, [], None

    preset_labels, row_lookup, _ = _preset_labels_and_lookup(preset_df)
    selected_preset_labels = st.multiselect(
        "Recommended series to use (defaults are recommended)",
        options=preset_labels,
        default=preset_labels,
        key=f"{key_prefix}_fred_labels_{model_key}",
    )
    selected_ids = [
        row_lookup[label]["series_id"]
        for label in selected_preset_labels
    ]
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        preset_start = st.date_input(
            "Data start",
            value=pd.Timestamp("2000-01-01"),
            key=f"{key_prefix}_fred_start_{model_key}",
        )
    with preset_col2:
        preset_end = st.date_input(
            "Data end",
            value=pd.Timestamp.today().date(),
            key=f"{key_prefix}_fred_end_{model_key}",
        )
    if st.button(
        "Download from FRED",
        key=f"{key_prefix}_download_fred_{model_key}",
        disabled=not selected_ids,
    ):
        _, failed = _download_preset_bundle(
            model_key,
            selected_ids,
            preset_start,
            preset_end,
        )
        if failed:
            st.warning(f"Could not load: {', '.join(failed)}")

    stored_bundle = st.session_state.get(f"fred_preset_bundle_{model_key}", {})
    stored_ids = st.session_state.get(f"fred_preset_bundle_ids_{model_key}", [])
    stored_dates = st.session_state.get(f"fred_preset_bundle_dates_{model_key}")
    if stored_bundle:
        st.caption(
            f"Loaded preset series: {', '.join(stored_ids)}"
            + (f" | Range: {stored_dates[0]} to {stored_dates[1]}" if stored_dates else "")
        )
    return stored_bundle, stored_ids, stored_dates


def _build_fred_targets_from_bundle(downloaded_bundle, preset_df, income_key, model_key):
    if not downloaded_bundle or preset_df.empty:
        return None, None, None
    try:
        selected_fred_ids = set(downloaded_bundle)
        active_preset_df = preset_df[preset_df["series_id"].isin(selected_fred_ids)].copy()
        if active_preset_df.empty:
            return None, None, None
        return build_targets_from_empirical_bundle(
            downloaded_bundle,
            active_preset_df,
            income_key=income_key,
            selected_moments=get_default_empirical_moment_list(model_key),
        )
    except Exception:
        return None, None, None


def _grid_variants(model_key, params):
    """Lightweight alternative grid sizes for optional convergence checks."""
    if model_key == "model1":
        key, minimum = "n_a", 40
    elif model_key == "model2":
        key, minimum = "n_k", 40
    elif params.get("include_assets", True):
        key, minimum = "n_a", 40
    else:
        key, minimum = "n_L", 20

    base = int(params[key])
    values = {
        max(minimum, int(round(base * 0.75 / 5) * 5)),
        base,
        max(minimum, int(round(base * 1.25 / 5) * 5)),
    }
    values = sorted(values)
    return key, [value for value in values if value != base]


def _active_shock_values(model_key, params):
    if model_key == "model1":
        return np.asarray(params["y_vals"], dtype=float)
    if model_key == "model2":
        return np.asarray(params["z_vals"], dtype=float)
    return np.asarray(params["w_vals"], dtype=float)


def _render_advanced_explanation(section_key, model_key):
    technical, intuition = get_advanced_explanations(section_key, model_key)
    st.subheader("Technical")
    st.markdown(technical)
    st.subheader("Intuition")
    st.markdown(intuition)


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
estimated_p_key = f"estimated_p_{MODEL_KEY}"
estimated_p_source_key = f"estimated_p_source_{MODEL_KEY}"
estimated_p_counts_key = f"estimated_p_counts_{MODEL_KEY}"
use_estimated_p_key = f"use_estimated_p_{MODEL_KEY}"
stored_estimated_P = st.session_state.get(estimated_p_key)
stored_estimated_P_source = st.session_state.get(estimated_p_source_key)
empirical_p_active = bool(st.session_state.get(use_estimated_p_key, False) and stored_estimated_P is not None)
advanced_view_key = f"advanced_view_{MODEL_KEY}"
advanced_analysis_view_key = f"advanced_analysis_view_{MODEL_KEY}"
advanced_view_options = [
    "Diagnostics",
    "Risk preferences",
    "Regime analytics",
    "Distribution lab",
    "Scorecard",
    "Phase diagrams",
    "Welfare",
    "Calibration",
]
requested_advanced_view = st.session_state.get(f"request_advanced_view_{MODEL_KEY}")
if requested_advanced_view in advanced_view_options:
    # Must be set before the sidebar radio with this key is instantiated.
    st.session_state[advanced_view_key] = requested_advanced_view
    st.session_state[f"request_advanced_view_{MODEL_KEY}"] = None
advanced_view = st.session_state.get(advanced_view_key, "Diagnostics")
fred_preset_df = get_fred_series_presets(MODEL_KEY)
fred_downloaded_bundle = st.session_state.get(f"fred_preset_bundle_{MODEL_KEY}", {})
fred_target_df = None
fred_target_preview = None
fred_empirical_moments = None

if advanced_mode:
    st.sidebar.markdown("### Advanced Menu")

    with st.sidebar.expander("Empirical parameter override", expanded=True):
        st.caption(
            "Replace the manual Markov transition matrix with an empirically estimated two-state matrix."
        )
        markov_data_source = st.radio(
            "Empirical Markov data source",
            ["Upload CSV", "Download from FRED"],
            horizontal=True,
            index=1,
            key=f"markov_source_{MODEL_KEY}",
        )
        markov_df = None

        if markov_data_source == "Upload CSV":
            uploaded_markov = st.file_uploader(
                "Optional empirical Markov CSV",
                type="csv",
                key=f"markov_upload_{MODEL_KEY}",
                help="Use a CSV with either a binary 'state' column or a numeric 'value' column.",
            )
            if uploaded_markov is not None:
                markov_df = pd.read_csv(uploaded_markov)
        else:
            markov_transform = "level"
            if not fred_preset_df.empty:
                markov_labels, markov_lookup, markov_transform_lookup = _preset_labels_and_lookup(fred_preset_df)
                selected_markov_label = st.selectbox(
                    "FRED series (defaults are recommended)",
                    options=markov_labels,
                    index=0,
                    key=f"markov_fred_preset_{MODEL_KEY}",
                )
                selected_markov_row = markov_lookup[selected_markov_label]
                default_markov_series = selected_markov_row["series_id"]
                custom_fred_series = st.text_input(
                    "Custom FRED series ID (optional)",
                    value="",
                    key=f"markov_custom_fred_series_{MODEL_KEY}",
                    help="Leave blank to use the selected default series.",
                ).strip().upper()
                fred_series_id = custom_fred_series or default_markov_series
                markov_transform = markov_transform_lookup.get(fred_series_id, "level")
            else:
                fred_series_id = st.text_input(
                    "FRED series ID",
                    value="UNRATE",
                    key=f"fred_series_{MODEL_KEY}",
                    help="Examples: UNRATE, INDPRO, GDPC1, PAYEMS.",
                ).strip().upper()
            if markov_transform != "level":
                st.caption(f"Applying '{markov_transform}' to values before state classification.")
            fred_col1, fred_col2 = st.columns(2)
            with fred_col1:
                fred_start = st.date_input(
                    "Start date",
                    value=pd.Timestamp("2000-01-01"),
                    key=f"fred_start_{MODEL_KEY}",
                )
            with fred_col2:
                fred_end = st.date_input(
                    "End date",
                    value=pd.Timestamp.today().date(),
                    key=f"fred_end_{MODEL_KEY}",
                )
            if st.button("Download FRED data", key=f"download_fred_{MODEL_KEY}"):
                try:
                    markov_df = download_fred_series(
                        fred_series_id,
                        start_date=str(fred_start),
                        end_date=str(fred_end),
                    )
                    markov_df = _apply_fred_transform(markov_df, markov_transform)
                    st.session_state[f"fred_markov_df_{MODEL_KEY}"] = markov_df
                except Exception as exc:
                    st.error(f"Could not download FRED data: {exc}")

            stored_fred_df = st.session_state.get(f"fred_markov_df_{MODEL_KEY}")
            if stored_fred_df is not None:
                markov_df = stored_fred_df

        if markov_df is not None:
            try:
                state_series = None
                if "state" in markov_df.columns:
                    state_series = markov_df["state"].to_numpy()
                    st.caption("Detected binary 'state' column for direct transition-matrix estimation.")
                elif "value" in markov_df.columns:
                    classify_choice = st.selectbox(
                        "Two-state classification rule",
                        ["Median split", "Manual threshold"],
                        key=f"classify_rule_{MODEL_KEY}",
                    )
                    threshold = None
                    if classify_choice == "Manual threshold":
                        threshold = st.number_input(
                            "Threshold for High state",
                            value=float(markov_df["value"].median()),
                            key=f"classify_threshold_{MODEL_KEY}",
                        )
                    state_series = classify_two_state_series(
                        markov_df["value"].to_numpy(),
                        method="threshold" if classify_choice == "Manual threshold" else "median",
                        threshold=threshold,
                    )
                else:
                    st.warning("CSV must include either a 'state' column or a 'value' column.")

                preview_cols = [col for col in ("date", "value", "state", "series_id") if col in markov_df.columns]
                if preview_cols:
                    st.markdown("**Empirical series preview**")
                    st.dataframe(markov_df[preview_cols].head(10), use_container_width=True)

                if state_series is not None:
                    estimated_P, estimated_transition_counts = estimate_transition_matrix_from_states(
                        state_series
                    )
                    st.session_state[estimated_p_key] = estimated_P
                    st.session_state[estimated_p_counts_key] = estimated_transition_counts
                    est_df = pd.DataFrame(
                        estimated_P,
                        index=["Low", "High"],
                        columns=["Low", "High"],
                    )
                    st.markdown("**Estimated transition matrix**")
                    st.dataframe(est_df.round(4), use_container_width=True)
                    st.markdown("**Transition counts**")
                    st.dataframe(estimated_transition_counts, use_container_width=True)
                    if markov_data_source == "Download from FRED" and "series_id" in markov_df.columns:
                        inferred_source = f"Estimated from FRED ({markov_df['series_id'].iloc[0]})"
                    else:
                        inferred_source = "Estimated from uploaded CSV"
                    st.session_state[estimated_p_source_key] = inferred_source
            except Exception as exc:
                st.error(f"Could not parse empirical Markov input: {exc}")

        source_col1, source_col2 = st.columns(2)
        if source_col1.button("Use manual P", key=f"use_manual_p_btn_{MODEL_KEY}"):
            st.session_state[use_estimated_p_key] = False
        if source_col2.button(
            "Use empirical P",
            key=f"use_empirical_p_btn_{MODEL_KEY}",
            disabled=st.session_state.get(estimated_p_key) is None,
        ):
            st.session_state[use_estimated_p_key] = True

        if st.session_state.get(estimated_p_key) is None:
            st.caption("No empirical transition matrix has been prepared yet.")

        current_source_label = (
            st.session_state.get(estimated_p_source_key)
            if st.session_state.get(use_estimated_p_key, False) and st.session_state.get(estimated_p_key) is not None
            else "Manual sliders"
        )
        st.markdown(f"**Active P source:** {current_source_label}")

    advanced_view = st.sidebar.radio(
        "Jump to advanced output",
        advanced_view_options,
        key=advanced_view_key,
    )
    st.sidebar.caption("Use the menu above to activate empirical inputs and jump to the advanced output you want to inspect.")
    st.sidebar.divider()

stored_estimated_P = st.session_state.get(estimated_p_key)
stored_estimated_P_source = st.session_state.get(estimated_p_source_key)
empirical_p_active = bool(st.session_state.get(use_estimated_p_key, False) and stored_estimated_P is not None)

# ---------------------------------------------------------------------------
# Sidebar -- shared parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Core Parameters")
beta = st.sidebar.slider("Discount factor (beta)", 0.80, 0.99, 0.95, 0.01)
with st.sidebar.expander("Estimate MY risk preferences", expanded=False):
    st.caption(
        "Estimate your CRRA risk aversion coefficient (sigma) from a short sequence of lottery choices "
        "and apply it to this model."
    )
    if st.button("Open estimator", key=f"open_crra_estimator_{MODEL_KEY}"):
        st.session_state["request_advanced_mode"] = True
        st.session_state[f"request_advanced_view_{MODEL_KEY}"] = "Risk preferences"
        st.rerun()
requested_sigma = st.session_state.get("request_sigma_slider")
if requested_sigma is not None:
    # Must happen before the sigma slider widget is instantiated.
    st.session_state["sigma_slider"] = float(np.clip(float(requested_sigma), 0.1, 10.0))
    st.session_state["request_sigma_slider"] = None
sigma = st.sidebar.slider(
    "Risk aversion (sigma)",
    0.1,
    10.0,
    2.0,
    0.1,
    key="sigma_slider",
)

# ---------------------------------------------------------------------------
# Sidebar -- Markov transition matrix (all models share this structure)
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Markov Transition Matrix")
if empirical_p_active:
    est_P = np.asarray(stored_estimated_P, dtype=float)
    st.sidebar.info(
        "Empirical transition matrix is active. Manual sliders are disabled until you switch back to manual P in the Advanced Menu."
    )
    st.sidebar.caption(f"Current source: {stored_estimated_P_source}")
    st.sidebar.caption(
        f"P(active) = [[{est_P[0, 0]:.2f}, {est_P[0, 1]:.2f}], [{est_P[1, 0]:.2f}, {est_P[1, 1]:.2f}]]"
    )
manual_p00 = st.sidebar.slider(
    "P(Low | Low)",
    0.50,
    0.99,
    0.90,
    0.01,
    disabled=empirical_p_active,
    key=f"manual_p00_{MODEL_KEY}",
)
manual_p11 = st.sidebar.slider(
    "P(High | High)",
    0.50,
    0.99,
    0.90,
    0.01,
    disabled=empirical_p_active,
    key=f"manual_p11_{MODEL_KEY}",
)
if not empirical_p_active:
    st.sidebar.caption(
        f"P(manual) = [[{manual_p00:.2f}, {1-manual_p00:.2f}], [{1-manual_p11:.2f}, {manual_p11:.2f}]]"
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

active_p00 = manual_p00
active_p11 = manual_p11
active_p_source = "Manual sliders"
estimated_P = None
estimated_transition_counts = None
stored_estimated_P = st.session_state.get(estimated_p_key)
stored_estimated_P_source = st.session_state.get(estimated_p_source_key)
empirical_p_active = bool(st.session_state.get(use_estimated_p_key, False) and stored_estimated_P is not None)
if empirical_p_active:
    active_p00 = float(np.asarray(stored_estimated_P, dtype=float)[0, 0])
    active_p11 = float(np.asarray(stored_estimated_P, dtype=float)[1, 1])
    active_p_source = stored_estimated_P_source or "Estimated empirically"


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
        active_p00, active_p11, d["a_min"], d["a_max"], n_a)
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
        active_p00, active_p11, d["k_min"], d["k_max"], n_k)
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
        active_p00, active_p11, d["a_min"], d["a_max"], _n_a, n_L, _inc)
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

if advanced_mode and fred_downloaded_bundle and not fred_preset_df.empty:
    fred_target_df, fred_target_preview, fred_empirical_moments = _build_fred_targets_from_bundle(
        fred_downloaded_bundle,
        fred_preset_df,
        income_key=income_key,
        model_key=MODEL_KEY,
    )

# Show convergence diagnostics
diag = result["diagnostics"]
st.success(
    f"VFI converged: {diag['converged']}  |  "
    f"Iterations: {diag['iterations']}  |  "
    f"Final sup-norm error: {diag['final_error']:.2e}"
)
if advanced_mode:
    st.caption(
        f"Active transition matrix source: {active_p_source} | "
        f"P = [[{P[0, 0]:.2f}, {P[0, 1]:.2f}], [{P[1, 0]:.2f}, {P[1, 1]:.2f}]]"
    )
    with st.expander("Advanced diagnostics", expanded=advanced_view == "Diagnostics"):
        _render_advanced_explanation("diagnostics", MODEL_KEY)
        st.markdown("**Built-in solver diagnostics**")
        st.dataframe(solver_diagnostics_summary(result), use_container_width=True)

        st.markdown("**Post-solve shape and feasibility checks**")
        st.dataframe(
            policy_shape_checks(result, MODEL_KEY).round(6),
            use_container_width=True,
        )

        edge_df = edge_usage_summary(result, MODEL_KEY)
        if not edge_df.empty:
            st.markdown("**Boundary usage**")
            st.dataframe(edge_df.round(4), use_container_width=True)

        grid_key, alt_sizes = _grid_variants(MODEL_KEY, params)
        if alt_sizes:
            st.caption(
                "Grid convergence runs are optional and may take longer, especially for Model 3 with assets."
            )
            if st.button("Run grid convergence checks", key=f"grid_diag_{MODEL_KEY}"):
                compare_results = {}
                for alt_size in alt_sizes:
                    alt_params = dict(params)
                    alt_params[grid_key] = int(alt_size)
                    compare_results[f"{grid_key}={alt_size}"] = _solve_model_from_params(
                        MODEL_KEY, alt_params
                    )
                grid_df = grid_convergence_check(result, compare_results, MODEL_KEY)
                if not grid_df.empty:
                    st.dataframe(grid_df.round(6), use_container_width=True)
                else:
                    st.info("No additional grid-comparison output is available for this model setup.")

    with st.expander(
        "Risk preference estimator (CRRA)",
        expanded=advanced_view == "Risk preferences",
    ):
        _render_advanced_explanation("risk", MODEL_KEY)
        st.caption(
            "Make choices between a sure payoff (Option A) and a risky payoff (Option B). "
            "Your answers update a posterior over CRRA risk aversion (sigma)."
        )

        pref_prefix = f"crra_{MODEL_KEY}"
        state_key = f"{pref_prefix}_state"

        col_a, col_b = st.columns(2)
        with col_a:
            worst = st.number_input(
                "Worst possible outcome",
                min_value=0.01,
                value=10.0,
                step=1.0,
                key=f"{pref_prefix}_worst",
            )
        with col_b:
            best = st.number_input(
                "Best possible outcome",
                min_value=0.02,
                value=30.0,
                step=1.0,
                key=f"{pref_prefix}_best",
            )

        viz_mode = st.selectbox(
            "Lottery probability display",
            ["Bar charts", "Pie charts"],
            index=0,
            key=f"{pref_prefix}_viz",
        )
        n_decisions = st.slider(
            "Decisions to make",
            5,
            50,
            20,
            1,
            key=f"{pref_prefix}_n",
        )
        sigma_max = st.slider(
            "Sigma search max",
            2.0,
            20.0,
            10.0,
            0.5,
            key=f"{pref_prefix}_sigma_max",
            help="Upper bound of the CRRA coefficient grid used for estimation.",
        )
        sensitivity = st.slider(
            "Choice sensitivity",
            2.0,
            20.0,
            8.0,
            0.5,
            key=f"{pref_prefix}_sensitivity",
            help="Higher means more deterministic choices; lower means noisier choices.",
        )

        if worst >= best:
            st.error("Worst outcome must be strictly less than best outcome.")
        else:
            settings = {
                "worst": float(worst),
                "best": float(best),
                "n": int(n_decisions),
                "sigma_max": float(sigma_max),
                "sensitivity": float(sensitivity),
            }

            def _reset_elicitation_state() -> None:
                step = 0.01
                sigma_grid = np.arange(0.0, float(sigma_max) + step / 2.0, step, dtype=float)
                logp = np.full_like(sigma_grid, -np.log(len(sigma_grid)), dtype=float)
                eps = 1e-6 * (float(best) - float(worst))
                sure_candidates = np.linspace(float(worst) + eps, float(best) - eps, 25)
                p_candidates = np.linspace(0.1, 0.9, 9)
                q = pick_next_question(
                    sigma_grid=sigma_grid,
                    logp=logp,
                    low=float(worst),
                    high=float(best),
                    p_candidates=p_candidates,
                    sure_candidates=sure_candidates,
                    sensitivity=float(sensitivity),
                )
                st.session_state[state_key] = {
                    "settings": settings,
                    "sigma_grid": sigma_grid,
                    "logp": logp,
                    "history": [],
                    "question": {"low": q.low, "high": q.high, "p_high": q.p_high, "sure": q.sure},
                }

            existing = st.session_state.get(state_key)
            if existing is not None and existing.get("settings") != settings:
                _reset_elicitation_state()
                st.info("Elicitation settings changed, so the elicitation state was reset.")

            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("Start / reset", key=f"{pref_prefix}_reset"):
                _reset_elicitation_state()
                st.rerun()

            state = st.session_state.get(state_key)
            if not state:
                st.caption("Click Start / reset to begin.")
            else:
                sigma_grid = np.asarray(state["sigma_grid"], dtype=float)
                logp = np.asarray(state["logp"], dtype=float)
                history = list(state.get("history", []))
                qd = state.get("question", {})
                q = ChoiceQuestion(
                    low=float(qd["low"]),
                    high=float(qd["high"]),
                    p_high=float(qd["p_high"]),
                    sure=float(qd["sure"]),
                )

                if history and btn_col2.button("Undo last", key=f"{pref_prefix}_undo"):
                    history = history[:-1]
                    logp = np.full_like(sigma_grid, -np.log(len(sigma_grid)), dtype=float)
                    for row in history:
                        prev_q = ChoiceQuestion(
                            low=float(row["low"]),
                            high=float(row["high"]),
                            p_high=float(row["p_high"]),
                            sure=float(row["sure"]),
                        )
                        logp = update_posterior_logp(
                            prior_logp=logp,
                            sigma_grid=sigma_grid,
                            question=prev_q,
                            chose_gamble=bool(row["choice"] == "B"),
                            sensitivity=float(sensitivity),
                        )
                    eps = 1e-6 * (float(best) - float(worst))
                    sure_candidates = np.linspace(float(worst) + eps, float(best) - eps, 25)
                    p_candidates = np.linspace(0.1, 0.9, 9)
                    next_q = pick_next_question(
                        sigma_grid=sigma_grid,
                        logp=logp,
                        low=float(worst),
                        high=float(best),
                        p_candidates=p_candidates,
                        sure_candidates=sure_candidates,
                        sensitivity=float(sensitivity),
                    )
                    state["history"] = history
                    state["logp"] = logp
                    state["question"] = {
                        "low": next_q.low,
                        "high": next_q.high,
                        "p_high": next_q.p_high,
                        "sure": next_q.sure,
                    }
                    st.session_state[state_key] = state
                    st.rerun()

                summary = posterior_summary(sigma_grid, logp)
                n_done = len(history)
                st.progress(min(1.0, n_done / int(n_decisions)))
                st.markdown(f"**Decisions recorded:** {n_done} / {int(n_decisions)}")
                st.caption(
                    f"Current posterior median sigma: {summary['median']:.2f} "
                    f"(95% CI [{summary['ci_lo']:.2f}, {summary['ci_hi']:.2f}])"
                )

                local_colors = _colors()

                def _lottery_chart(outcomes, probs, title: str):
                    if viz_mode == "Pie charts":
                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=[str(o) for o in outcomes],
                                    values=list(probs),
                                    hole=0.4,
                                    sort=False,
                                )
                            ],
                            layout=_layout(title=title),
                        )
                        fig.update_traces(textinfo="label+percent")
                        return fig
                    fig = go.Figure(layout=_layout(title=title, xaxis_title="Outcome", yaxis_title="Probability"))
                    fig.add_trace(
                        go.Bar(
                            x=[str(o) for o in outcomes],
                            y=list(probs),
                            marker_color=local_colors[1],
                            opacity=0.9,
                            text=[f"{p:.0%}" for p in probs],
                            textposition="outside",
                        )
                    )
                    fig.update_yaxes(range=[0, 1.05])
                    _style_axes(fig)
                    return fig

                if n_done < int(n_decisions):
                    st.markdown(f"### Decision {n_done + 1}")
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Option A (Sure payoff)**")
                        st.markdown(f"Receive **{q.sure:.2f}** with probability 100%.")
                        st.plotly_chart(
                            _lottery_chart([q.sure], [1.0], "Option A"),
                            width="stretch",
                            config=_PLOTLY_CFG,
                        )
                    with right:
                        st.markdown("**Option B (Lottery)**")
                        st.markdown(
                            f"Receive **{q.high:.2f}** with probability **{q.p_high:.0%}**, "
                            f"and **{q.low:.2f}** otherwise."
                        )
                        st.plotly_chart(
                            _lottery_chart([q.low, q.high], [1.0 - q.p_high, q.p_high], "Option B"),
                            width="stretch",
                            config=_PLOTLY_CFG,
                        )

                    choice = st.radio(
                        "Which do you prefer?",
                        ["Option A", "Option B"],
                        horizontal=True,
                        key=f"{pref_prefix}_choice",
                    )

                    if st.button("Submit choice", key=f"{pref_prefix}_submit"):
                        chose_b = choice == "Option B"
                        history.append(
                            {
                                "low": q.low,
                                "high": q.high,
                                "p_high": q.p_high,
                                "sure": q.sure,
                                "choice": "B" if chose_b else "A",
                            }
                        )
                        logp = update_posterior_logp(
                            prior_logp=logp,
                            sigma_grid=sigma_grid,
                            question=q,
                            chose_gamble=chose_b,
                            sensitivity=float(sensitivity),
                        )

                        eps = 1e-6 * (float(best) - float(worst))
                        sure_candidates = np.linspace(float(worst) + eps, float(best) - eps, 25)
                        p_candidates = np.linspace(0.1, 0.9, 9)
                        next_q = pick_next_question(
                            sigma_grid=sigma_grid,
                            logp=logp,
                            low=float(worst),
                            high=float(best),
                            p_candidates=p_candidates,
                            sure_candidates=sure_candidates,
                            sensitivity=float(sensitivity),
                        )

                        state["history"] = history
                        state["logp"] = logp
                        state["question"] = {
                            "low": next_q.low,
                            "high": next_q.high,
                            "p_high": next_q.p_high,
                            "sure": next_q.sure,
                        }
                        st.session_state[state_key] = state
                        st.rerun()

                else:
                    st.markdown("### Estimated CRRA (sigma)")
                    post_p = probs_from_logp(logp)
                    fig_post = go.Figure(
                        layout=_layout(
                            title="Posterior over sigma (CRRA risk aversion)",
                            xaxis_title="sigma",
                            yaxis_title="Posterior probability",
                        )
                    )
                    fig_post.add_trace(
                        go.Scatter(
                            x=sigma_grid,
                            y=post_p,
                            mode="lines",
                            line=dict(width=2.0, color=local_colors[0]),
                            name="p(sigma | choices)",
                        )
                    )
                    _style_axes(fig_post)
                    st.plotly_chart(fig_post, width="stretch", config=_PLOTLY_CFG)

                    st.markdown(
                        f"**Posterior median:** {summary['median']:.2f}  |  "
                        f"**Mean:** {summary['mean']:.2f}  |  "
                        f"**95% CI:** [{summary['ci_lo']:.2f}, {summary['ci_hi']:.2f}]"
                    )

                    benchmark = ChoiceQuestion(
                        low=float(worst),
                        high=float(best),
                        p_high=0.5,
                        sure=(float(worst) + float(best)) / 2.0,
                    )
                    ce = certainty_equivalent(benchmark, sigma_grid)
                    ev = 0.5 * float(worst) + 0.5 * float(best)
                    rp = ev - ce

                    hist, edges = np.histogram(rp, bins=40, weights=post_p)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    fig_rp = go.Figure(
                        layout=_layout(
                            title="Posterior over risk premium (benchmark 50/50 lottery)",
                            xaxis_title="Risk premium (EV - certainty equivalent)",
                            yaxis_title="Posterior probability (binned)",
                        )
                    )
                    fig_rp.add_trace(
                        go.Bar(x=centers, y=hist, marker_color=local_colors[2], opacity=0.9)
                    )
                    _style_axes(fig_rp)
                    st.plotly_chart(fig_rp, width="stretch", config=_PLOTLY_CFG)

                    rp_mean = float(np.sum(post_p * rp))
                    st.caption(f"Posterior mean risk premium (benchmark): {rp_mean:.3f}")

                    apply_col1, apply_col2 = st.columns(2)
                    clipped = float(np.clip(summary["median"], 0.1, 10.0))
                    if apply_col1.button(
                        "Apply posterior median to model sigma",
                        key=f"{pref_prefix}_apply_sigma",
                    ):
                        st.session_state["request_sigma_slider"] = clipped
                        st.rerun()
                    apply_col2.download_button(
                        "Download choices (CSV)",
                        data=pd.DataFrame(history).to_csv(index=False).encode("utf-8"),
                        file_name="crra_choices.csv",
                        mime="text/csv",
                        key=f"{pref_prefix}_dl",
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

if advanced_mode:
    with st.expander("Advanced regime-duration analytics", expanded=advanced_view == "Regime analytics"):
        _render_advanced_explanation("regime", MODEL_KEY)
        transitions = compute_transition_summary(sim["shock_idx"])
        spells = compute_regime_spells(sim["shock_idx"])
        spell_summary = summarize_spell_stats(spells, shock_labels)
        conditional_df = conditional_series_stats(sim, sim["shock_idx"], shock_labels)

        col_a, col_b = st.columns(2)
        col_a.markdown("**Empirical transition probabilities**")
        trans_prob = transitions["probabilities"].copy()
        trans_prob.index = shock_labels
        trans_prob.columns = shock_labels
        col_a.dataframe(trans_prob.round(4), use_container_width=True)

        col_b.markdown("**Empirical transition counts**")
        trans_counts = transitions["counts"].copy()
        trans_counts.index = shock_labels
        trans_counts.columns = shock_labels
        col_b.dataframe(trans_counts, use_container_width=True)

        st.markdown("**Spell duration summary**")
        st.dataframe(spell_summary.round(4), use_container_width=True)

        if not spell_summary.empty:
            fig_spell = go.Figure(layout=_layout(
                title="Average Regime Duration",
                xaxis_title="Shock state",
                yaxis_title="Average spell length",
            ))
            fig_spell.add_trace(go.Bar(
                x=spell_summary["state"],
                y=spell_summary["mean_duration"],
                marker_color=colors[: len(spell_summary)],
                text=[f"{v:.2f}" for v in spell_summary["mean_duration"]],
                textposition="outside",
            ))
            _style_axes(fig_spell)
            st.plotly_chart(fig_spell, width="stretch", config=_PLOTLY_CFG)

        st.markdown("**Conditional simulated moments by regime**")
        st.dataframe(conditional_df.round(4), use_container_width=True)

    with st.expander("Distribution lab", expanded=advanced_view == "Distribution lab"):
        _render_advanced_explanation("distribution", MODEL_KEY)
        replications = st.slider(
            "Monte Carlo replications",
            10,
            60,
            25,
            5,
            key=f"dist_replications_{MODEL_KEY}",
        )
        if grid is not None:
            init_mode = st.selectbox(
                "Initial-state sweep",
                ["Current initial state only", "Low / middle / high grid points"],
                key=f"dist_init_mode_{MODEL_KEY}",
            )
            if init_mode == "Low / middle / high grid points":
                initial_states = [
                    float(grid[0]),
                    float(grid[len(grid) // 2]),
                    float(grid[-1]),
                ]
            else:
                initial_states = [float(init_state)]
        else:
            initial_states = [float(init_state)]

        if st.button("Run distribution lab", key=f"run_dist_{MODEL_KEY}"):
            seeds = [int(seed) + i for i in range(replications)]
            dist_df = run_distributional_simulation(
                solver_result=result,
                shock_vals=shock_vals,
                P=P,
                initial_states=initial_states,
                seeds=seeds,
                model_name=sim_model_name,
                model_params=params,
                T_sim=int(T_sim),
            )
            dist_summary = summarize_distribution_runs(dist_df)
            st.markdown("**Cross-run summary**")
            st.dataframe(dist_summary.round(4), use_container_width=True)
            st.markdown("**Run-level outcomes**")
            st.dataframe(dist_df.round(4), use_container_width=True)

            metric_options = [
                col for col in dist_df.columns if col not in ("seed", "initial_state")
            ]
            if metric_options:
                metric_choice = st.selectbox(
                    "Histogram metric",
                    metric_options,
                    key=f"dist_metric_{MODEL_KEY}",
                )
                fig_hist = go.Figure(layout=_layout(
                    title=f"Distribution of {metric_choice}",
                    xaxis_title=metric_choice,
                    yaxis_title="Count",
                ))
                fig_hist.add_trace(go.Histogram(
                    x=dist_df[metric_choice],
                    nbinsx=20,
                    marker_color=colors[1],
                    opacity=0.85,
                ))
                _style_axes(fig_hist)
                st.plotly_chart(fig_hist, width="stretch", config=_PLOTLY_CFG)


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

if advanced_mode:
    with st.expander("Model-vs-data scorecard", expanded=advanced_view == "Scorecard"):
        _render_advanced_explanation("scorecard", MODEL_KEY)
        target_options = ["Download from FRED", "Illustrative defaults", "Upload CSV"]
        target_source = st.radio(
            "Target moments",
            target_options,
            horizontal=True,
            index=0,
            key=f"scorecard_target_source_{MODEL_KEY}",
        )
        if target_source == "Download from FRED":
            scorecard_bundle, _, _ = _render_fred_preset_controls(
                MODEL_KEY,
                fred_preset_df,
                key_prefix="scorecard",
            )
            scorecard_target_df, scorecard_target_preview, _ = _build_fred_targets_from_bundle(
                scorecard_bundle,
                fred_preset_df,
                income_key=income_key,
                model_key=MODEL_KEY,
            )
            if scorecard_target_df is not None and not scorecard_target_df.empty:
                target_df = scorecard_target_df
                st.caption("Using empirical target moments built from downloaded FRED defaults.")
                if scorecard_target_preview is not None:
                    st.markdown("**Aligned FRED sample used for moments**")
                    st.dataframe(scorecard_target_preview.head(12), use_container_width=True)
            else:
                target_df = get_default_targets(MODEL_KEY)
                st.caption("No downloaded FRED targets are available yet; illustrative defaults are shown below.")
        elif target_source == "Upload CSV":
            target_upload = st.file_uploader(
                "Upload target moments CSV",
                type="csv",
                key=f"scorecard_upload_{MODEL_KEY}",
                help="Expected columns: moment,target and optional weight.",
            )
            if target_upload is not None:
                target_df = targets_from_dataframe(pd.read_csv(target_upload))
            else:
                target_df = get_default_targets(MODEL_KEY)
                st.caption("No CSV uploaded yet, so the illustrative defaults are shown below.")
        else:
            target_df = get_default_targets(MODEL_KEY)

        scorecard_df, score_value = build_moment_scorecard(moments, target_df)
        st.metric("Weighted loss", f"{score_value:.4f}")
        st.markdown("**Target moments in use**")
        st.dataframe(target_df.round(4), use_container_width=True)
        st.markdown("**Model-vs-target scorecard**")
        st.dataframe(scorecard_df.round(4), use_container_width=True)


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

if advanced_mode:
    with st.expander("Advanced phase diagrams & policy surfaces", expanded=advanced_view == "Phase diagrams"):
        _render_advanced_explanation("phase", MODEL_KEY)
        if grid is not None:
            fig_phase = make_phase_diagram(
                MODEL_KEY, result, np.asarray(grid, dtype=float), shock_labels,
                colors, _layout, _style_axes,
            )
            st.plotly_chart(fig_phase, width="stretch", config=_PLOTLY_CFG)

            fig_value_surface = make_value_surface(
                result, np.asarray(grid, dtype=float), shock_labels,
                _layout, _style_axes,
            )
            st.plotly_chart(fig_value_surface, width="stretch", config=_PLOTLY_CFG)

            for idx, fig_surface in enumerate(
                make_policy_surfaces(
                    MODEL_KEY, result, np.asarray(grid, dtype=float), shock_labels,
                    _layout, _style_axes,
                ),
                start=1,
            ):
                st.plotly_chart(
                    fig_surface,
                    width="stretch",
                    config=_PLOTLY_CFG,
                    key=f"advanced_surface_{MODEL_KEY}_{idx}",
                )
        else:
            figs = make_policy_surfaces(
                MODEL_KEY, result, np.array([0.0, 1.0]), shock_labels,
                _layout, _style_axes,
            )
            for idx, fig_surface in enumerate(figs, start=1):
                st.plotly_chart(
                    fig_surface,
                    width="stretch",
                    config=_PLOTLY_CFG,
                    key=f"advanced_surface_labor_only_{idx}",
                )

    st.header("7. Advanced Analysis")
    if advanced_view == "Welfare":
        st.session_state[advanced_analysis_view_key] = "Welfare"
    elif advanced_view == "Calibration":
        st.session_state[advanced_analysis_view_key] = "Calibration Lab"

    advanced_analysis_view = st.radio(
        "Advanced analysis area",
        ["Welfare", "Calibration Lab"],
        horizontal=True,
        key=advanced_analysis_view_key,
    )

    if advanced_analysis_view == "Welfare":
        st.caption("Welfare outputs are optional and remain model-specific.")
        _render_advanced_explanation("welfare", MODEL_KEY)
        welfare_summary, welfare_detail = summarize_state_welfare(
            result=result,
            model_key=MODEL_KEY,
            beta=beta,
            sigma=sigma,
            shock_labels=shock_labels,
        )
        st.markdown("**Current-model welfare summary**")
        st.dataframe(welfare_summary.round(4), use_container_width=True)
        if not welfare_detail.empty:
            st.markdown("**Sampled state-level welfare values**")
            st.dataframe(welfare_detail.round(4), use_container_width=True)

        with st.expander("Optional counterfactual welfare comparison", expanded=False):
            bounds = parameter_bounds_for_model(
                MODEL_KEY,
                include_assets=params.get("include_assets", True),
            )
            if bounds:
                default_param = list(bounds)[0]
                counter_param = st.selectbox(
                    "Counterfactual parameter",
                    list(bounds),
                    index=0,
                    key=f"welfare_param_{MODEL_KEY}",
                )
                lower, upper = bounds[counter_param]
                current_value = float(params[counter_param])
                alt_value = st.slider(
                    f"Alternative {counter_param}",
                    float(lower),
                    float(upper),
                    float(np.clip(current_value, lower, upper)),
                    float((upper - lower) / 50.0),
                    key=f"welfare_alt_value_{MODEL_KEY}",
                )
                if st.button("Solve counterfactual", key=f"solve_welfare_cf_{MODEL_KEY}"):
                    alt_params = dict(params)
                    alt_params[counter_param] = float(alt_value)
                    alt_result = _solve_model_from_params(MODEL_KEY, alt_params)
                    welfare_cmp = compare_counterfactual_welfare(
                        result,
                        alt_result,
                        MODEL_KEY,
                        shock_labels,
                    )
                    st.dataframe(welfare_cmp.round(6), use_container_width=True)
            else:
                st.info("No counterfactual welfare controls are defined for this model configuration.")

    else:
        st.caption(
            "Experimental moment-matching lab. It is button-triggered and evaluation-limited to preserve demo readiness."
        )
        _render_advanced_explanation("calibration", MODEL_KEY)
        calibration_target_options = ["Download from FRED", "Illustrative defaults", "Upload CSV"]
        cal_target_source = st.radio(
            "Calibration targets",
            calibration_target_options,
            horizontal=True,
            index=0,
            key=f"cal_target_source_{MODEL_KEY}",
        )
        if cal_target_source == "Download from FRED":
            calibration_bundle, _, _ = _render_fred_preset_controls(
                MODEL_KEY,
                fred_preset_df,
                key_prefix="calibration",
            )
            calibration_target_df, _, _ = _build_fred_targets_from_bundle(
                calibration_bundle,
                fred_preset_df,
                income_key=income_key,
                model_key=MODEL_KEY,
            )
            if calibration_target_df is not None and not calibration_target_df.empty:
                calibration_targets = calibration_target_df
                st.caption("Using downloaded FRED defaults as calibration targets.")
            else:
                calibration_targets = get_default_targets(MODEL_KEY)
                st.caption("No downloaded FRED targets are available yet; illustrative defaults are being used.")
        elif cal_target_source == "Upload CSV":
            cal_target_upload = st.file_uploader(
                "Upload calibration target CSV",
                type="csv",
                key=f"calibration_targets_{MODEL_KEY}",
            )
            if cal_target_upload is not None:
                calibration_targets = targets_from_dataframe(pd.read_csv(cal_target_upload))
            else:
                calibration_targets = get_default_targets(MODEL_KEY)
                st.caption("No custom targets uploaded yet, so the illustrative defaults are being used.")
        else:
            calibration_targets = get_default_targets(MODEL_KEY)

        bounds = parameter_bounds_for_model(
            MODEL_KEY,
            include_assets=params.get("include_assets", True),
        )
        candidate_params = list(bounds)
        recommended_params = {
            "model1": ["beta", "r"],
            "model2": ["beta", "delta"],
            "model3": ["psi", "nu"],
        }.get(MODEL_KEY, candidate_params[:1])
        recommended_params = [name for name in recommended_params if name in candidate_params]
        selected_params = st.multiselect(
            "Parameters to calibrate (recommended: one or two)",
            candidate_params,
            default=recommended_params[:1] if recommended_params else candidate_params[:1],
            key=f"calibration_params_{MODEL_KEY}",
        )
        if len(selected_params) > 2:
            st.warning("To keep runtime manageable, only the first two selected parameters will be used.")
            selected_params = selected_params[:2]

        max_evals = 8 if MODEL_KEY == "model3" and params.get("include_assets", True) else 12
        n_evals = st.slider(
            "Maximum evaluations",
            4,
            max_evals,
            min(8, max_evals),
            1,
            key=f"calibration_evals_{MODEL_KEY}",
        )
        cal_T = st.slider(
            "Simulation periods for calibration objective",
            50,
            300,
            int(min(100, T_sim)),
            10,
            key=f"calibration_T_{MODEL_KEY}",
        )
        st.markdown("**Targets in use**")
        st.dataframe(calibration_targets.round(4), use_container_width=True)

        if st.button("Run structural calibration", key=f"run_calibration_{MODEL_KEY}"):
            def _evaluate_candidate(candidate):
                candidate_result = _solve_model_from_params(MODEL_KEY, candidate)
                candidate_shock_vals = _active_shock_values(MODEL_KEY, candidate)
                candidate_P = np.asarray(candidate["P"], dtype=float)
                candidate_sim = simulate_model(
                    candidate_result,
                    candidate_shock_vals,
                    candidate_P,
                    init_state,
                    T_sim=int(cal_T),
                    seed=int(seed),
                    model_name=sim_model_name,
                    model_params=candidate,
                )
                candidate_moments = compute_moments(
                    {k: v for k, v in candidate_sim.items() if k != "shock_idx"},
                    income_key=income_key,
                )
                objective, candidate_scorecard = objective_from_moments(
                    candidate_moments,
                    calibration_targets,
                )
                return objective, {
                    "matched_moments": int(candidate_scorecard["model"].notna().sum()),
                }

            calibration_results = run_structural_calibration(
                base_params=dict(params),
                selected_params=selected_params,
                bounds={name: bounds[name] for name in selected_params},
                n_evals=int(n_evals),
                evaluate_candidate=_evaluate_candidate,
                random_seed=int(seed),
            )
            st.markdown("**Calibration search results**")
            st.dataframe(calibration_results.round(6), use_container_width=True)
            if not calibration_results.empty:
                best_row = calibration_results.iloc[0]
                st.metric("Best objective", f"{best_row['objective']:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "ECO 317 -- Intermediate Macroeconomic Theory | Spring 2026 | "
    "AI-Assisted Macroeconomic Modeling Dashboard"
)
