from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_autorefresh import st_autorefresh

from finrec.config import load_config
from finrec.datasets.merge import build_merged_dataset
from finrec.jobs.runner import JobRunner
from finrec.providers.registry import get_registry
from finrec.providers.utils.optional import require_optional
from finrec.storage.sqlite import SQLiteStorage
from finrec.ui.pipelines import submit_provider_fetch, submit_recipe_run
from finrec.ui.timeseries_catalog import build_econ_options, build_finance_options
from finrec.viz.plotly_timeseries import build_recession_intervals, plot_timeseries


def _get_singletons():
    """
    Streamlit reruns scripts; keep the storage + runner stable in session_state.
    """
    if "finrec_config" not in st.session_state:
        st.session_state.finrec_config = load_config()

    cfg = st.session_state.finrec_config

    if "finrec_storage" not in st.session_state:
        storage = SQLiteStorage(cfg.db_path)
        storage.init_schema()
        st.session_state.finrec_storage = storage

    if "finrec_runner" not in st.session_state:
        st.session_state.finrec_runner = JobRunner(st.session_state.finrec_storage, cfg.results_dir)

    if "active_dataset_path" not in st.session_state:
        st.session_state.active_dataset_path = None
    if "dataset_builder_items" not in st.session_state:
        st.session_state.dataset_builder_items = {}
    if "dataset_builder_merge_job_id" not in st.session_state:
        st.session_state.dataset_builder_merge_job_id = None

    if "chart_filter_range" not in st.session_state:
        st.session_state.chart_filter_range = True
    if "chart_log_scale" not in st.session_state:
        st.session_state.chart_log_scale = False
    if "chart_secondary_axis" not in st.session_state:
        st.session_state.chart_secondary_axis = False
    if "chart_secondary_cols_raw" not in st.session_state:
        st.session_state.chart_secondary_cols_raw = ""
    if "chart_recession_shading" not in st.session_state:
        st.session_state.chart_recession_shading = False
    if "usrec_job_id" not in st.session_state:
        st.session_state.usrec_job_id = None
    if "usrec_intervals" not in st.session_state:
        st.session_state.usrec_intervals = None

    if "recent_finance_series" not in st.session_state:
        st.session_state.recent_finance_series = []
    if "recent_econ_series" not in st.session_state:
        st.session_state.recent_econ_series = []
    if "custom_finance_series_raw" not in st.session_state:
        st.session_state.custom_finance_series_raw = ""
    if "custom_econ_series_raw" not in st.session_state:
        st.session_state.custom_econ_series_raw = ""

    # News UI defaults
    if "news_fetch_enabled" not in st.session_state:
        st.session_state.news_fetch_enabled = True
    if "news_show_results" not in st.session_state:
        st.session_state.news_show_results = True
    if "news_language_choice" not in st.session_state:
        # "Any" | "english" | ... | "Custom"
        st.session_state.news_language_choice = "Any"
    if "news_language_custom_raw" not in st.session_state:
        st.session_state.news_language_custom_raw = ""

    # Refresh defaults
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = True

    return st.session_state.finrec_config, st.session_state.finrec_storage, st.session_state.finrec_runner


def _parse_language_list(raw: str) -> list[str]:
    """
    Parse a comma-separated list of languages for GDELT's `sourcelang:` filter.
    Example: "english, spanish"
    """
    items = [x.strip().lower() for x in (raw or "").split(",")]
    return [x for x in items if x]


def _df_with_rank(df: pd.DataFrame, *, start: int = 1) -> pd.DataFrame:
    """Add a visible 1-based rank column for display without mutating df."""
    out = df.copy()
    out.insert(0, "#", range(start, start + len(out)))
    return out


def _cols_with_visibility_selector(
    cols: list[str],
    *,
    max_visible: int = 3,
    displayed_count: int = 2,
    session_key: str,
    label: str = "Series shown in main chart",
) -> tuple[list[str], list[str]]:
    """
    If <= max_visible columns, return (cols, []).
    If > max_visible, let user choose displayed_count columns to show; return (chosen, remaining).
    """
    cols = [c for c in (cols or []) if c]
    if len(cols) <= max_visible:
        return cols, []

    default = cols[:displayed_count]
    chosen = st.multiselect(
        label,
        options=cols,
        default=default,
        max_selections=displayed_count,
        key=session_key,
        help=f"If more than {max_visible} series are selected, show only {displayed_count} in the main chart to save space.",
    )
    if not chosen:
        chosen = default[:1]
    remaining = [c for c in cols if c not in set(chosen)]
    return chosen, remaining


def _parse_csv_list(raw: str) -> list[str]:
    items = [x.strip().upper() for x in (raw or "").split(",")]
    return [x for x in items if x]


def _update_recent_list(recent: list[str], used_ids: list[str], max_len: int = 20) -> None:
    """Move used_ids to front of recent list (LRU); cap at max_len."""
    for uid in reversed(used_ids):
        uid = uid.strip().upper()
        if not uid:
            continue
        if uid in recent:
            recent.remove(uid)
        recent.insert(0, uid)
    while len(recent) > max_len:
        recent.pop()


def _auto_news_query_finance(tickers: list[str]) -> str:
    # Minimal, robust default: OR tickers. Add a few common-name expansions.
    if not tickers:
        return "markets"
    name_map = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "GOOG": "Google",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "NVDA": "Nvidia",
        "META": "Meta",
        "SPY": "S&P 500",
        "QQQ": "Nasdaq",
        "DIA": "Dow",
    }
    terms: list[str] = []
    for t in tickers:
        terms.append(t)
        if t in name_map:
            terms.append(name_map[t])
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            # Quote multi-word terms for GDELT query syntax.
            out.append(f"\"{term}\"" if any(ch.isspace() for ch in term) else term)
    return "(" + " OR ".join(out) + ")"


def _auto_news_query_econ(series_ids: list[str]) -> str:
    if not series_ids:
        return "economy"
    series_map = {
        "CPIAUCSL": "inflation",
        "CPILFESL": "core inflation",
        "PCEPI": "PCE inflation",
        "UNRATE": "unemployment",
        "FEDFUNDS": "fed funds rate",
        "GDP": "GDP",
        "GDPC1": "real GDP",
        "PAYEMS": "payrolls",
        "DGS10": "10-year Treasury",
    }
    terms: list[str] = []
    for sid in series_ids:
        terms.append(sid)
        if sid in series_map:
            terms.append(series_map[sid])
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            out.append(f"\"{term}\"" if any(ch.isspace() for ch in term) else term)
    return "(" + " OR ".join(out) + ")"


def _job_by_id(storage: SQLiteStorage, job_id: str):
    jobs = storage.list_jobs(limit=1000)
    for j in jobs:
        if j.job_id == job_id:
            return j
    return None


def _artifact_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _download_df(df: pd.DataFrame, *, label: str, file_name: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


def _parse_cols(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").split(",") if c.strip()]


def _preferred_market_provider(market_provider_ids: list[str]) -> str | None:
    """
    Pick a sensible default market provider.

    Prefer FMP when an API key is set (more reliable than Yahoo rate limits).
    """
    ids = [str(x) for x in (market_provider_ids or [])]
    if "fmp" in ids and (os.getenv("FINREC_FMP_API_KEY") or "").strip():
        return "fmp"
    if "yfinance" in ids:
        return "yfinance"
    return ids[0] if ids else None


def _filter_df_by_date_range(df: pd.DataFrame, *, x_col: str, start_date: date, end_date: date) -> pd.DataFrame:
    if x_col not in df.columns:
        return df
    x = pd.to_datetime(df[x_col], errors="coerce")
    if x.notna().sum() == 0:
        return df
    mask = (x.dt.date >= start_date) & (x.dt.date <= end_date)
    return df.loc[mask].copy()


def _split_cols_for_secondary_axis_by_scale(
    df: pd.DataFrame,
    cols: list[str],
    *,
    ratio_threshold: float = 50.0,
) -> tuple[list[str], list[str]]:
    """
    Heuristic: when series are on wildly different scales (e.g., CPI ~ 250 vs ICSA ~ 200,000),
    plotting on one axis makes the small series look "missing". We auto-assign small series to
    a secondary axis when the median magnitude ratio is large.
    """
    mags: dict[str, float] = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        mags[c] = float(s.abs().median())

    if len(mags) < 2:
        kept = [c for c in cols if c in mags]
        return kept, []

    base_col = max(mags, key=mags.get)
    base_mag = mags.get(base_col, 0.0)
    left: list[str] = [base_col]
    right: list[str] = []

    for c in cols:
        if c == base_col or c not in mags:
            continue
        m = mags[c]
        if base_mag > 0 and m > 0 and (base_mag / m) >= ratio_threshold:
            right.append(c)
        else:
            left.append(c)

    return left, right


def _plot_df(
    df: pd.DataFrame,
    *,
    x_col: str,
    left_cols: list[str],
    right_cols_override: list[str] | None = None,
    title: str = "",
    y_left_label: str = "",
    y_right_label: str = "",
    height: int = 300,
    recession_intervals=None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    cols_left = [c for c in left_cols if c in df.columns]
    if x_col not in df.columns or not cols_left:
        st.caption("No chartable columns found.")
        return

    df2 = df
    if st.session_state.get("chart_filter_range") and start_date is not None and end_date is not None:
        df2 = _filter_df_by_date_range(df2, x_col=x_col, start_date=start_date, end_date=end_date)

    right_cols: list[str] = []
    if right_cols_override is not None:
        right_cols = [c for c in right_cols_override if c in df2.columns and c not in cols_left]
    elif st.session_state.get("chart_secondary_axis"):
        want = _parse_cols(st.session_state.get("chart_secondary_cols_raw", ""))
        right_cols = [c for c in want if c in df2.columns and c not in cols_left]

    fig = plot_timeseries(
        df2,
        left_cols=cols_left,
        right_cols=right_cols or None,
        title=title,
        x_col=x_col,
        y_left_label=y_left_label,
        y_right_label=y_right_label,
        recession_intervals=recession_intervals,
        log_y_left=bool(st.session_state.get("chart_log_scale")),
        log_y_right=bool(st.session_state.get("chart_log_scale")),
    )
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def _plot_lp_irf(df_out: pd.DataFrame, *, height: int = 250, title: str = "Local projection IRF") -> None:
    """
    Plot LP-IRF output with CI band using Plotly.

    Expects columns: horizon, irf, ci_low, ci_high.
    """
    required = ["horizon", "irf", "ci_low", "ci_high"]
    missing = [c for c in required if c not in df_out.columns]
    if missing:
        st.caption(f"LP-IRF output missing columns: {missing}")
        return

    go = require_optional("plotly.graph_objects", extra_hint="dev")

    h = pd.to_numeric(df_out["horizon"], errors="coerce")
    irf = pd.to_numeric(df_out["irf"], errors="coerce")
    lo = pd.to_numeric(df_out["ci_low"], errors="coerce")
    hi = pd.to_numeric(df_out["ci_high"], errors="coerce")

    d = pd.DataFrame({"horizon": h, "irf": irf, "ci_low": lo, "ci_high": hi}).dropna(subset=["horizon"])
    if d.empty:
        st.caption("No chartable LP-IRF rows found.")
        return
    d = d.sort_values("horizon")

    custom = pd.concat([d["ci_low"].rename("lo"), d["ci_high"].rename("hi")], axis=1).to_numpy()

    fig = go.Figure()
    # Confidence band (draw low first, then fill to high)
    fig.add_trace(
        go.Scatter(
            x=d["horizon"],
            y=d["ci_low"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
            name="CI low",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["horizon"],
            y=d["ci_high"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.20)",
            showlegend=True,
            name="CI band",
            hoverinfo="skip",
        )
    )
    # IRF line
    fig.add_trace(
        go.Scatter(
            x=d["horizon"],
            y=d["irf"],
            mode="lines",
            line=dict(color="rgb(31, 119, 180)", width=2),
            name="IRF",
            customdata=custom,
            hovertemplate="Horizon: %{x}<br>Response: %{y:.6f}<br>CI: [%{customdata[0]:.6f}, %{customdata[1]:.6f}]<extra></extra>",
        )
    )

    fig.update_layout(
        title=title or None,
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=50 if title else 20, b=40),
        height=height,
    )
    fig.update_xaxes(title_text="Horizon")
    fig.update_yaxes(title_text="Response")

    st.plotly_chart(fig, use_container_width=True)


def _wide_from_market_artifacts(
    dfs_by_symbol: dict[str, pd.DataFrame],
    *,
    date_col: str = "date",
    value_col: str = "close",
    how: str = "outer",
) -> pd.DataFrame:
    """
    Convert multiple per-symbol market artifacts into a wide dataframe:
      date | AAPL_close | MSFT_close | ...
    """
    out: pd.DataFrame | None = None
    for sym, df in (dfs_by_symbol or {}).items():
        if df is None or df.empty or date_col not in df.columns:
            continue
        if value_col not in df.columns:
            continue
        tmp = df[[date_col, value_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        tmp = tmp.rename(columns={value_col: f"{sym}_{value_col}"})
        if out is None:
            out = tmp
        else:
            out = out.merge(tmp, on=date_col, how=how)
    if out is None:
        return pd.DataFrame()
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def _wide_from_macro_artifacts(
    dfs_by_series: dict[str, pd.DataFrame],
    *,
    date_col: str = "date",
    value_col: str = "value",
    how: str = "outer",
) -> pd.DataFrame:
    """
    Convert multiple per-series macro artifacts into a wide dataframe:
      date | CPIAUCSL | UNRATE | ...

    Series with different frequencies (e.g., weekly ICSA, monthly PAYEMS) produce NaNs
    after merge. We forward-fill each value column so the combined chart renders
    continuous lines instead of fragmented segments.
    """
    out: pd.DataFrame | None = None
    for sid, df in (dfs_by_series or {}).items():
        if df is None or df.empty or date_col not in df.columns:
            continue
        if value_col not in df.columns:
            continue
        tmp = df[[date_col, value_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        tmp = tmp.rename(columns={value_col: str(sid)})
        if out is None:
            out = tmp
        else:
            out = out.merge(tmp, on=date_col, how=how)
    if out is None:
        return pd.DataFrame()
    out = out.sort_values(date_col).reset_index(drop=True)
    # Forward-fill value columns so mixed-frequency series plot as continuous lines
    value_cols = [c for c in out.columns if c != date_col]
    for c in value_cols:
        out[c] = out[c].ffill()
    return out


def _dataset_builder_ui(
    *,
    cfg,
    storage: SQLiteStorage,
    runner: JobRunner,
    reg,
    start_date: date,
    end_date: date,
    default_fred_series: list[str],
    default_tickers: list[str],
):
    st.subheader("Dataset Builder")

    active_path = st.session_state.get("active_dataset_path")
    if active_path:
        st.success(f"Active merged dataset: `{active_path}`")
        try:
            df_active = _artifact_df(active_path)
            with st.expander("Preview active merged dataset", expanded=False):
                _download_df(df_active, label="Download active merged dataset CSV", file_name=Path(active_path).name)
                st.dataframe(df_active.tail(200), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load active merged dataset: {e}")
    else:
        st.info("No active merged dataset yet. Build one below to enable OLS/Taylor in Econ mode.")

    macro_provider_options = [p.meta.id for p in reg.list("macro")]
    market_provider_options = [p.meta.id for p in reg.list("market")]

    default_macro_provider = (
        "fred" if any(pid == "fred" for pid in macro_provider_options) else (macro_provider_options[0] if macro_provider_options else None)
    )
    default_market_provider = (
        _preferred_market_provider(market_provider_options)
    )

    with st.expander("Build / merge a dataset", expanded=False):
        left, right = st.columns(2)
        with left:
            _macro_opts = macro_provider_options or ["(no macro provider — install pandas-datareader)"]
            macro_provider_id = st.selectbox(
                "FRED provider_id",
                options=_macro_opts,
                index=(macro_provider_options.index(default_macro_provider) if default_macro_provider and default_macro_provider in macro_provider_options else 0),
                help="Select which macro data provider to use for fetching FRED-style series.",
            )
            fred_series_raw = st.text_input(
                "FRED series_id(s) for dataset (comma-separated)",
                value=",".join(default_fred_series) if default_fred_series else "CPIAUCSL",
                help="Example: CPIAUCSL, UNRATE, DGS10. These will be fetched and then merged into one dataset.",
            )
            fred_series_ids = [x.strip().upper() for x in fred_series_raw.split(",") if x.strip()]
            fred_items = [f"{macro_provider_id}:{sid}" for sid in fred_series_ids]
            selected_fred = st.multiselect(
                "Select FRED series to fetch (provider_id:series_id)",
                options=fred_items,
                default=fred_items,
                help="Choose which macro series to fetch for the dataset builder merge.",
            )

        with right:
            _market_opts = market_provider_options or ["(no market provider — install yfinance)"]
            market_provider_id = st.selectbox(
                "Market provider_id",
                options=_market_opts,
                index=(market_provider_options.index(default_market_provider) if default_market_provider and default_market_provider in market_provider_options else 0),
                help="Select which market data provider to use for fetching price series.",
            )
            tickers_raw = st.text_input(
                "Tickers for dataset (comma-separated)",
                value=",".join(default_tickers) if default_tickers else "",
                help="Example: AAPL, MSFT, SPY. These will be fetched and merged alongside macro series.",
            )
            tickers = _parse_csv_list(tickers_raw)
            selected_tickers = st.multiselect(
                "Select tickers to fetch",
                options=tickers,
                default=tickers,
                help="Pick which tickers to include in the dataset builder run.",
            )
            if market_provider_id == "fmp":
                st.caption("FMP provider currently supports daily bars only (interval=1d).")
                if selected_tickers:
                    st.warning(
                        "**FMP free API:** Does not cover all tickers. Unsupported symbols fall back to Yahoo Finance "
                        "(easily rate-limited). Avoid pulling large datasets quickly for symbols outside your FMP plan."
                    )
                market_interval = "1d"
            else:
                market_interval = st.selectbox(
                    "Market interval",
                    ["1d", "1h", "30m", "15m", "5m"],
                    index=0,
                    help="Higher-frequency intervals can be slower and may be limited by the provider.",
                )
            market_value_col = st.selectbox(
                "Market value column (for merge)",
                ["close", "open", "high", "low", "volume"],
                index=0,
                help="Which column from the market artifact should be used when merging into the dataset.",
            )

        merge_how = st.selectbox(
            "Merge how",
            ["outer", "inner"],
            index=0,
            help="Outer keeps all dates (more missing values). Inner keeps only overlapping dates (fewer rows).",
        )
        ffill = st.checkbox(
            "Forward-fill after merge",
            value=False,
            help="Forward-fill missing values after merge (useful for mixed frequencies).",
        )

        fetch_btn = st.button("Fetch selected series for dataset")
        if fetch_btn:
            if macro_provider_id.startswith("(no ") or market_provider_id.startswith("(no "):
                st.error(
                    "Install the required packages (pandas-datareader for FRED, and a market provider via `.[market]`) and restart the app."
                )
            else:
                items: dict[str, dict] = {}

                for spec in selected_fred:
                    if ":" not in spec:
                        continue
                    provider_id, series_id = spec.split(":", 1)
                    req = {
                        "series_id": series_id,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "n": 120,
                    }
                    job_id = submit_provider_fetch(runner=runner, kind="macro", provider_id=provider_id, request=req)
                    key = f"{provider_id}:{series_id}"
                    items[key] = {
                        "kind": "macro",
                        "provider_id": provider_id,
                        "series_id": series_id,
                        "job_id": job_id,
                        "date_col": "date",
                        "value_col": "value",
                        "alias": series_id,
                    }

                for t in selected_tickers:
                    req = {
                        "symbol": t,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "interval": market_interval,
                        "n": 252,
                    }
                    job_id = submit_provider_fetch(runner=runner, kind="market", provider_id=market_provider_id, request=req)
                    key = f"{market_provider_id}:{t}"
                    items[key] = {
                        "kind": "market",
                        "provider_id": market_provider_id,
                        "ticker": t,
                        "job_id": job_id,
                        "date_col": "date",
                        "value_col": market_value_col,
                        "alias": f"{t}_{market_value_col}",
                    }

                st.session_state.dataset_builder_items = items
                st.session_state.dataset_builder_merge_job_id = None
                st.success("Submitted dataset fetch jobs. Click Refresh results to watch them finish.")

        items = st.session_state.get("dataset_builder_items") or {}
        if items:
            jobs_by_id = {j.job_id: j for j in storage.list_jobs(limit=2000)}
            rows: list[dict] = []
            ready = True
            for k, spec in items.items():
                jid = spec.get("job_id")
                j = jobs_by_id.get(jid) if jid else None
                status = j.status if j else "UNKNOWN"
                out_path = j.output_path if j else None
                ready = ready and bool(out_path) and status == "SUCCEEDED"
                rows.append(
                    {
                        "series": k,
                        "job_id": jid,
                        "status": status,
                        "output_path": out_path,
                        "alias": spec.get("alias"),
                        "value_col": spec.get("value_col"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            merge_btn = st.button("Merge fetched artifacts into a dataset", disabled=not ready)
            if merge_btn:
                jobs_by_id = {j.job_id: j for j in storage.list_jobs(limit=4000)}
                inputs: list[dict] = []
                for _k, spec in items.items():
                    jid = spec["job_id"]
                    j = jobs_by_id.get(jid)
                    if not j or not j.output_path:
                        raise ValueError(f"Missing artifact for job_id={jid}")
                    inputs.append(
                        {
                            "path": j.output_path,
                            "date_col": spec.get("date_col", "date"),
                            "value_col": spec["value_col"],
                            "alias": spec["alias"],
                        }
                    )

                merge_request = {"inputs": inputs, "merge_how": merge_how, "ffill": ffill}

                def _merge_job_fn(ctx):
                    ctx.log("INFO", f"Merging {len(inputs)} inputs: how={merge_how}, ffill={ffill}")
                    df_out = build_merged_dataset(inputs=inputs, merge_how=merge_how, ffill=ffill, log=ctx.log)
                    ctx.log("INFO", f"Merged dataset shape={df_out.shape}")
                    return df_out

                merge_job_id = runner.submit(
                    kind="dataset_merge",
                    provider_kind="dataset",
                    provider_id="merged_dataset",
                    request=merge_request,
                    fn=_merge_job_fn,
                )
                st.session_state.dataset_builder_merge_job_id = merge_job_id
                st.success("Submitted merge job. Click Refresh results to watch it finish.")

        merge_job_id = st.session_state.get("dataset_builder_merge_job_id")
        if merge_job_id:
            jobs_by_id = {j.job_id: j for j in storage.list_jobs(limit=4000)}
            jm = jobs_by_id.get(merge_job_id)
            if jm:
                st.write({"merge_job_id": merge_job_id, "status": jm.status, "output_path": jm.output_path})
                if jm.status == "FAILED":
                    st.error("Dataset merge failed. See job logs in Results sections for details.")
                if jm.output_path and jm.status == "SUCCEEDED":
                    st.session_state.active_dataset_path = jm.output_path


def _render_query_and_submit(
    *,
    mode: str,
    cfg,
    storage: SQLiteStorage,
    runner: JobRunner,
    reg,
    start_date: date,
    end_date: date,
) -> None:
    tickers: list[str] = []
    series_ids: list[str] = []

    if mode == "Finance":
        recent_fin = st.session_state.get("recent_finance_series", [])
        custom_fin_raw = st.session_state.get("custom_finance_series_raw", "")
        custom_fin_ids = _parse_csv_list(custom_fin_raw)
        finance_options = build_finance_options(recent_fin, custom_fin_ids)
        default_fin = [o for o in finance_options if o.id == "AAPL"][:1]
        selected_fin = st.multiselect(
            "Time series",
            options=finance_options,
            default=default_fin,
            format_func=lambda o: o.label,
            key="finance_timeseries",
            help="Search or select. Recently used appear first.",
        )
        st.text_input(
            "Custom series (comma-separated)",
            value=custom_fin_raw,
            key="custom_finance_series_raw",
            help="Add tickers not in the list (e.g. TSCO, VOD).",
        )
        tickers = [o.id for o in selected_fin]
        if tickers:
            st.warning(
                "**Stock tickers:** The FMP free API does not cover all tickers. Unsupported symbols fall back to "
                "Yahoo Finance, which is easily rate-limited. If a symbol is not under your FMP subscription, "
                "avoid pulling large datasets quickly."
            )
        interval = st.selectbox(
            "Interval",
            ["1d", "1h", "30m", "15m", "5m"],
            index=0,
            help="Price sampling interval for the market provider (e.g., 1d for daily).",
        )
        indicators = st.multiselect(
            "Indicators",
            ["sma", "ema", "rsi", "macd", "rolling_vol"],
            default=["sma", "rsi"],
            help="Technical indicators to compute on the selected Primary series after it finishes fetching.",
        )
        primary = st.selectbox(
            "Primary series for indicators",
            tickers or ["AAPL"],
            help="Indicators run only on this one series to keep compute time low.",
        )

        st.markdown("#### Forecasting")
        finance_forecast_methods = st.multiselect(
            "Forecast methods",
            [
                "forecast_naive",
                "forecast_drift",
                "forecast_ets",
                "forecast_arima",
                "forecast_ridge_lags",
                "forecast_rf_lags",
            ],
            default=["forecast_naive", "forecast_ridge_lags"],
            key="finance_forecast_methods",
            help="Adds forecast jobs for the Primary series once its data is ready.",
        )
        finance_forecast_target = st.selectbox(
            "Forecast target",
            ["close", "log_return(close)", "both"],
            index=2,
            key="finance_forecast_target",
            help="Forecast the close level, log returns, or both.",
        )
        finance_forecast_horizon = st.number_input(
            "Forecast horizon (steps)",
            min_value=1,
            max_value=500,
            value=30,
            step=1,
            key="finance_forecast_horizon",
            help="Number of future steps to forecast beyond the end of the training data.",
        )
        finance_forecast_test_size = st.number_input(
            "Test size (fraction)",
            min_value=0.0,
            max_value=0.8,
            value=0.2,
            step=0.05,
            key="finance_forecast_test_size",
            help="Fraction of history reserved for out-of-sample evaluation.",
        )
        finance_forecast_lookback = st.number_input(
            "ML lookback (lags)",
            min_value=5,
            max_value=200,
            value=20,
            step=1,
            key="finance_forecast_lookback",
            help="Number of lag observations used as features for ML lag models.",
        )
        news_auto = _auto_news_query_finance(tickers)
    else:
        recent_econ = st.session_state.get("recent_econ_series", [])
        custom_econ_raw = st.session_state.get("custom_econ_series_raw", "")
        custom_econ_ids = _parse_csv_list(custom_econ_raw)
        econ_options = build_econ_options(recent_econ, custom_econ_ids)
        default_econ = [o for o in econ_options if o.id == "CPIAUCSL"][:1]
        selected_econ = st.multiselect(
            "Time series",
            options=econ_options,
            default=default_econ,
            format_func=lambda o: o.label,
            key="econ_timeseries",
            help="Search or select. Recently used appear first.",
        )
        st.text_input(
            "Custom series (comma-separated)",
            value=custom_econ_raw,
            key="custom_econ_series_raw",
            help="Add FRED series IDs not in the list.",
        )
        series_ids = [o.id for o in selected_econ]
        models = st.multiselect(
            "Models",
            ["ar1", "ols", "lp_irf"],
            default=["ar1"],
            help="Econometric recipes to run once at least one selected macro series finishes fetching.",
        )

        st.markdown("#### Forecasting")
        econ_forecast_methods = st.multiselect(
            "Forecast methods",
            ["forecast_naive", "forecast_drift", "forecast_ets", "forecast_arima"],
            default=["forecast_naive", "forecast_arima"],
            key="econ_forecast_methods",
            help="Runs forecasts for each selected macro series once its artifact is ready.",
        )
        econ_forecast_horizon = st.number_input(
            "Forecast horizon (steps)",
            min_value=1,
            max_value=120,
            value=12,
            step=1,
            key="econ_forecast_horizon",
            help="Number of future steps to forecast beyond the end of training data.",
        )
        econ_forecast_test_size = st.number_input(
            "Test size (fraction)",
            min_value=0.0,
            max_value=0.8,
            value=0.2,
            step=0.05,
            key="econ_forecast_test_size",
            help="Fraction of history reserved for out-of-sample evaluation.",
        )
        # Note: single-series FRED artifacts are long-form (columns: date, series_id, value),
        # so y_col should default to 'value' unless you build a wide merged dataset.
        y_col = st.text_input(
            "Model y_col (after merge)",
            value="value",
            help="Column name for the dependent variable y in model inputs.",
        )
        x_cols = st.text_input(
            "Model x_cols (OLS, comma-separated)",
            value="",
            help="For OLS: comma-separated list of regressor columns (required). Example: UNRATE, FEDFUNDS",
        )
        shock_col = st.text_input(
            "LP-IRF shock_col",
            value="shock",
            help="For LP-IRF: which column represents the shock series (unless you choose Δy as shock).",
        )
        horizons = st.number_input(
            "LP-IRF horizons",
            min_value=1,
            max_value=60,
            value=12,
            step=1,
            help="Number of horizons (periods) to compute the impulse response for.",
        )
        lp_shock_mode = st.selectbox(
            "LP-IRF shock source",
            ["use shock_col from data", "use Δy (first difference) as shock"],
            index=1,
            help="If you don't have a shock series, Δy is a simple fallback so LP-IRF can run.",
        )

        news_auto = _auto_news_query_econ(series_ids)

        if any(m in {"ols", "taylor"} for m in models) and not st.session_state.get("active_dataset_path"):
            st.warning("OLS/Taylor require an active merged dataset. Use the Dataset Builder below first.")

    _dataset_builder_ui(
        cfg=cfg,
        storage=storage,
        runner=runner,
        reg=reg,
        start_date=start_date,
        end_date=end_date,
        default_fred_series=(series_ids if mode == "Econ" else []),
        default_tickers=(tickers if mode == "Finance" else []),
    )

    with st.expander("News", expanded=False):
        st.checkbox(
            "Fetch news",
            value=bool(st.session_state.news_fetch_enabled),
            key="news_fetch_enabled",
            help="If unchecked, the Run button will skip submitting the news job (faster, less clutter).",
        )
        st.checkbox(
            "Show news in results",
            value=bool(st.session_state.news_show_results),
            key="news_show_results",
            help="If unchecked, the News sections in Results will be hidden (jobs may still run).",
        )

        use_auto_news = st.checkbox(
            "Use auto news query",
            value=True,
            help="Auto-build a reasonable GDELT query from the selected series. Uncheck to write your own query.",
        )
        if use_auto_news:
            news_query = news_auto
            st.code(news_query)
            st.caption("Uncheck to override the query text.")
        else:
            news_query = st.text_input(
                "News query (override)",
                value=news_auto,
                help="Advanced: you can use GDELT query syntax (AND/OR, quotes, sourcelang:, domain:, etc.).",
            )

        lang_choices = [
            "Any",
            "english",
            "spanish",
            "french",
            "german",
            "portuguese",
            "italian",
            "russian",
            "chinese",
            "japanese",
            "korean",
            "arabic",
            "Custom",
        ]
        st.selectbox(
            "News language",
            options=lang_choices,
            index=lang_choices.index(str(st.session_state.news_language_choice)),
            key="news_language_choice",
            help="Filters articles by language (GDELT `sourcelang:`). Choose Custom for a comma-separated list.",
        )
        if st.session_state.news_language_choice == "Custom":
            st.text_input(
                "Custom languages (comma-separated)",
                value=str(st.session_state.news_language_custom_raw),
                key="news_language_custom_raw",
                help="Example: english, spanish. These map to GDELT's `sourcelang:<lang>` filter.",
            )

        maxrecords = st.number_input(
            "News maxrecords (<=250)",
            min_value=1,
            max_value=250,
            value=50,
            step=5,
            help="GDELT typically caps at 250. Larger values may be slower.",
        )
        run_sentiment = st.checkbox(
            "Compute daily news sentiment (FinBERT)",
            value=False,
            help="Runs a local model to score article titles/snippets and aggregates daily means.",
        )

    run_col, refresh_col = st.columns([1, 1])
    with run_col:
        run_btn = st.button("Run", type="primary")
    with refresh_col:
        refresh_btn = st.button("Refresh results", key="refresh_results_bottom")
        if refresh_btn:
            st.rerun()

    if run_btn:
        if mode == "Finance" and not tickers:
            st.error("Select at least one time series.")
            st.stop()
        if mode == "Econ" and not series_ids:
            st.error("Select at least one time series.")
            st.stop()

        # Reset latest run state
        st.session_state.latest_run = {}

        if mode == "Finance":
            market_jobs: dict[str, str] = {}
            for t in tickers:
                req = {
                    "symbol": t,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "interval": interval,  # type: ignore[name-defined]
                    "n": 252,
                }
                market_providers = [p.meta.id for p in reg.list("market")]
                provider_id = _preferred_market_provider(market_providers)
                if not provider_id:
                    st.error('No market provider available. Install with: pip install -e ".[market]"')
                    st.stop()
                job_id = submit_provider_fetch(
                    runner=runner, kind="market", provider_id=provider_id, request=req
                )
                market_jobs[t] = job_id

            news_providers = [p.meta.id for p in reg.list("news")]
            news_provider = (
                "gdelt" if "gdelt" in news_providers else (news_providers[0] if news_providers else None)
            )
            if news_provider and st.session_state.get("news_fetch_enabled", True):
                lang_choice = str(st.session_state.get("news_language_choice", "Any"))
                custom_langs = _parse_language_list(
                    str(st.session_state.get("news_language_custom_raw", ""))
                )
                if lang_choice == "Any":
                    news_lang: str | None = None
                    news_langs: list[str] = []
                elif lang_choice == "Custom":
                    news_lang = None
                    news_langs = custom_langs
                else:
                    news_lang = lang_choice
                    news_langs = []

                news_req: dict = {
                    "query": news_query,  # type: ignore[name-defined]
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "n": int(maxrecords),
                }
                if news_langs:
                    news_req["languages"] = news_langs
                elif news_lang:
                    news_req["language"] = news_lang
                news_job_id = submit_provider_fetch(
                    runner=runner, kind="news", provider_id=news_provider, request=news_req
                )
            else:
                news_job_id = None
                if not news_provider:
                    st.warning("No news provider available. Install requests for GDELT.")

            st.session_state.latest_run = {
                "mode": "Finance",
                "tickers": tickers,
                "primary": primary,  # type: ignore[name-defined]
                "indicators": indicators,  # type: ignore[name-defined]
                "forecast_methods": finance_forecast_methods,
                "forecast_target": finance_forecast_target,
                "forecast_horizon": int(finance_forecast_horizon),
                "forecast_test_size": float(finance_forecast_test_size),
                "forecast_lookback": int(finance_forecast_lookback),
                "market_jobs": market_jobs,
                "news_job_id": news_job_id,
                "run_sentiment": run_sentiment,
            }
            _update_recent_list(st.session_state.recent_finance_series, tickers)
        else:
            macro_jobs: dict[str, str] = {}
            for sid in series_ids:
                req = {
                    "series_id": sid,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "n": 120,
                }
                macro_providers = [p.meta.id for p in reg.list("macro")]
                provider_id = (
                    "fred" if "fred" in macro_providers else (macro_providers[0] if macro_providers else None)
                )
                if not provider_id:
                    st.error("No macro provider available. Install pandas-datareader for FRED.")
                    st.stop()
                job_id = submit_provider_fetch(
                    runner=runner, kind="macro", provider_id=provider_id, request=req
                )
                macro_jobs[sid] = job_id

            news_providers = [p.meta.id for p in reg.list("news")]
            news_provider = (
                "gdelt" if "gdelt" in news_providers else (news_providers[0] if news_providers else None)
            )
            if news_provider and st.session_state.get("news_fetch_enabled", True):
                lang_choice = str(st.session_state.get("news_language_choice", "Any"))
                custom_langs = _parse_language_list(
                    str(st.session_state.get("news_language_custom_raw", ""))
                )
                if lang_choice == "Any":
                    news_lang = None
                    news_langs: list[str] = []
                elif lang_choice == "Custom":
                    news_lang = None
                    news_langs = custom_langs
                else:
                    news_lang = lang_choice
                    news_langs = []

                news_req = {
                    "query": news_query,  # type: ignore[name-defined]
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "n": int(maxrecords),
                }
                if news_langs:
                    news_req["languages"] = news_langs
                elif news_lang:
                    news_req["language"] = news_lang
                news_job_id = submit_provider_fetch(
                    runner=runner, kind="news", provider_id=news_provider, request=news_req
                )
            else:
                news_job_id = None
                if not news_provider:
                    st.warning("No news provider available. Install requests for GDELT.")

            st.session_state.latest_run = {
                "mode": "Econ",
                "series_ids": series_ids,
                "models": models,  # type: ignore[name-defined]
                "forecast_methods": econ_forecast_methods,
                "forecast_horizon": int(econ_forecast_horizon),
                "forecast_test_size": float(econ_forecast_test_size),
                "y_col": y_col,  # type: ignore[name-defined]
                "x_cols": x_cols,  # type: ignore[name-defined]
                "shock_col": shock_col,  # type: ignore[name-defined]
                "lp_shock_mode": lp_shock_mode,  # type: ignore[name-defined]
                "horizons": int(horizons),
                "macro_jobs": macro_jobs,
                "news_job_id": news_job_id,
                "run_sentiment": run_sentiment,
            }
            _update_recent_list(st.session_state.recent_econ_series, series_ids)

        st.success("Submitted jobs. Results will update automatically while jobs run.")


def _render_results(
    *,
    storage: SQLiteStorage,
    runner: JobRunner,
    start_date: date,
    end_date: date,
    recession_intervals,
) -> None:
    latest = st.session_state.get("latest_run") or {}
    if not latest:
        st.caption("No run submitted yet.")
        return

    # Display job status + artifacts when ready
    jobs = storage.list_jobs(limit=500)
    jobs_by_id = {j.job_id: j for j in jobs}

    def _forecast_metrics(df_fcst: pd.DataFrame) -> dict[str, float]:
        """
        Compute lightweight forecast metrics from artifact rows where segment == 'test'.
        Expects columns: y, yhat, segment.
        """
        if df_fcst is None or df_fcst.empty:
            return {}
        if not {"segment", "y", "yhat"}.issubset(set(df_fcst.columns)):
            return {}
        d = df_fcst[df_fcst["segment"].astype(str) == "test"].copy()
        if d.empty:
            return {}
        y = pd.to_numeric(d["y"], errors="coerce")
        yhat = pd.to_numeric(d["yhat"], errors="coerce")
        mask = y.notna() & yhat.notna()
        y = y[mask]
        yhat = yhat[mask]
        if y.empty:
            return {}
        err = yhat - y
        mae = float(err.abs().mean())
        rmse = float((err.pow(2)).mean() ** 0.5)
        denom = y.abs()
        mape = float(((err.abs() / denom.replace(0, pd.NA)).dropna()).mean() * 100.0) if (denom > 0).any() else float("nan")
        return {"mae": mae, "rmse": rmse, "mape_pct": mape}

    def _iter_active_job_ids() -> list[str]:
        ids: list[str] = []
        latest = st.session_state.get("latest_run") or {}

        for _k, jid in (latest.get("market_jobs") or {}).items():
            if jid:
                ids.append(str(jid))
        for _k, jid in (latest.get("macro_jobs") or {}).items():
            if jid:
                ids.append(str(jid))
        for _k, jid in (latest.get("indicator_jobs") or {}).items():
            if jid:
                ids.append(str(jid))
        # Macro transform jobs (shape: dict[key] -> dict{job_id,...} OR dict[key]->job_id)
        for _k, spec in (latest.get("macro_transform_jobs") or {}).items():
            jid = None
            if isinstance(spec, dict):
                jid = spec.get("job_id")
            else:
                jid = spec
            if jid:
                ids.append(str(jid))
        for _k, jid in (latest.get("model_jobs") or {}).items():
            if jid:
                ids.append(str(jid))
        for k in ["news_job_id", "news_sentiment_job_id"]:
            jid = latest.get(k)
            if jid:
                ids.append(str(jid))

        for k in ["dataset_builder_merge_job_id", "usrec_job_id"]:
            jid = st.session_state.get(k)
            if jid:
                ids.append(str(jid))

        seen: set[str] = set()
        out: list[str] = []
        for jid in ids:
            if jid not in seen:
                seen.add(jid)
                out.append(jid)
        return out

    def _is_job_active(job_id: str) -> bool:
        j = jobs_by_id.get(job_id)
        if not j:
            return False
        return j.status == "RUNNING"

    if st.session_state.get("auto_refresh_enabled", True):
        active_ids = _iter_active_job_ids()
        if any(_is_job_active(jid) for jid in active_ids):
            st.caption("Auto-refreshing until all jobs finish…")
            st_autorefresh(interval=2000, key="job_poll")

    def _status_badge(job_id: str) -> str:
        j = jobs_by_id.get(job_id)
        if j is None:
            return "UNKNOWN"
        return j.status

    def _show_job_error(job_id: str) -> None:
        j = jobs_by_id.get(job_id)
        if not j or j.status != "FAILED":
            return
        with st.expander(f"Error details ({job_id})", expanded=False):
            if j.error:
                lines = j.error.splitlines()
                st.code("\n".join(lines[:60]))
            logs = storage.list_logs(job_id, limit=200)
            if logs:
                st.markdown("**Last logs**")
                for r in logs[-30:]:
                    st.write(f"`{r.ts}` **{r.level}** — {r.message}")

    if latest.get("mode") == "Finance":
        market_jobs = latest.get("market_jobs", {})
        with st.expander("Market data", expanded=True):
            st.write({t: _status_badge(jid) for t, jid in market_jobs.items()})
            for _t, jid in market_jobs.items():
                _show_job_error(jid)

            ready_market: dict[str, pd.DataFrame] = {}
            for t, jid in market_jobs.items():
                j = jobs_by_id.get(jid)
                if not j or j.status != "SUCCEEDED" or not j.output_path:
                    continue
                try:
                    ready_market[t] = _artifact_df(j.output_path)
                except Exception as e:
                    st.warning(f"Could not load artifact for {t} ({jid}): {e}")

            if ready_market:
                wide_close = _wide_from_market_artifacts(ready_market, value_col="close")
                if not wide_close.empty:
                    st.markdown("#### Selected series (close)")
                    cols = [c for c in wide_close.columns if c != "date"]
                    cols_to_plot, remaining_cols = _cols_with_visibility_selector(
                        cols,
                        max_visible=3,
                        displayed_count=2,
                        session_key="finance_main_series_close",
                    )
                    left_cols, right_cols = _split_cols_for_secondary_axis_by_scale(
                        wide_close, cols_to_plot, ratio_threshold=20.0
                    )
                    if right_cols:
                        st.caption(
                            "Auto-scaled: some series moved to a secondary axis due to very different magnitudes."
                        )
                    _plot_df(
                        wide_close,
                        x_col="date",
                        left_cols=left_cols,
                        right_cols_override=right_cols,
                        title="Market close (selected tickers)",
                        height=350,
                        recession_intervals=recession_intervals,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if remaining_cols:
                        with st.expander("Other series (not shown in main chart)", expanded=False):
                            for c in remaining_cols:
                                if c not in wide_close.columns:
                                    continue
                                _plot_df(
                                    wide_close,
                                    x_col="date",
                                    left_cols=[c],
                                    title=c,
                                    height=220,
                                    recession_intervals=recession_intervals,
                                    start_date=start_date,
                                    end_date=end_date,
                                )
                    with st.expander("Preview merged close series", expanded=False):
                        st.dataframe(wide_close.tail(200), use_container_width=True)

                st.markdown("#### Per-series outputs")
                for t in sorted(ready_market.keys()):
                    jid = market_jobs.get(t)
                    j = jobs_by_id.get(jid) if jid else None
                    df_t = ready_market[t]
                    with st.expander(f"{t} ({jid})", expanded=False):
                        if j and j.output_path:
                            st.write(f"Artifact: `{j.output_path}`")
                        _download_df(
                            df_t,
                            label=f"Download {t} CSV",
                            file_name=f"{t}_{jid}.csv" if jid else f"{t}.csv",
                        )
                        if "date" in df_t.columns:
                            _plot_df(
                                df_t,
                                x_col="date",
                                left_cols=[c for c in ["close", "open", "high", "low"] if c in df_t.columns],
                                title=f"{t} price",
                                height=250,
                                recession_intervals=recession_intervals,
                                start_date=start_date,
                                end_date=end_date,
                            )
                        st.dataframe(df_t.tail(100), use_container_width=True)

        with st.expander("Indicator jobs", expanded=False):
            primary = latest.get("primary")
            primary_job = market_jobs.get(primary)
            if primary_job and jobs_by_id.get(primary_job) and jobs_by_id[primary_job].output_path:
                df_price = None
                if primary and isinstance(market_jobs, dict):
                    # If it was loaded above, use it; otherwise read artifact.
                    # (Avoiding a dependency on ready_market scope.)
                    try:
                        df_price = _artifact_df(jobs_by_id[primary_job].output_path)
                    except Exception:
                        df_price = None
                if df_price is not None:
                    st.write(f"Artifact: `{jobs_by_id[primary_job].output_path}`")
                    _download_df(df_price, label="Download primary series CSV", file_name=f"{primary_job}.csv")

                # Submit indicator jobs once, when primary series is available
                if "indicator_jobs" not in latest:
                    indicator_jobs: dict[str, str] = {}
                    for rid in latest.get("indicators", []):
                        if rid == "sma":
                            params = {"value_col": "close", "window": 20, "out_col": "sma_20"}
                        elif rid == "ema":
                            params = {"value_col": "close", "span": 20, "out_col": "ema_20"}
                        elif rid == "rsi":
                            params = {"price_col": "close", "window": 14, "out_col": "rsi_14"}
                        elif rid == "macd":
                            params = {"price_col": "close", "fast": 12, "slow": 26, "signal": 9, "prefix": "macd"}
                        else:  # rolling_vol
                            params = {"price_col": "close", "window": 20, "annualize": True, "periods_per_year": 252, "out_col": "vol_20"}

                        jid = submit_recipe_run(
                            runner=runner,
                            input_job_id=primary_job,
                            input_path=jobs_by_id[primary_job].output_path,
                            recipe_id=rid,
                            params=params,
                        )
                        indicator_jobs[rid] = jid
                    latest["indicator_jobs"] = indicator_jobs
                    st.session_state.latest_run = latest
            elif primary:
                st.info(
                    f"Primary series '{primary}' is not ready yet; indicators will run once it finishes."
                )

            ind_jobs = latest.get("indicator_jobs", {})
            if ind_jobs:
                st.write({rid: _status_badge(jid) for rid, jid in ind_jobs.items()})
                for rid, jid in ind_jobs.items():
                    j = jobs_by_id.get(jid)
                    if j and j.output_path:
                        df_ind = _artifact_df(j.output_path)
                        with st.expander(f"{rid} output ({jid})", expanded=False):
                            st.write(f"`{j.output_path}`")
                            _download_df(df_ind, label=f"Download {rid} CSV", file_name=f"{jid}.csv")
                            if "date" in df_ind.columns:
                                if rid in {"sma", "ema"}:
                                    col = "sma_20" if rid == "sma" else "ema_20"
                                    _plot_df(
                                        df_ind,
                                        x_col="date",
                                        left_cols=["close", col],
                                        title=f"{rid} ({primary})",
                                        height=250,
                                        recession_intervals=recession_intervals,
                                        start_date=start_date,
                                        end_date=end_date,
                                    )
                                elif rid == "rsi":
                                    _plot_df(
                                        df_ind,
                                        x_col="date",
                                        left_cols=["rsi_14"],
                                        title=f"{rid} ({primary})",
                                        height=250,
                                        recession_intervals=recession_intervals,
                                        start_date=start_date,
                                        end_date=end_date,
                                    )
                                elif rid == "macd":
                                    _plot_df(
                                        df_ind,
                                        x_col="date",
                                        left_cols=["macd_line", "macd_signal", "macd_hist"],
                                        title=f"{rid} ({primary})",
                                        height=250,
                                        recession_intervals=recession_intervals,
                                        start_date=start_date,
                                        end_date=end_date,
                                    )
                                else:
                                    _plot_df(
                                        df_ind,
                                        x_col="date",
                                        left_cols=["vol_20"],
                                        title=f"{rid} ({primary})",
                                        height=250,
                                        recession_intervals=recession_intervals,
                                        start_date=start_date,
                                        end_date=end_date,
                                    )
                            st.dataframe(df_ind.tail(100), use_container_width=True)

        with st.expander("Forecast jobs", expanded=False):
            primary = latest.get("primary")
            market_jobs = latest.get("market_jobs", {})
            primary_job_id = market_jobs.get(primary) if primary and isinstance(market_jobs, dict) else None
            primary_job = jobs_by_id.get(primary_job_id) if primary_job_id else None

            methods = latest.get("forecast_methods") or []
            if not methods:
                st.caption("No forecast methods selected.")
            elif not primary_job or primary_job.status != "SUCCEEDED" or not primary_job.output_path:
                st.info("Primary series is not ready yet; forecasts will run once it finishes.")
            else:
                horizon = int(latest.get("forecast_horizon", 30))
                test_size = float(latest.get("forecast_test_size", 0.2))
                lookback = int(latest.get("forecast_lookback", 20))
                target_choice = str(latest.get("forecast_target", "both"))

                targets: list[tuple[str, str]] = []
                if target_choice in {"close", "both"}:
                    targets.append(("close", "none"))
                if target_choice in {"log_return(close)", "both"}:
                    targets.append(("log_return(close)", "log_return"))

                # Submit forecast jobs once
                if "forecast_jobs" not in latest:
                    latest["forecast_jobs"] = {}
                forecast_jobs: dict[str, str] = latest.get("forecast_jobs") or {}

                if primary_job_id and primary_job.output_path:
                    for rid in methods:
                        for _label, transform in targets:
                            key = f"{rid}|{transform}"
                            if key in forecast_jobs:
                                continue
                            params: dict = {
                                "date_col": "date",
                                "y_col": "close",
                                "transform": transform,
                                "horizon": horizon,
                                "test_size": test_size,
                            }
                            if rid in {"forecast_ridge_lags", "forecast_rf_lags"}:
                                params["lookback"] = lookback
                            jid = submit_recipe_run(
                                runner=runner,
                                input_job_id=primary_job_id,
                                input_path=primary_job.output_path,
                                recipe_id=rid,
                                params=params,
                            )
                            forecast_jobs[key] = jid

                    latest["forecast_jobs"] = forecast_jobs
                    st.session_state.latest_run = latest

                if forecast_jobs:
                    st.write({k: _status_badge(jid) for k, jid in forecast_jobs.items()})
                    for k, jid in forecast_jobs.items():
                        _show_job_error(jid)
                        j = jobs_by_id.get(jid)
                        if not j or j.status != "SUCCEEDED" or not j.output_path:
                            continue
                        df_out = _artifact_df(j.output_path)
                        with st.expander(f"{k} output ({jid})", expanded=False):
                            st.write(f"`{j.output_path}`")
                            _download_df(df_out, label=f"Download {k} forecast CSV", file_name=f"{jid}.csv")
                            mets = _forecast_metrics(df_out)
                            if mets:
                                st.write(mets)
                            if "date" in df_out.columns and "yhat" in df_out.columns:
                                cols = [c for c in ["y", "yhat"] if c in df_out.columns]
                                _plot_df(
                                    df_out,
                                    x_col="date",
                                    left_cols=cols,
                                    title=f"Forecast ({k})",
                                    height=250,
                                    recession_intervals=recession_intervals,
                                    start_date=start_date,
                                    end_date=end_date,
                                )
                            st.dataframe(df_out.tail(200), use_container_width=True)

        if st.session_state.get("news_show_results", True):
            with st.expander("News", expanded=False):
                news_job_id = latest.get("news_job_id")
                if news_job_id:
                    st.write({"news_job_id": news_job_id, "status": _status_badge(news_job_id)})
                    _show_job_error(news_job_id)
                    jn = jobs_by_id.get(news_job_id)
                    if jn and jn.output_path:
                        df_news = _artifact_df(jn.output_path)
                        st.write(f"News artifact: `{jn.output_path}`")
                        _download_df(df_news, label="Download news CSV", file_name=f"{news_job_id}.csv")

                        lang_choice = str(st.session_state.get("news_language_choice", "Any"))
                        if lang_choice != "Any" and "language" in df_news.columns:
                            if lang_choice == "Custom":
                                langs = set(_parse_language_list(str(st.session_state.get("news_language_custom_raw", ""))))
                            else:
                                langs = {lang_choice.lower()}
                            if langs:
                                df_news = df_news[df_news["language"].astype(str).str.lower().isin(langs)].reset_index(drop=True)

                        cols = [c for c in ["date", "ts", "language", "source", "title", "url"] if c in df_news.columns]
                        df_show = df_news[cols].head(50) if cols else df_news.head(50)
                        st.dataframe(_df_with_rank(df_show), use_container_width=True, hide_index=True)

                        if "url" in df_news.columns and "title" in df_news.columns:
                            st.markdown("#### Top articles")
                            for i, (_, row) in enumerate(df_news.head(10).iterrows(), start=1):
                                title = str(row.get("title", "")).strip()
                                url = str(row.get("url", "")).strip()
                                if title and url:
                                    st.markdown(f"{i}. [{title}]({url})")

                        if latest.get("run_sentiment") and "news_sentiment_job_id" not in latest:
                            sent_jid = submit_recipe_run(
                                runner=runner,
                                input_job_id=news_job_id,
                                input_path=jn.output_path,
                                recipe_id="news_sentiment",
                                params={"ts_col": "ts", "title_col": "title", "snippet_col": "snippet", "prefix": "sent", "batch_size": 16},
                            )
                            latest["news_sentiment_job_id"] = sent_jid
                            st.session_state.latest_run = latest

                        sent_jid = latest.get("news_sentiment_job_id")
                        if sent_jid:
                            st.write({"sentiment_job_id": sent_jid, "status": _status_badge(sent_jid)})
                            js = jobs_by_id.get(sent_jid)
                            if js and js.output_path:
                                df_sent = _artifact_df(js.output_path)
                                with st.expander("Daily sentiment time series", expanded=True):
                                    _download_df(df_sent, label="Download sentiment CSV", file_name=f"{sent_jid}.csv")
                                    if "date" in df_sent.columns and any(c.endswith("_mean") for c in df_sent.columns):
                                        mean_cols = [c for c in df_sent.columns if c.endswith("_mean")]
                                        _plot_df(
                                            df_sent,
                                            x_col="date",
                                            left_cols=mean_cols,
                                            title="Daily news sentiment (mean)",
                                            height=250,
                                            recession_intervals=recession_intervals,
                                            start_date=start_date,
                                            end_date=end_date,
                                        )
                                    st.dataframe(df_sent, use_container_width=True)
        else:
            st.caption("News hidden (enable “Show news in results” above to display it).")

    else:
        macro_jobs = latest.get("macro_jobs", {})
        with st.expander("Macro data", expanded=True):
            st.write({sid: _status_badge(jid) for sid, jid in macro_jobs.items()})
            for _sid, jid in macro_jobs.items():
                _show_job_error(jid)

            ready_macro: dict[str, pd.DataFrame] = {}
            for sid, jid in macro_jobs.items():
                j = jobs_by_id.get(jid)
                if not j or j.status != "SUCCEEDED" or not j.output_path:
                    continue
                try:
                    ready_macro[sid] = _artifact_df(j.output_path)
                except Exception as e:
                    st.warning(f"Could not load artifact for {sid} ({jid}): {e}")

            if ready_macro:
                wide_macro = _wide_from_macro_artifacts(ready_macro, value_col="value")
                if not wide_macro.empty:
                    st.markdown("#### Selected series")
                    cols = [c for c in wide_macro.columns if c != "date"]
                    cols_to_plot, remaining_cols = _cols_with_visibility_selector(
                        cols,
                        max_visible=3,
                        displayed_count=2,
                        session_key="econ_main_series_macro",
                    )
                    left_cols, right_cols = _split_cols_for_secondary_axis_by_scale(
                        wide_macro, cols_to_plot, ratio_threshold=50.0
                    )
                    if right_cols:
                        st.caption(
                            "Auto-scaled: some series moved to a secondary axis due to very different magnitudes."
                        )
                    _plot_df(
                        wide_macro,
                        x_col="date",
                        left_cols=left_cols,
                        right_cols_override=right_cols,
                        title="Macro series (selected)",
                        height=350,
                        recession_intervals=recession_intervals,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if remaining_cols:
                        with st.expander("Other series (not shown in main chart)", expanded=False):
                            for c in remaining_cols:
                                if c not in wide_macro.columns:
                                    continue
                                _plot_df(
                                    wide_macro,
                                    x_col="date",
                                    left_cols=[c],
                                    title=c,
                                    height=220,
                                    recession_intervals=recession_intervals,
                                    start_date=start_date,
                                    end_date=end_date,
                                )
                    with st.expander("Preview merged macro series", expanded=False):
                        st.dataframe(wide_macro.tail(200), use_container_width=True)

                st.markdown("#### Per-series outputs")
                for sid in sorted(ready_macro.keys()):
                    jid = macro_jobs.get(sid)
                    j = jobs_by_id.get(jid) if jid else None
                    df_sid = ready_macro[sid]
                    with st.expander(f"{sid} ({jid})", expanded=False):
                        if j and j.output_path:
                            st.write(f"Artifact: `{j.output_path}`")
                        _download_df(
                            df_sid,
                            label=f"Download {sid} CSV",
                            file_name=f"{sid}_{jid}.csv" if jid else f"{sid}.csv",
                        )
                        if "date" in df_sid.columns and "value" in df_sid.columns:
                            _plot_df(
                                df_sid,
                                x_col="date",
                                left_cols=["value"],
                                title=f"{sid}",
                                height=250,
                                recession_intervals=recession_intervals,
                                start_date=start_date,
                                end_date=end_date,
                            )
                        st.dataframe(df_sid.tail(100), use_container_width=True)

        with st.expander("Macro transforms", expanded=False):
            st.caption("Run macro cleaning/transforms from the main UI (no legacy pages needed).")

            macro_transform_jobs: dict = latest.get("macro_transform_jobs") or {}
            if "macro_transform_jobs" not in latest:
                latest["macro_transform_jobs"] = macro_transform_jobs

            # --------------------
            # Single-series transforms (long-form FRED artifact: date, series_id, value)
            # --------------------
            st.markdown("#### Single-series transforms")
            ready_series: list[tuple[str, str, str]] = []
            for sid, jid in (macro_jobs or {}).items():
                j = jobs_by_id.get(jid)
                if j and j.status == "SUCCEEDED" and j.output_path:
                    ready_series.append((sid, str(jid), str(j.output_path)))
            ready_series = sorted(ready_series, key=lambda x: x[0])

            if not ready_series:
                st.caption("No completed macro artifacts yet. Fetch at least one series first.")
            else:
                sid_options = [sid for (sid, _jid, _p) in ready_series]
                sid_to_job = {sid: jid for (sid, jid, _p) in ready_series}
                sid_to_path = {sid: p for (sid, _jid, p) in ready_series}

                single_sid = st.selectbox(
                    "Input series (completed)",
                    options=sid_options,
                    index=0,
                    key="macro_transform_single_sid",
                )
                single_recipe = st.selectbox(
                    "Transform",
                    options=["inflation_yoy", "inflation_mom_ann", "growth_qoq_ann"],
                    index=0,
                    key="macro_transform_single_recipe",
                    help="These transforms run on a single long-form series with a 'value' column.",
                )
                default_out = (
                    "inflation_yoy"
                    if single_recipe == "inflation_yoy"
                    else ("inflation_mom_ann" if single_recipe == "inflation_mom_ann" else "growth_qoq_ann")
                )
                single_out_col = st.text_input(
                    "Output column",
                    value=default_out,
                    key="macro_transform_single_out_col",
                )

                run_single = st.button("Run single-series transform", key="macro_transform_run_single")
                if run_single:
                    in_jid = sid_to_job.get(single_sid)
                    in_path = sid_to_path.get(single_sid)
                    if not in_jid or not in_path:
                        st.error("Selected input series is not ready yet.")
                    else:
                        spec_key = f"single|{single_sid}|{single_recipe}|{single_out_col}"
                        if spec_key in macro_transform_jobs:
                            st.info("That transform job has already been submitted in this run.")
                        else:
                            params = {"date_col": "date", "value_col": "value", "out_col": single_out_col}
                            jid_out = submit_recipe_run(
                                runner=runner,
                                input_job_id=in_jid,
                                input_path=in_path,
                                recipe_id=single_recipe,
                                params=params,
                            )
                            macro_transform_jobs[spec_key] = {
                                "job_id": jid_out,
                                "label": f"{single_recipe} on {single_sid}",
                                "out_col": single_out_col,
                            }
                            latest["macro_transform_jobs"] = macro_transform_jobs
                            st.session_state.latest_run = latest
                            st.success(f"Submitted transform job: {jid_out}")

            st.divider()

            # --------------------
            # Merged-dataset transforms (wide dataset: date + many columns)
            # --------------------
            st.markdown("#### Merged-dataset transforms")
            active_dataset_path = st.session_state.get("active_dataset_path")
            if not active_dataset_path:
                st.info("No active merged dataset. Build one in Dataset Builder to run spreads/real-rate on wide data.")
            else:
                try:
                    cols = [str(c) for c in pd.read_csv(active_dataset_path, nrows=5).columns]
                except Exception as e:
                    cols = []
                    st.warning(f"Could not read merged dataset columns: {e}")

                wide_cols = [c for c in cols if c != "date"]
                if not wide_cols:
                    st.caption("Merged dataset has no usable columns beyond 'date'.")
                else:
                    merged_recipe = st.selectbox(
                        "Transform (merged dataset)",
                        options=["spread", "real_rate"],
                        index=0,
                        key="macro_transform_merged_recipe",
                        help="These transforms run on the active merged dataset (wide columns).",
                    )

                    merged_params: dict = {"date_col": "date"}
                    merged_out_col = ""

                    if merged_recipe == "spread":
                        a = st.selectbox(
                            "series_a (A - B)",
                            options=wide_cols,
                            index=0,
                            key="macro_transform_spread_a",
                        )
                        b = st.selectbox(
                            "series_b (A - B)",
                            options=wide_cols,
                            index=(1 if len(wide_cols) > 1 else 0),
                            key="macro_transform_spread_b",
                        )
                        merged_out_col = st.text_input(
                            "Output column (optional)",
                            value="",
                            key="macro_transform_spread_out",
                            help="Leave blank to use default: series_a_minus_series_b",
                        )
                        merged_params.update({"series_a": a, "series_b": b})
                        if merged_out_col.strip():
                            merged_params["out_col"] = merged_out_col.strip()
                    else:  # real_rate
                        nominal = st.selectbox(
                            "nominal_col",
                            options=wide_cols,
                            index=0,
                            key="macro_transform_real_nominal",
                            help="Example: FEDFUNDS or DGS10",
                        )
                        infl = st.selectbox(
                            "inflation_col",
                            options=wide_cols,
                            index=(wide_cols.index("inflation_yoy") if "inflation_yoy" in wide_cols else 0),
                            key="macro_transform_real_infl",
                            help="Example: inflation_yoy column from CPI transform",
                        )
                        merged_out_col = st.text_input(
                            "Output column",
                            value="real_rate",
                            key="macro_transform_real_out",
                        )
                        merged_params.update(
                            {"nominal_col": nominal, "inflation_col": infl, "out_col": merged_out_col.strip() or "real_rate"}
                        )

                    run_merged = st.button("Run merged-dataset transform", key="macro_transform_run_merged")
                    if run_merged:
                        input_job_id = st.session_state.get("dataset_builder_merge_job_id") or "active_dataset"
                        spec_key = f"merged|{merged_recipe}|{merged_params}"
                        if spec_key in macro_transform_jobs:
                            st.info("That merged-dataset transform job has already been submitted in this run.")
                        else:
                            jid_out = submit_recipe_run(
                                runner=runner,
                                input_job_id=str(input_job_id),
                                input_path=str(active_dataset_path),
                                recipe_id=merged_recipe,
                                params=merged_params,
                            )
                            label = f"{merged_recipe} on merged dataset"
                            if merged_recipe == "spread":
                                label = f"spread ({merged_params.get('series_a')} - {merged_params.get('series_b')})"
                            macro_transform_jobs[spec_key] = {
                                "job_id": jid_out,
                                "label": label,
                                "out_col": merged_params.get("out_col", merged_out_col).strip() if isinstance(merged_params.get("out_col", ""), str) else "",
                            }
                            latest["macro_transform_jobs"] = macro_transform_jobs
                            st.session_state.latest_run = latest
                            st.success(f"Submitted transform job: {jid_out}")

            # --------------------
            # Outputs
            # --------------------
            if macro_transform_jobs:
                st.markdown("#### Transform job outputs")
                # Compact status view
                status_map: dict[str, str] = {}
                for k, spec in macro_transform_jobs.items():
                    if isinstance(spec, dict) and spec.get("job_id"):
                        label = str(spec.get("label") or k)
                        status_map[label] = _status_badge(str(spec["job_id"]))
                if status_map:
                    st.write(status_map)

                for k, spec in macro_transform_jobs.items():
                    if not isinstance(spec, dict) or not spec.get("job_id"):
                        continue
                    jid = str(spec["job_id"])
                    label = str(spec.get("label") or k)
                    out_col = str(spec.get("out_col") or "").strip()
                    _show_job_error(jid)

                    j = jobs_by_id.get(jid)
                    with st.expander(f"{label} ({jid})", expanded=False):
                        if j and j.output_path:
                            st.write(f"Artifact: `{j.output_path}`")
                            try:
                                df_out = _artifact_df(j.output_path)
                                _download_df(df_out, label="Download transform CSV", file_name=f"{jid}.csv")
                                if "date" in df_out.columns:
                                    cols_to_plot: list[str] = []
                                    if out_col and out_col in df_out.columns:
                                        cols_to_plot.append(out_col)
                                    # Show original value if present (single-series case)
                                    if "value" in df_out.columns and "value" not in cols_to_plot:
                                        cols_to_plot = ["value"] + cols_to_plot
                                    # Fallback: first few non-date columns
                                    if not cols_to_plot:
                                        cols_to_plot = [c for c in df_out.columns if c != "date"][:3]
                                    _plot_df(
                                        df_out,
                                        x_col="date",
                                        left_cols=cols_to_plot,
                                        title=label,
                                        height=250,
                                        recession_intervals=recession_intervals,
                                        start_date=start_date,
                                        end_date=end_date,
                                    )
                                st.dataframe(df_out.tail(200), use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not load transform artifact: {e}")
                        else:
                            st.caption(f"Status: {_status_badge(jid)}")

        with st.expander("Model jobs", expanded=False):
            # Pick a model input series that is actually ready (fallback: first selected)
            series_order = latest.get("series_ids") or []
            # We might not have ready_macro in this scope if nothing finished; rebuild minimal set.
            ready_macro_ids = set()
            for sid, jid in macro_jobs.items():
                j = jobs_by_id.get(jid)
                if j and j.status == "SUCCEEDED" and j.output_path:
                    ready_macro_ids.add(sid)
            model_sid = next((sid for sid in series_order if sid in ready_macro_ids), (series_order[0] if series_order else None))
            model_job_id = macro_jobs.get(model_sid) if model_sid else None
            model_job = jobs_by_id.get(model_job_id) if model_job_id else None
            df_macro = None
            if model_job and model_job.status == "SUCCEEDED" and model_job.output_path:
                try:
                    df_macro = _artifact_df(model_job.output_path)
                except Exception:
                    df_macro = None

            can_submit_models = bool(
                model_sid
                and model_job_id
                and model_job
                and model_job.status == "SUCCEEDED"
                and model_job.output_path
                and df_macro is not None
                and not df_macro.empty
            )

            if "model_jobs" not in latest:
                if not can_submit_models and (latest.get("models") or []):
                    st.info("Model input series not ready yet. Auto-refresh will update once a selected series finishes.")
                elif can_submit_models:
                    model_jobs: dict[str, str] = {}
                    for mid in latest.get("models", []):
                        input_job_id = model_job_id
                        input_path = model_job.output_path  # type: ignore[union-attr]

                        if mid in {"ols", "taylor"}:
                            active_dataset_path = st.session_state.get("active_dataset_path")
                            if not active_dataset_path:
                                st.warning("Skipping OLS/Taylor: no active merged dataset. Build one in Dataset Builder.")
                                continue
                            input_path = active_dataset_path
                            input_job_id = st.session_state.get("dataset_builder_merge_job_id") or input_job_id

                        if mid == "ar1":
                            params = {"y_col": "value"}
                        elif mid == "ols":
                            x_cols_raw = str(latest.get("x_cols", "")).strip()
                            if not x_cols_raw:
                                st.warning("Skipping OLS: x_cols is required (comma-separated). Example: UNRATE, FEDFUNDS")
                                continue
                            params = {"y_col": latest.get("y_col", "value"), "x_cols": x_cols_raw, "add_constant": True}
                        elif mid == "lp_irf":
                            if latest.get("lp_shock_mode") == "use Δy (first difference) as shock":
                                df_tmp = df_macro.copy()  # type: ignore[union-attr]
                                yname = latest.get("y_col", "value")
                                if yname not in df_tmp.columns:
                                    raise ValueError(f"LP-IRF y_col '{yname}' not found in artifact. Columns: {list(df_tmp.columns)}")
                                df_tmp["shock"] = pd.to_numeric(df_tmp[yname], errors="coerce").diff()
                                tmp_path = jobs_by_id[model_job_id].output_path + ".lp_shock.csv"  # type: ignore[index,union-attr]
                                df_tmp.to_csv(tmp_path, index=False)
                                input_path = tmp_path
                                shock_col = "shock"
                            else:
                                shock_col = latest.get("shock_col", "shock")
                            params = {
                                "y_col": latest.get("y_col", "value"),
                                "shock_col": shock_col,
                                "controls": "",
                                "horizons": latest.get("horizons", 12),
                                "ci_level": 0.95,
                                "add_constant": True,
                            }
                        else:
                            st.warning(f"Unknown model id '{mid}'. Skipping.")
                            continue
                        jid = submit_recipe_run(
                            runner=runner,
                            input_job_id=input_job_id,
                            input_path=input_path,
                            recipe_id=mid,
                            params=params,
                        )
                        model_jobs[mid] = jid
                    if model_jobs:
                        latest["model_jobs"] = model_jobs
                        st.session_state.latest_run = latest

            model_jobs = latest.get("model_jobs", {})
            if model_jobs:
                st.write({mid: _status_badge(jid) for mid, jid in model_jobs.items()})
                for mid, jid in model_jobs.items():
                    j = jobs_by_id.get(jid)
                    if j and j.output_path:
                        df_out = _artifact_df(j.output_path)
                        with st.expander(f"{mid} output ({jid})", expanded=False):
                            st.write(f"`{j.output_path}`")
                            _download_df(df_out, label=f"Download {mid} output CSV", file_name=f"{jid}.csv")
                            if mid == "lp_irf" and "horizon" in df_out.columns and "irf" in df_out.columns:
                                _plot_lp_irf(df_out, height=250, title="Local projection IRF")
                            st.dataframe(df_out, use_container_width=True)

        with st.expander("Forecast jobs", expanded=False):
            methods = latest.get("forecast_methods") or []
            if not methods:
                st.caption("No forecast methods selected.")
            else:
                horizon = int(latest.get("forecast_horizon", 12))
                test_size = float(latest.get("forecast_test_size", 0.2))

                if "forecast_jobs" not in latest:
                    latest["forecast_jobs"] = {}
                forecast_jobs = latest.get("forecast_jobs") or {}

                # Submit missing forecast jobs for each ready macro series
                for sid, jid_in in (macro_jobs or {}).items():
                    j_in = jobs_by_id.get(jid_in)
                    if not j_in or j_in.status != "SUCCEEDED" or not j_in.output_path:
                        continue
                    if sid not in forecast_jobs:
                        forecast_jobs[sid] = {}
                    for rid in methods:
                        if rid in forecast_jobs[sid]:
                            continue
                        params = {
                            "date_col": "date",
                            "y_col": "value",
                            "transform": "none",
                            "horizon": horizon,
                            "test_size": test_size,
                        }
                        jid_out = submit_recipe_run(
                            runner=runner,
                            input_job_id=jid_in,
                            input_path=j_in.output_path,
                            recipe_id=rid,
                            params=params,
                        )
                        forecast_jobs[sid][rid] = jid_out

                latest["forecast_jobs"] = forecast_jobs
                st.session_state.latest_run = latest

                # Display
                for sid in (latest.get("series_ids") or []):
                    sid_jobs = (forecast_jobs or {}).get(sid) or {}
                    if not sid_jobs:
                        continue
                    with st.expander(f"{sid} forecasts", expanded=False):
                        st.write({rid: _status_badge(jid) for rid, jid in sid_jobs.items()})
                        for rid, jid in sid_jobs.items():
                            _show_job_error(jid)
                            j = jobs_by_id.get(jid)
                            if not j or j.status != "SUCCEEDED" or not j.output_path:
                                continue
                            df_out = _artifact_df(j.output_path)
                            mets = _forecast_metrics(df_out)
                            if mets:
                                st.write({rid: mets})
                            if "date" in df_out.columns and "yhat" in df_out.columns:
                                cols = [c for c in ["y", "yhat"] if c in df_out.columns]
                                _plot_df(
                                    df_out,
                                    x_col="date",
                                    left_cols=cols,
                                    title=f"{sid} — {rid}",
                                    height=240,
                                    recession_intervals=recession_intervals,
                                    start_date=start_date,
                                    end_date=end_date,
                                )
                            _download_df(df_out, label=f"Download {sid} {rid} forecast CSV", file_name=f"{jid}.csv")
                            st.dataframe(df_out.tail(200), use_container_width=True)

        if st.session_state.get("news_show_results", True):
            with st.expander("News", expanded=False):
                news_job_id = latest.get("news_job_id")
                if news_job_id:
                    st.write({"news_job_id": news_job_id, "status": _status_badge(news_job_id)})
                    _show_job_error(news_job_id)
                    jn = jobs_by_id.get(news_job_id)
                    if jn and jn.output_path:
                        df_news = _artifact_df(jn.output_path)
                        st.write(f"News artifact: `{jn.output_path}`")
                        _download_df(df_news, label="Download news CSV", file_name=f"{news_job_id}.csv")

                        lang_choice = str(st.session_state.get("news_language_choice", "Any"))
                        if lang_choice != "Any" and "language" in df_news.columns:
                            if lang_choice == "Custom":
                                langs = set(_parse_language_list(str(st.session_state.get("news_language_custom_raw", ""))))
                            else:
                                langs = {lang_choice.lower()}
                            if langs:
                                df_news = df_news[df_news["language"].astype(str).str.lower().isin(langs)].reset_index(drop=True)

                        cols = [c for c in ["date", "ts", "language", "source", "title", "url"] if c in df_news.columns]
                        df_show = df_news[cols].head(50) if cols else df_news.head(50)
                        st.dataframe(_df_with_rank(df_show), use_container_width=True, hide_index=True)
                        if "url" in df_news.columns and "title" in df_news.columns:
                            st.markdown("#### Top articles")
                            for i, (_, row) in enumerate(df_news.head(10).iterrows(), start=1):
                                title = str(row.get("title", "")).strip()
                                url = str(row.get("url", "")).strip()
                                if title and url:
                                    st.markdown(f"{i}. [{title}]({url})")

                        if latest.get("run_sentiment") and "news_sentiment_job_id" not in latest:
                            sent_jid = submit_recipe_run(
                                runner=runner,
                                input_job_id=news_job_id,
                                input_path=jn.output_path,
                                recipe_id="news_sentiment",
                                params={"ts_col": "ts", "title_col": "title", "snippet_col": "snippet", "prefix": "sent", "batch_size": 16},
                            )
                            latest["news_sentiment_job_id"] = sent_jid
                            st.session_state.latest_run = latest

                        sent_jid = latest.get("news_sentiment_job_id")
                        if sent_jid:
                            st.write({"sentiment_job_id": sent_jid, "status": _status_badge(sent_jid)})
                            js = jobs_by_id.get(sent_jid)
                            if js and js.output_path:
                                df_sent = _artifact_df(js.output_path)
                                with st.expander("Daily sentiment time series", expanded=True):
                                    _download_df(df_sent, label="Download sentiment CSV", file_name=f"{sent_jid}.csv")
                                    if "date" in df_sent.columns and any(c.endswith("_mean") for c in df_sent.columns):
                                        mean_cols = [c for c in df_sent.columns if c.endswith("_mean")]
                                        _plot_df(
                                            df_sent,
                                            x_col="date",
                                            left_cols=mean_cols,
                                            title="Daily news sentiment (mean)",
                                            height=250,
                                            recession_intervals=recession_intervals,
                                            start_date=start_date,
                                            end_date=end_date,
                                        )
                                    st.dataframe(df_sent, use_container_width=True)
        else:
            st.caption("News hidden (enable “Show news in results” above to display it).")


def main():
    st.set_page_config(page_title="finrec-app", layout="wide")
    cfg, storage, runner = _get_singletons()

    st.title("finrec-app")

    top_left, top_right = st.columns([2, 1])
    with top_left:
        mode = st.radio(
            "Mode",
            ["Finance", "Econ"],
            horizontal=True,
            help="Finance focuses on tickers + indicators. Econ focuses on macro series + models.",
        )
    with top_right:
        st.checkbox(
            "Auto-refresh while jobs run",
            value=bool(st.session_state.auto_refresh_enabled),
            key="auto_refresh_enabled",
            help="When enabled, the page will refresh periodically while any job is RUNNING (so you don't need to click Refresh).",
        )
        if st.button("Refresh results"):
            st.rerun()

    with st.expander("Runtime configuration", expanded=False):
        st.code(
            "\n".join(
                [
                    f"FINREC_DB_PATH={cfg.db_path.as_posix()}",
                    f"FINREC_RESULTS_DIR={cfg.results_dir.as_posix()}",
                ]
            )
        )

    reg = get_registry()

    # Date range inputs (shared)
    today = date.today()
    default_start = today - timedelta(days=365)
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        start_date = st.date_input(
            "start_date",
            value=default_start,
            min_value=date(1900, 1, 1),
            max_value=today,
            help="Start date for all fetches (market, macro, and news).",
        )
    with dcol2:
        end_date = st.date_input(
            "end_date",
            value=today,
            min_value=date(1900, 1, 1),
            max_value=today,
            help="End date for all fetches (market, macro, and news).",
        )

    if end_date < start_date:
        st.error("end_date must be >= start_date")
        st.stop()

    # Chart controls (shared)
    with st.expander("Chart options", expanded=False):
        st.checkbox(
            "Filter charts to selected date range",
            value=bool(st.session_state.chart_filter_range),
            key="chart_filter_range",
            help="If enabled, plots are clipped to the start/end date range selected above.",
        )
        st.checkbox(
            "Log scale (y)",
            value=bool(st.session_state.chart_log_scale),
            key="chart_log_scale",
            help="Use a logarithmic y-axis (useful for growth-like series).",
        )
        st.checkbox(
            "Secondary axis compare",
            value=bool(st.session_state.chart_secondary_axis),
            key="chart_secondary_axis",
            help="Move selected columns onto a secondary y-axis for better comparison across scales.",
        )
        st.text_input(
            "Secondary axis columns (comma-separated; if present in the current chart's dataframe)",
            value=str(st.session_state.chart_secondary_cols_raw),
            key="chart_secondary_cols_raw",
            help="Example: CPIAUCSL, UNRATE, AAPL_close",
        )
        st.checkbox(
            "Recession shading (USREC)",
            value=bool(st.session_state.chart_recession_shading),
            key="chart_recession_shading",
            help="Fetches USREC from FRED once and shades recession periods on charts.",
        )

    # Recession shading support (fetch USREC once via the job runner)
    recession_intervals = None
    if st.session_state.get("chart_recession_shading"):
        has_real_fred = any(p.meta.id == "fred" for p in reg.list("macro"))
        if not has_real_fred:
            st.warning("Recession shading requires the real FRED provider (pandas-datareader).")
        else:
            jobs_by_id = {j.job_id: j for j in storage.list_jobs(limit=5000)}
            if st.session_state.get("usrec_job_id") is None:
                req = {"series_id": "USREC", "start_date": "1947-01-01", "end_date": today.isoformat(), "n": 1200}
                st.session_state.usrec_job_id = submit_provider_fetch(
                    runner=runner,
                    kind="macro",
                    provider_id="fred",
                    request=req,
                )
                st.info("Fetching USREC for recession shading. Click Refresh results once it finishes.")
            else:
                jid = st.session_state.get("usrec_job_id")
                j = jobs_by_id.get(jid)
                if j and j.status == "SUCCEEDED" and j.output_path:
                    if st.session_state.get("usrec_intervals") is None:
                        try:
                            df_usrec = _artifact_df(j.output_path)
                            st.session_state.usrec_intervals = build_recession_intervals(df_usrec)
                        except Exception as e:
                            st.warning(f"Could not build recession intervals from USREC artifact: {e}")
                            st.session_state.usrec_intervals = []
                    recession_intervals = st.session_state.get("usrec_intervals") or []
                elif j and j.status == "FAILED":
                    st.warning("USREC fetch failed; recession shading disabled for now.")

    with st.expander("Query + data pull", expanded=True):
        _render_query_and_submit(
            mode=mode,
            cfg=cfg,
            storage=storage,
            runner=runner,
            reg=reg,
            start_date=start_date,
            end_date=end_date,
        )

    st.divider()
    with st.expander("Results (latest run)", expanded=True):
        _render_results(
            storage=storage,
            runner=runner,
            start_date=start_date,
            end_date=end_date,
            recession_intervals=recession_intervals,
        )


if __name__ == "__main__":
    main()

