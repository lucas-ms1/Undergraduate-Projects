from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from finrec.config import load_config
from finrec.jobs.runner import JobRunner
from finrec.providers.registry import get_registry
from finrec.storage.sqlite import SQLiteStorage
from finrec.ui.pipelines import submit_provider_fetch, submit_recipe_run


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

    return st.session_state.finrec_config, st.session_state.finrec_storage, st.session_state.finrec_runner


def _parse_csv_list(raw: str) -> list[str]:
    items = [x.strip().upper() for x in (raw or "").split(",")]
    return [x for x in items if x]


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
            out.append(term)
    return " OR ".join(out)


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
            out.append(term)
    return " OR ".join(out)


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


def _safe_line_chart(df: pd.DataFrame, *, x_col: str, y_cols: list[str], height: int = 300) -> None:
    cols = [c for c in y_cols if c in df.columns]
    if x_col in df.columns and cols:
        st.line_chart(df.set_index(x_col)[cols], height=height)
    else:
        st.caption("No chartable columns found.")


def main():
    st.set_page_config(page_title="finrec-app", layout="wide")
    cfg, storage, runner = _get_singletons()

    st.title("finrec-app (single page)")

    top_left, top_right = st.columns([2, 1])
    with top_left:
        mode = st.radio("Mode", ["Finance", "Econ"], horizontal=True)
    with top_right:
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
        start_date = st.date_input("start_date", value=default_start)
    with dcol2:
        end_date = st.date_input("end_date", value=today)

    if end_date < start_date:
        st.error("end_date must be >= start_date")
        st.stop()

    # NEWS controls (shared)
    st.subheader("Query + data pull")
    if mode == "Finance":
        tickers_raw = st.text_input("Tickers (comma-separated)", value="AAPL")
        tickers = _parse_csv_list(tickers_raw)
        market_source = st.selectbox(
            "Market data source",
            ["auto (real if available)", "stub"],
            index=0,
            help="If you get rate-limited, switch to stub for reliable demo runs.",
        )
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
        indicators = st.multiselect(
            "Indicators",
            ["sma", "ema", "rsi", "macd", "rolling_vol"],
            default=["sma", "rsi"],
        )

        primary = st.selectbox("Primary series for indicators", tickers or ["AAPL"])
        news_auto = _auto_news_query_finance(tickers)
    else:
        series_raw = st.text_input("FRED series_id(s) (comma-separated)", value="CPIAUCSL")
        series_ids = [x.strip().upper() for x in series_raw.split(",") if x.strip()]
        macro_source = st.selectbox(
            "Macro data source",
            ["auto (real if available)", "stub"],
            index=0,
        )
        models = st.multiselect(
            "Models",
            ["ar1", "ols", "lp_irf"],
            default=["ar1"],
        )
        # Note: single-series FRED artifacts are long-form (columns: date, series_id, value),
        # so y_col should default to 'value' unless you build a wide merged dataset.
        y_col = st.text_input("Model y_col (after merge)", value="value")
        x_cols = st.text_input("Model x_cols (OLS, comma-separated)", value="")
        shock_col = st.text_input("LP-IRF shock_col", value="shock")
        horizons = st.number_input("LP-IRF horizons", min_value=1, max_value=60, value=12, step=1)
        lp_shock_mode = st.selectbox(
            "LP-IRF shock source",
            ["use shock_col from data", "use Δy (first difference) as shock"],
            index=1,
            help="If you don't have a shock series, Δy is a simple fallback so LP-IRF can run.",
        )

        news_auto = _auto_news_query_econ(series_ids)

    st.markdown("### News")
    news_source = st.selectbox(
        "News source",
        ["auto (real if available)", "stub"],
        index=0,
        help="GDELT is rate-limited; stub is useful for offline/demo runs.",
    )
    use_auto_news = st.checkbox("Use auto news query", value=True)
    if use_auto_news:
        news_query = news_auto
        st.code(news_query)
        st.caption("Uncheck to override the query text.")
    else:
        news_query = st.text_input("News query (override)", value=news_auto)
    maxrecords = st.number_input("News maxrecords (<=250)", min_value=1, max_value=250, value=50, step=5)
    run_sentiment = st.checkbox("Compute daily news sentiment (FinBERT)", value=False)

    run_btn = st.button("Run", type="primary")

    if run_btn:
        # Reset latest run state
        st.session_state.latest_run = {}

        if mode == "Finance":
            # Submit market fetch jobs
            market_jobs: dict[str, str] = {}
            for t in tickers:
                req = {
                    "symbol": t,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "interval": interval,
                    "n": 252,
                }
                # Select provider
                if market_source == "stub":
                    provider_id = "yfinance_stub"
                else:
                    provider_id = "yfinance" if any(p.meta.id == "yfinance" for p in reg.list("market")) else "yfinance_stub"
                job_id = submit_provider_fetch(runner=runner, kind="market", provider_id=provider_id, request=req)
                market_jobs[t] = job_id

            # Submit news fetch job
            if news_source == "stub":
                news_provider = "gdelt_stub"
            else:
                news_provider = "gdelt" if any(p.meta.id == "gdelt" for p in reg.list("news")) else "gdelt_stub"
            news_req = {
                "query": news_query,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "n": int(maxrecords),
            }
            news_job_id = submit_provider_fetch(runner=runner, kind="news", provider_id=news_provider, request=news_req)

            st.session_state.latest_run = {
                "mode": "Finance",
                "tickers": tickers,
                "primary": primary,
                "indicators": indicators,
                "market_jobs": market_jobs,
                "news_job_id": news_job_id,
                "run_sentiment": run_sentiment,
            }
        else:
            macro_jobs: dict[str, str] = {}
            for sid in series_ids:
                req = {
                    "series_id": sid,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "n": 120,
                }
                if macro_source == "stub":
                    provider_id = "fred_stub"
                else:
                    provider_id = "fred" if any(p.meta.id == "fred" for p in reg.list("macro")) else "fred_stub"
                job_id = submit_provider_fetch(runner=runner, kind="macro", provider_id=provider_id, request=req)
                macro_jobs[sid] = job_id

            if news_source == "stub":
                news_provider = "gdelt_stub"
            else:
                news_provider = "gdelt" if any(p.meta.id == "gdelt" for p in reg.list("news")) else "gdelt_stub"
            news_req = {
                "query": news_query,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "n": int(maxrecords),
            }
            news_job_id = submit_provider_fetch(runner=runner, kind="news", provider_id=news_provider, request=news_req)

            st.session_state.latest_run = {
                "mode": "Econ",
                "series_ids": series_ids,
                "models": models,
                "y_col": y_col,
                "x_cols": x_cols,
                "shock_col": shock_col,
                "lp_shock_mode": lp_shock_mode,
                "horizons": int(horizons),
                "macro_jobs": macro_jobs,
                "news_job_id": news_job_id,
                "run_sentiment": run_sentiment,
            }

        st.success("Submitted jobs. Click Refresh results to see outputs as they finish.")

    st.divider()
    st.subheader("Results (latest run)")

    latest = st.session_state.get("latest_run") or {}
    if not latest:
        st.caption("No run submitted yet.")
        return

    # Display job status + artifacts when ready
    jobs = storage.list_jobs(limit=500)
    jobs_by_id = {j.job_id: j for j in jobs}

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
        st.markdown("### Market data")
        st.write({t: _status_badge(jid) for t, jid in market_jobs.items()})
        for _t, jid in market_jobs.items():
            _show_job_error(jid)

        # If primary is ready, chart it + optionally show indicator artifacts (once computed)
        primary = latest.get("primary")
        primary_job = market_jobs.get(primary)
        if primary_job and jobs_by_id.get(primary_job) and jobs_by_id[primary_job].output_path:
            df_price = _artifact_df(jobs_by_id[primary_job].output_path)
            st.write(f"Artifact: `{jobs_by_id[primary_job].output_path}`")
            _download_df(df_price, label="Download primary series CSV", file_name=f"{primary_job}.csv")
            if "date" in df_price.columns:
                _safe_line_chart(df_price, x_col="date", y_cols=["close", "open", "high", "low"], height=300)
            st.dataframe(df_price.tail(100), use_container_width=True)

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

            st.markdown("### Indicator jobs")
            ind_jobs = latest.get("indicator_jobs", {})
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
                                _safe_line_chart(df_ind, x_col="date", y_cols=["close", col], height=250)
                            elif rid == "rsi":
                                _safe_line_chart(df_ind, x_col="date", y_cols=["rsi_14"], height=250)
                            elif rid == "macd":
                                _safe_line_chart(df_ind, x_col="date", y_cols=["macd_line", "macd_signal", "macd_hist"], height=250)
                            else:
                                _safe_line_chart(df_ind, x_col="date", y_cols=["vol_20"], height=250)
                        st.dataframe(df_ind.tail(100), use_container_width=True)

        st.markdown("### News")
        news_job_id = latest.get("news_job_id")
        if news_job_id:
            st.write({"news_job_id": news_job_id, "status": _status_badge(news_job_id)})
            _show_job_error(news_job_id)
            jn = jobs_by_id.get(news_job_id)
            if jn and jn.output_path:
                df_news = _artifact_df(jn.output_path)
                st.write(f"News artifact: `{jn.output_path}`")
                _download_df(df_news, label="Download news CSV", file_name=f"{news_job_id}.csv")
                cols = [c for c in ["date", "ts", "source", "title", "url"] if c in df_news.columns]
                st.dataframe(df_news[cols].head(50) if cols else df_news.head(50), use_container_width=True)
                if "url" in df_news.columns and "title" in df_news.columns:
                    st.markdown("#### Top articles")
                    for _, row in df_news.head(10).iterrows():
                        title = str(row.get("title", "")).strip()
                        url = str(row.get("url", "")).strip()
                        if title and url:
                            st.markdown(f"- [{title}]({url})")

                if latest.get("run_sentiment") and "news_sentiment_job_id" not in latest:
                    # Submit sentiment aggregation job
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
                                st.line_chart(df_sent.set_index("date")[mean_cols], height=250)
                            st.dataframe(df_sent, use_container_width=True)

    else:
        st.markdown("### Macro data")
        macro_jobs = latest.get("macro_jobs", {})
        st.write({sid: _status_badge(jid) for sid, jid in macro_jobs.items()})
        for _sid, jid in macro_jobs.items():
            _show_job_error(jid)

        # If first series is ready, show it and submit model jobs on that series artifact (basic)
        first_sid = (latest.get("series_ids") or [None])[0]
        first_job = macro_jobs.get(first_sid) if first_sid else None
        if first_job and jobs_by_id.get(first_job) and jobs_by_id[first_job].output_path:
            df_macro = _artifact_df(jobs_by_id[first_job].output_path)
            st.write(f"Artifact: `{jobs_by_id[first_job].output_path}`")
            _download_df(df_macro, label="Download series CSV", file_name=f"{first_job}.csv")
            if "date" in df_macro.columns and "value" in df_macro.columns:
                st.line_chart(df_macro.set_index("date")[["value"]], height=300)
            else:
                st.dataframe(df_macro, use_container_width=True)

            if "model_jobs" not in latest:
                model_jobs: dict[str, str] = {}
                for mid in latest.get("models", []):
                    if mid == "ar1":
                        params = {"y_col": "value"}
                    elif mid == "ols":
                        params = {"y_col": latest.get("y_col", "value"), "x_cols": latest.get("x_cols", ""), "add_constant": True}
                    else:
                        # If the selected artifact doesn't have a shock column, we can create a simple shock
                        # as first-difference of y (Δy) so LP-IRF can run.
                        if latest.get("lp_shock_mode") == "use Δy (first difference) as shock":
                            df_tmp = df_macro.copy()
                            yname = latest.get("y_col", "value")
                            if yname not in df_tmp.columns:
                                raise ValueError(f"LP-IRF y_col '{yname}' not found in artifact. Columns: {list(df_tmp.columns)}")
                            df_tmp["shock"] = pd.to_numeric(df_tmp[yname], errors="coerce").diff()
                            tmp_path = jobs_by_id[first_job].output_path + ".lp_shock.csv"
                            df_tmp.to_csv(tmp_path, index=False)
                            input_job_id = first_job
                            input_path = tmp_path
                            shock_col = "shock"
                        else:
                            input_job_id = first_job
                            input_path = jobs_by_id[first_job].output_path
                            shock_col = latest.get("shock_col", "shock")
                        params = {
                            "y_col": latest.get("y_col", "value"),
                            "shock_col": shock_col,
                            "controls": "",
                            "horizons": latest.get("horizons", 12),
                            "ci_level": 0.95,
                            "add_constant": True,
                        }
                    jid = submit_recipe_run(
                        runner=runner,
                        input_job_id=input_job_id,
                        input_path=input_path,
                        recipe_id=mid,
                        params=params,
                    )
                    model_jobs[mid] = jid
                latest["model_jobs"] = model_jobs
                st.session_state.latest_run = latest

            st.markdown("### Model jobs")
            model_jobs = latest.get("model_jobs", {})
            st.write({mid: _status_badge(jid) for mid, jid in model_jobs.items()})
            for mid, jid in model_jobs.items():
                j = jobs_by_id.get(jid)
                if j and j.output_path:
                    df_out = _artifact_df(j.output_path)
                    with st.expander(f"{mid} output ({jid})", expanded=False):
                        st.write(f"`{j.output_path}`")
                        _download_df(df_out, label=f"Download {mid} output CSV", file_name=f"{jid}.csv")
                        if mid == "lp_irf" and "horizon" in df_out.columns and "irf" in df_out.columns:
                            _safe_line_chart(df_out, x_col="horizon", y_cols=["irf", "ci_low", "ci_high"], height=250)
                        st.dataframe(df_out, use_container_width=True)

        st.markdown("### News")
        news_job_id = latest.get("news_job_id")
        if news_job_id:
            st.write({"news_job_id": news_job_id, "status": _status_badge(news_job_id)})
            _show_job_error(news_job_id)
            jn = jobs_by_id.get(news_job_id)
            if jn and jn.output_path:
                df_news = _artifact_df(jn.output_path)
                st.write(f"News artifact: `{jn.output_path}`")
                _download_df(df_news, label="Download news CSV", file_name=f"{news_job_id}.csv")
                cols = [c for c in ["date", "ts", "source", "title", "url"] if c in df_news.columns]
                st.dataframe(df_news[cols].head(50) if cols else df_news.head(50), use_container_width=True)
                if "url" in df_news.columns and "title" in df_news.columns:
                    st.markdown("#### Top articles")
                    for _, row in df_news.head(10).iterrows():
                        title = str(row.get("title", "")).strip()
                        url = str(row.get("url", "")).strip()
                        if title and url:
                            st.markdown(f"- [{title}]({url})")

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
                                st.line_chart(df_sent.set_index("date")[mean_cols], height=250)
                            st.dataframe(df_sent, use_container_width=True)


if __name__ == "__main__":
    main()

