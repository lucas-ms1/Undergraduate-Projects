from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from finrec.providers.registry import get_registry


def _get_singletons():
    # streamlit_app.py initializes these
    cfg = st.session_state.finrec_config
    storage = st.session_state.finrec_storage
    runner = st.session_state.finrec_runner
    return cfg, storage, runner


st.set_page_config(page_title="Data Pull", layout="wide")
st.title("Data Pull")

cfg, storage, runner = _get_singletons()
reg = get_registry()

kind = st.selectbox("Provider kind", ["market", "macro", "news"])
providers = reg.list(kind)

if not providers:
    st.warning("No providers registered for this kind.")
    st.stop()

provider_label_to_id = {f"{p.meta.name} ({p.meta.id})": p.meta.id for p in providers}
provider_label = st.selectbox("Provider", list(provider_label_to_id.keys()))
provider_id = provider_label_to_id[provider_label]

st.markdown("### Request")

request = {}

# Prefer date-range requests; keep n as an optional fallback.
today = date.today()
default_start = today - timedelta(days=365)
col_start, col_end = st.columns(2)
with col_start:
    start_date = st.date_input("start_date", value=default_start)
with col_end:
    end_date = st.date_input("end_date", value=today)

request["start_date"] = start_date.isoformat()
request["end_date"] = end_date.isoformat()

if kind == "market":
    request["symbol"] = st.text_input("symbol", value="AAPL")
    request["n"] = st.number_input(
        "n (rows, fallback)",
        min_value=5,
        max_value=5000,
        value=252,
        step=10,
        help="Used only if the provider cannot honor start/end dates.",
    )
    if provider_id == "yfinance":
        request["interval"] = st.selectbox(
            "interval",
            ["1d", "1h", "30m", "15m", "5m"],
            index=0,
        )
elif kind == "macro":
    request["series_id"] = st.text_input("series_id", value="CPIAUCSL")
    request["n"] = st.number_input(
        "n (rows, fallback)",
        min_value=6,
        max_value=5000,
        value=120,
        step=6,
        help="Used only if the provider cannot honor start/end dates.",
    )
else:
    request["query"] = st.text_input("query", value="inflation")
    request["n"] = st.number_input(
        "n (rows, fallback)",
        min_value=5,
        max_value=250,
        value=50,
        step=5,
        help="Used only if the provider cannot honor start/end dates.",
    )

st.markdown("### Submit")
col1, col2 = st.columns([1, 2])
with col1:
    submit = st.button("Submit job", type="primary")

with col2:
    st.caption("This uses a thread-based runner for Prompt #1. Later we can move to a true worker process.")

if submit:
    provider = reg.get(kind, provider_id)

    def job_fn(ctx):
        ctx.log("INFO", f"Executing provider fetch: {provider.meta.kind}:{provider.meta.id}")
        df = provider.fetch(request, ctx=ctx)
        ctx.log("INFO", f"Provider returned dataframe: shape={df.shape}")
        return df

    job_id = runner.submit(
        kind="provider_fetch",
        provider_kind=kind,
        provider_id=provider_id,
        request=request,
        fn=job_fn,
    )
    st.success(f"Submitted job: {job_id}")
    st.info("Go to the Jobs page to watch status/logs update.")

