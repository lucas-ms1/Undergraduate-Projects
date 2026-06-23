from __future__ import annotations

import pandas as pd
import streamlit as st

from finrec.datasets.merge import build_merged_dataset


def _get_singletons():
    storage = st.session_state.finrec_storage
    runner = st.session_state.finrec_runner
    return storage, runner


def _load_columns(path: str) -> list[str]:
    # Read only the header (and a few rows) to keep UI responsive.
    df = pd.read_csv(path, nrows=5)
    return [str(c) for c in df.columns]


def _default_date_col(cols: list[str]) -> str:
    for c in ["date", "ts", "datetime"]:
        if c in cols:
            return c
    return cols[0] if cols else "date"


def _default_value_col(cols: list[str]) -> str:
    for c in ["close", "value", "sentiment", "score"]:
        if c in cols:
            return c
    # pick the first non-date-ish numeric-ish column name
    for c in cols:
        if c.lower() not in {"date", "ts", "datetime", "symbol", "series_id", "query", "title", "snippet", "source", "url"}:
            return c
    return cols[0] if cols else "value"


st.set_page_config(page_title="Datasets", layout="wide")
st.title("Datasets — Build an aligned time series dataset")

storage, runner = _get_singletons()
jobs = storage.list_jobs(limit=500)
completed = [j for j in jobs if j.status == "SUCCEEDED" and j.output_path]

if not completed:
    st.info("No completed artifacts yet. Submit a job from Data Pull or run a Recipe first.")
    st.stop()

label_to_job = {f"{j.job_id} — {j.kind} — {j.provider_kind}:{j.provider_id}": j for j in completed}
labels = list(label_to_job.keys())

st.markdown("### Select inputs")
selected_labels = st.multiselect(
    "Completed artifacts to merge",
    labels,
    default=labels[:2] if len(labels) >= 2 else labels,
)

if len(selected_labels) < 1:
    st.caption("Pick at least one artifact to build a dataset.")
    st.stop()

merge_how = st.selectbox("Join type", ["outer", "inner"], index=0)
ffill = st.checkbox("Forward-fill within each series after merge", value=False)

st.markdown("### Map inputs")
inputs: list[dict] = []

for i, lab in enumerate(selected_labels):
    job = label_to_job[lab]
    cols = _load_columns(job.output_path)

    with st.expander(f"Input {i+1}: {lab}", expanded=True):
        st.write(f"**path:** `{job.output_path}`")
        date_col = st.selectbox(
            "date column",
            cols,
            index=cols.index(_default_date_col(cols)) if cols else 0,
            key=f"date_col_{job.job_id}",
        )
        value_col = st.selectbox(
            "value column",
            cols,
            index=cols.index(_default_value_col(cols)) if cols else 0,
            key=f"value_col_{job.job_id}",
        )
        default_alias = job.provider_id or job.job_id[:8]
        alias = st.text_input(
            "output column name",
            value=default_alias,
            key=f"alias_{job.job_id}",
            help="This becomes the column name in the merged dataset.",
        )

        inputs.append(
            {
                "job_id": job.job_id,
                "path": job.output_path,
                "date_col": date_col,
                "value_col": value_col,
                "alias": alias,
            }
        )

st.markdown("### Run")
col1, col2 = st.columns([1, 2])
with col1:
    run_btn = st.button("Build dataset", type="primary")
with col2:
    st.caption("This submits a background job that loads the selected artifacts and merges them on date.")

if run_btn:

    def job_fn(ctx):
        ctx.log("INFO", f"Building dataset from {len(inputs)} input artifacts")
        ctx.log("INFO", f"merge_how={merge_how}, ffill={ffill}")
        merged = build_merged_dataset(inputs=inputs, merge_how=merge_how, ffill=ffill, log=ctx.log)
        ctx.log("INFO", f"Dataset built: rows={len(merged)}, cols={list(merged.columns)}")
        return merged

    request = {
        "merge_how": merge_how,
        "ffill": ffill,
        "inputs": inputs,
    }

    job_id = runner.submit(
        kind="dataset_build",
        provider_kind="dataset",
        provider_id="merge",
        request=request,
        fn=job_fn,
    )
    st.success(f"Submitted dataset build job: {job_id}")
    st.info("Go to Jobs to watch status/logs; Preview to download the merged dataset artifact.")

