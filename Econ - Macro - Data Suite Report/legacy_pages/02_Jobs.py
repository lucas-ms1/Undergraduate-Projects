from __future__ import annotations

import streamlit as st


def _get_singletons():
    storage = st.session_state.finrec_storage
    return storage


st.set_page_config(page_title="Jobs", layout="wide")
st.title("Jobs")

storage = _get_singletons()

top = st.columns([1, 1, 2, 2])
with top[0]:
    if st.button("Refresh"):
        st.rerun()
with top[1]:
    limit = st.number_input("Limit", min_value=10, max_value=500, value=100, step=10)

jobs = storage.list_jobs(limit=int(limit))
if not jobs:
    st.info("No jobs yet. Submit one from Data Pull.")
    st.stop()

kinds = sorted(set(j.kind for j in jobs))
kind_filter = st.multiselect("Filter by kind", kinds, default=kinds)

filtered = [j for j in jobs if j.kind in kind_filter]

st.subheader("Job list")
st.dataframe([j.__dict__ for j in filtered], use_container_width=True)

st.subheader("Logs / detail")
job_ids = [j.job_id for j in filtered]
selected = st.selectbox("Select job_id", job_ids)

job = next(j for j in jobs if j.job_id == selected)
st.write(f"**status:** `{job.status}`")
st.write(f"**kind:** `{job.kind}`")
st.write(f"**provider:** `{job.provider_kind}:{job.provider_id}`")
st.write(f"**output_path:** `{job.output_path}`")
if job.error:
    st.error(job.error)

logs = storage.list_logs(selected, limit=500)
if not logs:
    st.caption("No logs yet for this job.")
else:
    for row in logs:
        st.write(f"`{row.ts}` **{row.level}** — {row.message}")

