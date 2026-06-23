from __future__ import annotations

import pandas as pd
import streamlit as st


def _get_singletons():
    storage = st.session_state.finrec_storage
    return storage


st.set_page_config(page_title="Preview", layout="wide")
st.title("Preview artifacts")

storage = _get_singletons()
jobs = storage.list_jobs(limit=500)

completed = [j for j in jobs if j.status == "SUCCEEDED" and j.output_path]
if not completed:
    st.info("No completed jobs with artifacts yet. Submit a job from Data Pull or run a Recipe.")
    st.stop()

label_to_job = {f"{j.job_id} — {j.kind} — {j.provider_kind}:{j.provider_id}": j for j in completed}
label = st.selectbox("Pick a completed job artifact", list(label_to_job.keys()))
job = label_to_job[label]

st.write(f"**output_path:** `{job.output_path}`")

try:
    df = pd.read_csv(job.output_path)
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{job.job_id}.csv",
        mime="text/csv",
        type="primary",
    )
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"Failed to load artifact: {e}")

