from __future__ import annotations

from pathlib import Path

from finrec.storage.sqlite import SQLiteStorage


def test_sqlite_storage_job_lifecycle(tmp_path: Path):
    db_path = tmp_path / "finrec.db"
    st = SQLiteStorage(db_path)
    st.init_schema()

    job_id = "job123"
    st.create_job(
        job_id,
        kind="provider_fetch",
        provider_kind="market",
        provider_id="yfinance_stub",
        request={"symbol": "AAPL"},
    )

    st.set_status(job_id, "RUNNING")
    st.append_log(job_id, "INFO", "hello")
    st.set_output_path(job_id, str(tmp_path / "out.csv"))
    st.set_status(job_id, "SUCCEEDED")

    jobs = st.list_jobs(limit=10)
    assert len(jobs) == 1
    assert jobs[0].job_id == job_id
    assert jobs[0].status == "SUCCEEDED"
    assert jobs[0].output_path is not None

    logs = st.list_logs(job_id, limit=10)
    assert len(logs) == 1
    assert logs[0].level == "INFO"

