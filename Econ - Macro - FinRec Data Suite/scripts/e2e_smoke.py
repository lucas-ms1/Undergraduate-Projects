"""E2E smoke test using real providers. Requires network and optional deps: yfinance, requests, pandas-datareader."""
from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path

import pandas as pd

from finrec.jobs.runner import JobRunner
from finrec.storage.sqlite import SQLiteStorage
from finrec.ui.pipelines import submit_provider_fetch, submit_recipe_run


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="finrec_smoke_"))
    db = base / "finrec.db"
    outdir = base / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStorage(db)
    storage.init_schema()
    runner = JobRunner(storage, outdir)

    try:
        # Finance
        market_provider = "fmp" if (os.getenv("FINREC_FMP_API_KEY") or "").strip() else "yfinance"
        market_job = submit_provider_fetch(
            runner=runner,
            kind="market",
            provider_id=market_provider,
            request={"symbol": "AAPL", "start_date": "2020-01-01", "end_date": "2020-02-01", "n": 30},
        )
        news_job = submit_provider_fetch(
            runner=runner,
            kind="news",
            provider_id="gdelt",
            request={"query": "AAPL", "start_date": "2020-01-01", "end_date": "2020-02-01", "n": 5},
        )

        for _ in range(200):
            jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
            if (
                jobs.get(market_job)
                and jobs.get(news_job)
                and jobs[market_job].status == "SUCCEEDED"
                and jobs[news_job].status == "SUCCEEDED"
            ):
                break
            time.sleep(0.05)

        jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
        assert jobs[market_job].output_path
        assert jobs[news_job].output_path

        df_mkt = pd.read_csv(jobs[market_job].output_path)
        df_news = pd.read_csv(jobs[news_job].output_path)
        assert "close" in df_mkt.columns
        assert "title" in df_news.columns

        sma_job = submit_recipe_run(
            runner=runner,
            input_job_id=market_job,
            input_path=jobs[market_job].output_path,
            recipe_id="sma",
            params={"value_col": "close", "window": 5, "out_col": "sma_5"},
        )

        for _ in range(200):
            jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
            if jobs.get(sma_job) and jobs[sma_job].status == "SUCCEEDED":
                break
            time.sleep(0.05)

        jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
        assert jobs[sma_job].output_path
        df_sma = pd.read_csv(jobs[sma_job].output_path)
        assert "sma_5" in df_sma.columns

        # Econ
        fred_job = submit_provider_fetch(
            runner=runner,
            kind="macro",
            provider_id="fred",
            request={"series_id": "CPIAUCSL", "start_date": "2018-01-01", "end_date": "2020-01-01", "n": 24},
        )

        for _ in range(200):
            jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
            if jobs.get(fred_job) and jobs[fred_job].status == "SUCCEEDED":
                break
            time.sleep(0.05)

        jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
        assert jobs[fred_job].output_path

        # Macro transform: inflation YoY
        infl_job = submit_recipe_run(
            runner=runner,
            input_job_id=fred_job,
            input_path=jobs[fred_job].output_path,
            recipe_id="inflation_yoy",
            params={"date_col": "date", "value_col": "value", "out_col": "inflation_yoy"},
        )

        for _ in range(200):
            jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
            if jobs.get(infl_job) and jobs[infl_job].status == "SUCCEEDED":
                break
            time.sleep(0.05)

        jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
        assert jobs[infl_job].output_path
        df_infl = pd.read_csv(jobs[infl_job].output_path)
        assert "inflation_yoy" in df_infl.columns

        ar1_job = submit_recipe_run(
            runner=runner,
            input_job_id=fred_job,
            input_path=jobs[fred_job].output_path,
            recipe_id="ar1",
            params={"y_col": "value"},
        )

        for _ in range(200):
            jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
            if jobs.get(ar1_job) and jobs[ar1_job].status == "SUCCEEDED":
                break
            time.sleep(0.05)

        jobs = {j.job_id: j for j in storage.list_jobs(limit=50)}
        assert jobs[ar1_job].output_path
        df_ar1 = pd.read_csv(jobs[ar1_job].output_path)
        assert "term" in df_ar1.columns
    finally:
        runner._executor.shutdown(wait=True)  # type: ignore[attr-defined]

    print(f"e2e_ok (artifacts in {base})")


if __name__ == "__main__":
    main()

