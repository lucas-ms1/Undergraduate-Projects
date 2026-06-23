from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from finrec.storage.sqlite import SQLiteStorage


@dataclass
class JobContext:
    job_id: str
    storage: SQLiteStorage

    def log(self, level: str, message: str) -> None:
        self.storage.append_log(self.job_id, level, message)


JobFn = Callable[[JobContext], Optional[pd.DataFrame]]


class JobRunner:
    """
    Prompt #1: thread-based runner. Later prompts can swap to a separate worker process/queue.
    """

    def __init__(self, storage: SQLiteStorage, results_dir: Path):
        self.storage = storage
        self.results_dir = Path(results_dir)

        # Lazy import to keep the mental model simple.
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=2)
        self._futures: Dict[str, object] = {}

    def submit(
        self,
        *,
        kind: str,
        provider_kind: str,
        provider_id: str,
        request: dict,
        fn: JobFn,
    ) -> str:
        job_id = uuid.uuid4().hex
        self.storage.create_job(
            job_id,
            kind=kind,
            provider_kind=provider_kind,
            provider_id=provider_id,
            request=request,
        )
        ctx = JobContext(job_id=job_id, storage=self.storage)

        def _run() -> None:
            try:
                self.storage.set_status(job_id, "RUNNING")
                ctx.log("INFO", f"Job started: kind={kind}, provider={provider_kind}:{provider_id}")

                df = fn(ctx)

                # If the job produced a dataframe, persist it as a CSV artifact.
                if df is not None:
                    out_path = self.results_dir / f"{job_id}.csv"
                    df.to_csv(out_path, index=False)
                    self.storage.set_output_path(job_id, out_path.as_posix())
                    ctx.log("INFO", f"Wrote artifact: {out_path.as_posix()} (rows={len(df)})")
                else:
                    ctx.log("INFO", "No artifact produced (df=None).")

                self.storage.set_status(job_id, "SUCCEEDED")
                ctx.log("INFO", "Job finished: SUCCEEDED")
            except Exception as e:
                self.storage.set_status(job_id, "FAILED")
                tb = traceback.format_exc()
                self.storage.set_error(job_id, f"{e}\n{tb}")
                ctx.log("ERROR", f"Job failed: {e}")
                ctx.log("ERROR", tb)

        fut = self._executor.submit(_run)
        self._futures[job_id] = fut
        return job_id

