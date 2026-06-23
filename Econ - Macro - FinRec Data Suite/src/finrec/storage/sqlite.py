from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class JobRow:
    job_id: str
    created_at: str
    updated_at: str
    status: str
    kind: str
    provider_kind: str
    provider_id: str
    request_json: str
    output_path: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class LogRow:
    id: int
    job_id: str
    ts: str
    level: str
    message: str


class SQLiteStorage:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path.as_posix(), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        expected_jobs_cols = [
            "job_id",
            "created_at",
            "updated_at",
            "status",
            "kind",
            "provider_kind",
            "provider_id",
            "request_json",
            "output_path",
            "error",
        ]
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs';"
            ).fetchone()
            if existing:
                cols = [r["name"] for r in conn.execute("PRAGMA table_info(jobs);").fetchall()]
                if cols != expected_jobs_cols:
                    conn.execute("DROP TABLE IF EXISTS job_logs;")
                    conn.execute("DROP TABLE IF EXISTS jobs;")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    provider_kind TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    output_path TEXT,
                    error TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                """
            )
            conn.commit()

    def create_job(
        self,
        job_id: str,
        *,
        kind: str,
        provider_kind: str,
        provider_id: str,
        request: Dict[str, Any],
    ) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, created_at, updated_at, status, kind, provider_kind, provider_id, request_json, output_path, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL);
                """,
                (
                    job_id,
                    now,
                    now,
                    "QUEUED",
                    kind,
                    provider_kind,
                    provider_id,
                    json.dumps(request, sort_keys=True),
                ),
            )
            conn.commit()

    def set_status(self, job_id: str, status: str) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?;",
                (status, now, job_id),
            )
            conn.commit()

    def set_output_path(self, job_id: str, output_path: str) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET output_path = ?, updated_at = ? WHERE job_id = ?;",
                (output_path, now, job_id),
            )
            conn.commit()

    def set_error(self, job_id: str, error: str) -> None:
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET error = ?, updated_at = ? WHERE job_id = ?;",
                (error, now, job_id),
            )
            conn.commit()

    def append_log(self, job_id: str, level: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO job_logs (job_id, ts, level, message) VALUES (?, ?, ?, ?);",
                (job_id, _utc_now_iso(), level, message),
            )
            conn.commit()

    def list_jobs(self, limit: int = 50) -> List[JobRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        return [JobRow(**dict(r)) for r in rows]

    def list_logs(self, job_id: str, limit: int = 200) -> List[LogRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM job_logs WHERE job_id = ? ORDER BY id ASC LIMIT ?;",
                (job_id, limit),
            ).fetchall()
        return [LogRow(**dict(r)) for r in rows]
