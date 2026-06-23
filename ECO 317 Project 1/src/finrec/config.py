from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    db_path: Path
    results_dir: Path


def load_config() -> AppConfig:
    """
    Loads configuration from environment variables with sensible local defaults.
    """
    load_dotenv(override=False)

    db_path = Path(os.getenv("FINREC_DB_PATH", "data/finrec.db"))
    results_dir = Path(os.getenv("FINREC_RESULTS_DIR", "data/results"))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(db_path=db_path, results_dir=results_dir)
