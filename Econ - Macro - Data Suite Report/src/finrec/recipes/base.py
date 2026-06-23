from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class RecipeMeta:
    id: str
    name: str
    description: str


class JobContext(Protocol):
    def log(self, level: str, message: str) -> None: ...


class Recipe(Protocol):
    meta: RecipeMeta

    def run(self, df: pd.DataFrame, params: dict, ctx: JobContext) -> pd.DataFrame:
        """
        Run a transformation on an input dataframe.
        Must return a dataframe (to be persisted as CSV).
        """
        ...

