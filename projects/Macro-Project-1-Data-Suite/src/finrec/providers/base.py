from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

ProviderKind = Literal["market", "macro", "news"]


@dataclass(frozen=True)
class ProviderMeta:
    id: str
    name: str
    kind: ProviderKind


class JobContext(Protocol):
    def log(self, level: str, message: str) -> None: ...


class Provider(Protocol):
    meta: ProviderMeta

    def fetch(self, request: dict, ctx: JobContext) -> pd.DataFrame:
        """
        Fetch data for a given request. For Prompt #1, implementations are stubs.
        """
        ...

