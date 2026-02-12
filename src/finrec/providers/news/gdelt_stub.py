from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import random

import pandas as pd

from finrec.providers.base import Provider, ProviderMeta


@dataclass
class GDELTStubProvider(Provider):
    meta: ProviderMeta = ProviderMeta(
        id="gdelt_stub",
        name="GDELT (stub)",
        kind="news",
    )

    def fetch(self, request: dict, ctx) -> pd.DataFrame:
        query = str(request.get("query", "inflation"))
        n = int(request.get("n", 10))

        start_date = request.get("start_date")
        end_date = request.get("end_date")

        if start_date and end_date:
            start = date.fromisoformat(str(start_date))
            end = date.fromisoformat(str(end_date))
            if end < start:
                raise ValueError(f"end_date ({end}) must be >= start_date ({start})")
            # use the range as a time window and generate up to n hits within it
            base = datetime.combine(end, datetime.min.time())
            n_hint = None
        else:
            ctx.log("INFO", f"[{self.meta.id}] Using fallback n={n} (no start/end provided)")
            base = datetime.utcnow()
            n_hint = n

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Generating synthetic news hits for query='{query}', n={n_hint}",
        )

        rows = []
        for i in range(n):
            ts = base - timedelta(hours=i * random.randint(1, 6))
            rows.append(
                {
                    "ts": ts.isoformat(timespec="seconds"),
                    "query": query,
                    "title": f"Stub headline about {query} #{i+1}",
                    "snippet": f"Stub snippet mentioning {query}. This is synthetic.",
                    "source": random.choice(["Reuters", "WSJ", "Bloomberg", "FT", "AP"]),
                    "url": f"https://example.com/{query}/{i+1}",
                }
            )

        return pd.DataFrame(rows)

