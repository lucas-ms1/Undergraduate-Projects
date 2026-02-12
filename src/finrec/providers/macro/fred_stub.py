from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import random

import pandas as pd

from finrec.providers.base import Provider, ProviderMeta


@dataclass
class FREDStubProvider(Provider):
    meta: ProviderMeta = ProviderMeta(
        id="fred_stub",
        name="FRED (stub)",
        kind="macro",
    )

    def fetch(self, request: dict, ctx) -> pd.DataFrame:
        series_id = str(request.get("series_id", "CPIAUCSL")).upper()
        n = int(request.get("n", 24))

        start_date = request.get("start_date")
        end_date = request.get("end_date")

        if start_date and end_date:
            start = date.fromisoformat(str(start_date))
            end = date.fromisoformat(str(end_date))
            if end < start:
                raise ValueError(f"end_date ({end}) must be >= start_date ({start})")
            # monthly-ish synthetic series: step 30 days
            dates = []
            cur = start
            while cur <= end:
                dates.append(cur)
                cur = cur + timedelta(days=30)
            if not dates:
                dates = [start]
            n_hint = None
        else:
            ctx.log("INFO", f"[{self.meta.id}] Using fallback n={n} (no start/end provided)")
            end = datetime.utcnow().date()
            dates = [end - timedelta(days=30 * i) for i in reversed(range(n))]
            n_hint = n

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Generating synthetic macro series for series_id={series_id}, n={n_hint}, "
            f"start={dates[0].isoformat()}, end={dates[-1].isoformat()}",
        )

        level = 200.0
        rows = []
        for d in dates:
            level *= (1.0 + random.uniform(-0.01, 0.02))
            rows.append({"date": d.isoformat(), "series_id": series_id, "value": round(level, 4)})

        return pd.DataFrame(rows)

