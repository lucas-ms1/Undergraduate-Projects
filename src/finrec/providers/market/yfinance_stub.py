from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import random

import pandas as pd

from finrec.providers.base import Provider, ProviderMeta


@dataclass
class YFinanceStubProvider(Provider):
    meta: ProviderMeta = ProviderMeta(
        id="yfinance_stub",
        name="yfinance (stub)",
        kind="market",
    )

    def fetch(self, request: dict, ctx) -> pd.DataFrame:
        symbol = str(request.get("symbol", "AAPL")).upper()
        n = int(request.get("n", 30))

        start_date = request.get("start_date")
        end_date = request.get("end_date")

        if start_date and end_date:
            start = date.fromisoformat(str(start_date))
            end = date.fromisoformat(str(end_date))
            if end < start:
                raise ValueError(f"end_date ({end}) must be >= start_date ({start})")
            # daily synthetic series
            num_days = (end - start).days + 1
            dates = [start + timedelta(days=i) for i in range(num_days)]
            n_hint = None
        else:
            ctx.log("INFO", f"[{self.meta.id}] Using fallback n={n} (no start/end provided)")
            end = datetime.utcnow().date()
            dates = [end - timedelta(days=i) for i in reversed(range(n))]
            n_hint = n

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Generating synthetic market data for symbol={symbol}, n={n_hint}, "
            f"start={dates[0].isoformat()}, end={dates[-1].isoformat()}",
        )

        price = 100.0
        rows = []
        for d in dates:
            price *= (1.0 + random.uniform(-0.02, 0.02))
            rows.append({"date": d.isoformat(), "symbol": symbol, "close": round(price, 4)})

        return pd.DataFrame(rows)

