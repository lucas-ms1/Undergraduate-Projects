from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from finrec.providers.base import Provider, ProviderMeta
from finrec.providers.utils.optional import require_optional


@dataclass
class FREDProvider(Provider):
    meta: ProviderMeta = ProviderMeta(
        id="fred",
        name="FRED (pandas-datareader)",
        kind="macro",
    )

    def fetch(self, request: dict, ctx) -> pd.DataFrame:
        pdr_data = require_optional("pandas_datareader.data", extra_hint="macro")

        series_id = str(request.get("series_id", "CPIAUCSL")).upper()
        n = int(request.get("n", 24))

        start_date = request.get("start_date")
        end_date = request.get("end_date")

        if start_date and end_date:
            start = date.fromisoformat(str(start_date))
            end = date.fromisoformat(str(end_date))
            if end < start:
                raise ValueError(f"end_date ({end}) must be >= start_date ({start})")
            n_hint = None
        else:
            # Fallback mode: buffered window then tail(n)
            end = datetime.now(timezone.utc).date()
            start = end - timedelta(days=max(120, 35 * n))
            n_hint = n

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Fetching: series_id={series_id}, n={n_hint}, "
            f"start={start.isoformat()}, end={end.isoformat()}",
        )

        df = pdr_data.DataReader(series_id, "fred", start, end)
        if df is None or df.empty:
            raise ValueError(f"FRED returned no data for series_id={series_id}")

        df = df.reset_index()
        # pandas-datareader FRED index column is usually "DATE"
        date_col = "DATE" if "DATE" in df.columns else df.columns[0]

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
        out["series_id"] = series_id
        out["value"] = pd.to_numeric(df[series_id], errors="coerce")

        out = out.dropna(subset=["date"])
        if n_hint is not None:
            out = out.tail(n_hint)
        out = out.reset_index(drop=True)

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out

