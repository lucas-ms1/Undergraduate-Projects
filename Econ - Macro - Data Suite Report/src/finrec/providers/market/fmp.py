from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from finrec.providers.base import Provider, ProviderMeta
from finrec.providers.utils.optional import require_optional


def _is_fmp_premium_symbol_error(*, status_code: int, body: str) -> bool:
    """
    Detect the common FMP failure mode for free-tier symbols.

    Example:
      status=402, body="Premium Query Parameter ... value set for 'symbol' is not available ..."
    """
    if int(status_code) != 402:
        return False
    b = (body or "").lower()
    return ("not available under your current subscription" in b) or ("premium query parameter" in b)


@dataclass
class FMPProvider(Provider):
    """
    Financial Modeling Prep (FMP) market data provider.

    Uses the stable EOD history endpoint to retrieve daily OHLCV:
      https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=AAPL

    Notes
    -----
    - This provider currently supports daily bars only. If `interval` is provided and not "1d",
      it is ignored (with a log message).
    - Requires an API key via environment variable: FINREC_FMP_API_KEY
      (loaded from .env via finrec.config.load_config()).
    """

    meta: ProviderMeta = ProviderMeta(
        id="fmp",
        name="Financial Modeling Prep (FMP)",
        kind="market",
    )

    def fetch(self, request: dict, ctx) -> pd.DataFrame:
        requests = require_optional("requests", extra_hint="market")

        api_key = (os.getenv("FINREC_FMP_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "Missing FINREC_FMP_API_KEY. Add it to your .env file (not committed) and restart the app."
            )

        symbol = str(request.get("symbol", "AAPL")).upper()
        n = int(request.get("n", 252))
        interval = str(request.get("interval", "1d")).strip().lower()
        if interval and interval != "1d":
            ctx.log("WARNING", f"[{self.meta.id}] interval='{interval}' not supported; using daily (1d).")

        start_date = request.get("start_date")
        end_date = request.get("end_date")

        today = datetime.now(timezone.utc).date()
        if start_date and end_date:
            start_d = date.fromisoformat(str(start_date))
            end_d = date.fromisoformat(str(end_date))
            if end_d < start_d:
                raise ValueError(f"end_date ({end_d}) must be >= start_date ({start_d})")
            if end_d > today:
                end_d = today
                ctx.log("INFO", f"[{self.meta.id}] end_date clamped to today ({today})")
            if start_d > today:
                start_d = today
                ctx.log("INFO", f"[{self.meta.id}] start_date clamped to today ({today})")
            n_hint = None
        else:
            # Fallback: buffer window then tail(n)
            end_d = today
            start_d = end_d - timedelta(days=max(10, int(n * 2.5)))
            n_hint = n

        # Use the non-legacy stable endpoint (FMP has deprecated many /api/v3 "legacy" endpoints).
        url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        params = {
            "symbol": symbol,
            "from": start_d.isoformat(),
            "to": end_d.isoformat(),
            "apikey": api_key,
        }

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Fetching: symbol={symbol}, n={n_hint}, "
            f"start={start_d.isoformat()}, end={end_d.isoformat()}",
        )

        resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "macro-project-1-data-suite/0.1"})
        if resp.status_code != 200:
            body = (resp.text or "")[:500]
            if _is_fmp_premium_symbol_error(status_code=resp.status_code, body=body):
                ctx.log(
                    "WARNING",
                    f"[{self.meta.id}] Symbol={symbol} not available on your FMP plan; "
                    "falling back to yfinance.",
                )
                try:
                    # Import only on demand so FMP can be used without yfinance installed.
                    from finrec.providers.market.yfinance import YFinanceProvider

                    return YFinanceProvider().fetch(request, ctx=ctx)
                except Exception as yf_err:
                    raise ValueError(
                        "FMP symbol requires a paid plan and yfinance fallback also failed. "
                        f"FMP status={resp.status_code}, body={body}"
                    ) from yf_err
            raise ValueError(f"FMP request failed: status={resp.status_code}, body={body}")

        data: Any = None
        try:
            data = resp.json()
        except Exception as e:
            raise ValueError(f"FMP returned non-JSON. Body head: {(resp.text or '')[:500]}") from e

        # Response shape can vary by endpoint: either a list[dict] or a dict wrapper.
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            hist = None
            for key in ["historical", "data", "results"]:
                if isinstance(data.get(key), list):
                    hist = data.get(key)
                    break
            if hist is None:
                # Some error responses come back as JSON objects with an error message.
                msg = (data.get("Error Message") or data.get("error") or data.get("message") or "").strip()
                if msg:
                    if _is_fmp_premium_symbol_error(status_code=402, body=msg):
                        ctx.log(
                            "WARNING",
                            f"[{self.meta.id}] Symbol={symbol} not available on your FMP plan; "
                            "falling back to yfinance.",
                        )
                        try:
                            from finrec.providers.market.yfinance import YFinanceProvider

                            return YFinanceProvider().fetch(request, ctx=ctx)
                        except Exception as yf_err:
                            raise ValueError(
                                "FMP symbol requires a paid plan and yfinance fallback also failed. "
                                f"FMP message={msg[:300]}"
                            ) from yf_err
                    raise ValueError(f"FMP request failed: {msg}")
                raise ValueError(f"Unexpected FMP response shape. Keys: {list(data.keys())}")
            if not hist:
                raise ValueError(f"FMP returned no historical data for symbol={symbol}")
            df = pd.DataFrame(hist)
        else:
            raise ValueError(f"Unexpected FMP response type: {type(data)}")

        # Common FMP fields: date, open, high, low, close, volume
        if "date" not in df.columns:
            raise ValueError(f"FMP response missing 'date'. Columns: {list(df.columns)}")

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        out["symbol"] = symbol
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                out[c] = pd.to_numeric(df[c], errors="coerce")

        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if n_hint is not None:
            out = out.tail(n_hint).reset_index(drop=True)

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}, cols={list(out.columns)}")
        return out

