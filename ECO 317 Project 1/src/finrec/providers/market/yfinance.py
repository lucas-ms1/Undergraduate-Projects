from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import threading
import time

import pandas as pd

from finrec.providers.base import Provider, ProviderMeta
from finrec.providers.utils.optional import require_optional


_YF_LOCK = threading.Lock()
_YF_LAST_REQUEST_AT = 0.0


def _throttle_yfinance(min_interval_s: float = 3.0) -> None:
    """
    Best-effort global throttle for yfinance/Yahoo requests.

    yfinance can be rate-limited unpredictably; limiting burstiness (especially with concurrent jobs)
    materially reduces YFRateLimitError frequency.
    """
    global _YF_LAST_REQUEST_AT
    min_interval_s = float(min_interval_s)
    if min_interval_s <= 0:
        return
    with _YF_LOCK:
        now = time.time()
        wait_s = (_YF_LAST_REQUEST_AT + min_interval_s) - now
        if wait_s > 0:
            time.sleep(wait_s)
        _YF_LAST_REQUEST_AT = time.time()


@dataclass
class YFinanceProvider(Provider):
    meta: ProviderMeta = ProviderMeta(
        id="yfinance",
        name="yfinance",
        kind="market",
    )

    def fetch(self, request: dict, ctx) -> pd.DataFrame:
        yf = require_optional("yfinance", extra_hint="market")
        try:
            from yfinance.exceptions import YFRateLimitError  # type: ignore
        except Exception:  # pragma: no cover
            YFRateLimitError = Exception  # type: ignore

        symbol = str(request.get("symbol", "AAPL")).upper()
        n = int(request.get("n", 30))
        interval = str(request.get("interval", "1d"))

        start_date = request.get("start_date")
        end_date = request.get("end_date")

        if start_date and end_date:
            # Date-range mode: honor the requested range.
            start_d = date.fromisoformat(str(start_date))
            end_d = date.fromisoformat(str(end_date))
            if end_d < start_d:
                raise ValueError(f"end_date ({end_d}) must be >= start_date ({start_d})")

            # yfinance history end is exclusive-ish; add 1 day to include end_date.
            start = datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc)
            end = datetime.combine(end_d, datetime.min.time(), tzinfo=timezone.utc)
            n_hint = None
        else:
            # Fallback mode: buffer for weekends/holidays; pull more than needed then tail(n)
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=max(10, int(n * 2.5)))
            n_hint = n

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Fetching: symbol={symbol}, n={n_hint}, interval={interval}, "
            f"start={start.date().isoformat()}, end={end.date().isoformat()}",
        )

        ticker = yf.Ticker(symbol)

        # yfinance is frequently rate-limited; retry with backoff.
        hist = None
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                _throttle_yfinance(float(request.get("throttle_s", 3.0)))
                hist = ticker.history(
                    start=start,
                    end=end + timedelta(days=1),
                    interval=interval,
                    auto_adjust=False,
                )
                last_err = None
                break
            except YFRateLimitError as e:  # type: ignore[misc]
                last_err = e
                # When Yahoo throttles, short backoffs often aren't enough.
                wait_s = [3.0, 10.0, 30.0, 60.0][attempt]
                ctx.log("WARNING", f"[{self.meta.id}] Rate-limited (attempt={attempt+1}/4). Sleeping {wait_s}s.")
                time.sleep(wait_s)
            except Exception as e:
                last_err = e
                break

        if last_err is not None:
            raise ValueError(
                "yfinance request failed. If you see rate limiting, wait a bit and try again later."
            ) from last_err

        if hist is None or hist.empty:
            raise ValueError(f"yfinance returned no data for symbol={symbol}")

        # Normalize columns
        hist = hist.reset_index()
        hist.columns = [str(c).lower() for c in hist.columns]

        # yfinance uses "date" for daily and "datetime" for intraday in some cases
        if "date" in hist.columns:
            dt_col = "date"
        elif "datetime" in hist.columns:
            dt_col = "datetime"
        else:
            # fallback: first column is usually the time index
            dt_col = hist.columns[0]

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(hist[dt_col], utc=True, errors="coerce").dt.date.astype(str)
        out["symbol"] = symbol

        # Standard OHLCV
        for c in ["open", "high", "low", "close", "volume"]:
            if c in hist.columns:
                out[c] = pd.to_numeric(hist[c], errors="coerce")

        out = out.dropna(subset=["date"])
        if n_hint is not None:
            # Keep last n observations (fallback mode)
            out = out.tail(n_hint)
        out = out.reset_index(drop=True)

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}, cols={list(out.columns)}")
        return out

