from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import threading
import time
from typing import Any

import pandas as pd
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

from finrec.providers.base import Provider, ProviderMeta
from finrec.providers.utils.optional import require_optional

_GDELT_LOCK = threading.Lock()
_GDELT_LAST_REQUEST_AT = 0.0


def _gdelt_dt(d: date, *, is_end: bool) -> str:
    # GDELT expects YYYYMMDDHHMMSS
    if is_end:
        return d.strftime("%Y%m%d") + "235959"
    return d.strftime("%Y%m%d") + "000000"


def _normalize_gdelt_query(q: str) -> str:
    """
    GDELT query syntax notes:
    - If a query contains OR'd terms, GDELT requires parentheses around the OR group.
      Example: (AAPL OR Apple)
    - We only apply minimal normalization to avoid breaking advanced user queries.
    """
    q = (q or "").strip()
    if not q:
        return q

    # Wrap top-level OR queries in parentheses if not already wrapped.
    if " OR " in q and not (q.startswith("(") and q.endswith(")")):
        q = f"({q})"
    return q


def _append_sourcelang_filter(query: str, languages: list[str]) -> str:
    """
    Append one or more GDELT language filters to an existing query.

    GDELT doc API supports `sourcelang:<language>` (e.g., sourcelang:english).
    Multiple languages are OR'd: (sourcelang:english OR sourcelang:spanish)
    """
    langs = [str(x).strip().lower() for x in (languages or []) if str(x).strip()]
    if not langs:
        return query
    if len(langs) == 1:
        return f"{query} sourcelang:{langs[0]}"
    group = "(" + " OR ".join([f"sourcelang:{l}" for l in langs]) + ")"
    return f"{query} {group}"


def _throttle_gdelt(min_interval_s: float = 5.0) -> None:
    """
    GDELT explicitly asks for <= 1 request per 5 seconds (429 otherwise).
    Enforce a global per-process throttle so concurrent jobs don't trip the limit.
    """
    global _GDELT_LAST_REQUEST_AT
    with _GDELT_LOCK:
        now = time.time()
        wait_s = (_GDELT_LAST_REQUEST_AT + min_interval_s) - now
        if wait_s > 0:
            time.sleep(wait_s)
        _GDELT_LAST_REQUEST_AT = time.time()


@dataclass
class GDELTProvider(Provider):
    meta: ProviderMeta = ProviderMeta(
        id="gdelt",
        name="GDELT (doc API)",
        kind="news",
    )

    def fetch(self, request: dict, ctx) -> pd.DataFrame:
        requests = require_optional("requests", extra_hint="news")

        query = _normalize_gdelt_query(str(request.get("query", "inflation")))
        if not query:
            raise ValueError("query must be a non-empty string.")

        # Optional language filter(s) (GDELT `sourcelang:`).
        language = (request.get("language") or "").strip()
        languages = request.get("languages") or []
        langs: list[str] = []
        if isinstance(languages, list):
            langs = [str(x).strip() for x in languages if str(x).strip()]
        if language:
            langs = [language]
        if langs:
            query = _append_sourcelang_filter(query, langs)

        start_date = request.get("start_date")
        end_date = request.get("end_date")
        if not (start_date and end_date):
            raise ValueError("GDELTProvider requires start_date and end_date.")

        start_d = date.fromisoformat(str(start_date))
        end_d = date.fromisoformat(str(end_date))
        if end_d < start_d:
            raise ValueError(f"end_date ({end_d}) must be >= start_date ({start_d})")

        # GDELT v2 doc API maxrecords is typically capped (often 250).
        maxrecords = int(request.get("n", 50))
        maxrecords = max(1, min(maxrecords, 250))

        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": _gdelt_dt(start_d, is_end=False),
            "enddatetime": _gdelt_dt(end_d, is_end=True),
            "maxrecords": str(maxrecords),
            "sort": "datedesc",
        }

        ctx.log(
            "INFO",
            f"[{self.meta.id}] Fetching: query='{query}', start={start_d.isoformat()}, end={end_d.isoformat()}, "
            f"maxrecords={maxrecords}",
        )

        # Respect GDELT rate limit + retry on 429 with waits.
        resp = None
        for attempt in range(4):
            _throttle_gdelt(5.0)
            resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "finrec-app/0.1"})
            if resp.status_code == 200:
                break
            if resp.status_code == 429:
                wait_s = [0.0, 5.0, 10.0, 20.0][attempt]
                ctx.log(
                    "WARNING",
                    f"[{self.meta.id}] Rate-limited by GDELT (attempt={attempt+1}/4). Sleeping {wait_s}s.",
                )
                time.sleep(wait_s)
                continue
            break

        if resp is None or resp.status_code != 200:
            status = getattr(resp, "status_code", None)
            body = (getattr(resp, "text", "") or "")[:500]
            raise ValueError(f"GDELT request failed: status={status}, body={body}")

        # Sometimes GDELT returns a plain-text rate-limit message even with status=200.
        # Be defensive: if JSON decoding fails, retry.
        data: dict[str, Any] | None = None
        try:
            data = resp.json()
        except RequestsJSONDecodeError:
            txt = (resp.text or "").strip()
            ctx.log("WARNING", f"[{self.meta.id}] Non-JSON response (len={len(txt)}). Retrying as rate-limit.")
            # One more strict throttle + retry cycle
            _throttle_gdelt(5.0)
            time.sleep(5.0)
            resp2 = requests.get(url, params=params, timeout=30, headers={"User-Agent": "finrec-app/0.1"})
            if resp2.status_code != 200:
                raise ValueError(f"GDELT request failed: status={resp2.status_code}, body={(resp2.text or '')[:500]}")
            try:
                data = resp2.json()
            except RequestsJSONDecodeError as e:
                raise ValueError(f"GDELT returned non-JSON. Body head: {(resp2.text or '')[:500]}") from e

        assert data is not None
        articles = data.get("articles") or []
        if not articles:
            ctx.log("WARNING", f"[{self.meta.id}] No articles returned.")
            return pd.DataFrame(columns=["ts", "date", "query", "language", "title", "snippet", "source", "url"])

        rows: list[dict[str, Any]] = []
        for a in articles:
            # Common fields: seendate, url, title, sourceCountry/sourceCollection, domain, language, etc.
            seendate = a.get("seendate") or a.get("seenDate") or a.get("datetime")
            ts = None
            if seendate:
                # GDELT often returns YYYYMMDDTHHMMSSZ or similar; be tolerant.
                try:
                    ts = pd.to_datetime(seendate, utc=True, errors="coerce")
                except Exception:
                    ts = pd.NaT

            title = a.get("title") or ""
            snippet = a.get("summary") or a.get("snippet") or ""
            src = a.get("sourceCountry") or a.get("sourceCollection") or a.get("domain") or ""
            link = a.get("url") or ""
            lang = (a.get("language") or a.get("sourceLang") or a.get("sourcelang") or "").strip().lower()

            rows.append(
                {
                    "ts": ts.isoformat(timespec="seconds") if isinstance(ts, pd.Timestamp) and not pd.isna(ts) else "",
                    "date": ts.date().isoformat() if isinstance(ts, pd.Timestamp) and not pd.isna(ts) else "",
                    "query": query,
                    "language": lang,
                    "title": title,
                    "snippet": snippet,
                    "source": src,
                    "url": link,
                }
            )

        out = pd.DataFrame(rows)
        out = out.dropna(axis=0, how="all")
        out = out[out["title"].astype(str).str.len() > 0].reset_index(drop=True)

        ctx.log("INFO", f"[{self.meta.id}] Done. rows={len(out)}")
        return out

