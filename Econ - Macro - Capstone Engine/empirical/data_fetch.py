"""
empirical/data_fetch.py
=======================
Lightweight FRED and Yahoo Finance data retrieval for the Empirical Data Suite.

Uses fredapi for FRED series and yfinance for market data.
All retrieval functions are designed to be wrapped in @st.cache_data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# FRED data
# ---------------------------------------------------------------------------
def fetch_fred_series(
    series_ids: list[str],
    start: str = "1960-01-01",
    end: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch one or more FRED series and return a merged DataFrame.

    Parameters
    ----------
    series_ids : list of FRED series IDs (e.g., ["GDPC1", "CPIAUCSL", "UNRATE"])
    start, end : date range strings
    api_key : FRED API key (falls back to FRED_API_KEY env var)

    Returns
    -------
    pd.DataFrame with DatetimeIndex and one column per series
    """
    from fredapi import Fred
    import os

    key = api_key or os.getenv("FRED_API_KEY", "")
    if not key:
        raise ValueError(
            "No FRED API key found. Set FRED_API_KEY environment variable "
            "or pass api_key= directly."
        )

    fred = Fred(api_key=key)
    end = end or datetime.now().strftime("%Y-%m-%d")

    frames = {}
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.name = sid
            frames[sid] = s
        except Exception as e:
            frames[sid] = pd.Series(dtype=float, name=sid)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Yahoo Finance data
# ---------------------------------------------------------------------------
def fetch_yahoo_series(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str | None = None,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Fetch closing prices for one or more tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : list of ticker symbols (e.g., ["SPY", "^TNX"])
    start, end : date range
    column : which price column to extract (default: "Close")

    Returns
    -------
    pd.DataFrame with DatetimeIndex and one column per ticker
    """
    import yfinance as yf

    end = end or datetime.now().strftime("%Y-%m-%d")
    frames = {}
    for t in tickers:
        try:
            data = yf.download(t, start=start, end=end, progress=False)
            if not data.empty and column in data.columns:
                frames[t] = data[column]
            elif not data.empty:
                # Handle multi-level columns from yfinance
                frames[t] = data[column] if column in data.columns else data.iloc[:, 0]
        except Exception:
            frames[t] = pd.Series(dtype=float, name=t)

    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# NBER recession dates (for shading)
# ---------------------------------------------------------------------------
def fetch_recession_dates(api_key: str | None = None) -> list[tuple]:
    """
    Fetch USREC (recession indicator) from FRED and return a list of
    (start_date, end_date) tuples for each recession.
    """
    try:
        df = fetch_fred_series(["USREC"], start="1960-01-01", api_key=api_key)
        usrec = df["USREC"].dropna()
    except Exception:
        return []

    recessions = []
    in_recession = False
    start = None

    for date, val in usrec.items():
        if val == 1 and not in_recession:
            start = date
            in_recession = True
        elif val == 0 and in_recession:
            recessions.append((start, date))
            in_recession = False

    if in_recession and start is not None:
        recessions.append((start, usrec.index[-1]))

    return recessions
