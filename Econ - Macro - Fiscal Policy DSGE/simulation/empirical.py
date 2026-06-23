"""
simulation/empirical.py
Step 8b – Empirical moments from FRED
----------------------------------------
Owner: Steps 8-9 contributor

Pulls US quarterly macroeconomic data from FRED, log-transforms levels,
HP-filters at λ=1600, and computes the same moments as moments.py so that
Tab 1 can display a direct model-vs-data comparison.

Series pulled:
  GDPC1    – Real GDP (output proxy)
  PCECC96  – Real personal consumption expenditures
  PNFIC1   – Real private nonresidential fixed investment
  HOABS    – Nonfarm business sector: hours of all persons
  CPIAUCSL – CPI All Urban Consumers (→ log-diff = inflation)

Caching: Results are cached to disk as a parquet file so the Streamlit app
can use @st.cache_data with ttl=86400 (24 h) without re-fetching on every
slider move.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.moments import compute_moments, hp_cycle

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

FRED_SERIES: Dict[str, str] = {
    "y":  "GDPC1",      # Real GDP
    "c":  "PCECC96",    # Real PCE
    "i":  "PNFIC1",     # Real nonresidential investment
    "l":  "HOABS",      # Aggregate hours
    "pi": "CPIAUCSL",   # CPI (will be log-differenced → inflation)
}

CACHE_PATH = Path(__file__).parent / "_fred_cache.csv"
HP_LAMBDA  = 1600.0
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60

# ──────────────────────────────────────────────────────────────────────────────
# FRED download
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_from_fred(
    start: str = "1960-01-01",
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download FRED series.  Tries fredapi first, falls back to
    pandas-datareader.
    """
    tickers = list(FRED_SERIES.values())

    # --- attempt 1: fredapi ---
    try:
        import fredapi  # type: ignore
        key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        if not key:
            raise EnvironmentError("No FRED_API_KEY set")
        fred = fredapi.Fred(api_key=key)
        frames = {t: fred.get_series(t, observation_start=start) for t in tickers}
        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.resample("QS").first()
        return df
    except Exception as e:
        warnings.warn(f"fredapi failed ({e}); trying pandas-datareader ...")

    # --- attempt 2: pandas-datareader ---
    try:
        import pandas_datareader.data as web   # type: ignore
        df = web.DataReader(tickers, "fred", start=start, end=end)
        df = df.resample("QS").first()
        return df
    except Exception as e2:
        raise RuntimeError(
            f"Could not retrieve FRED data via either fredapi or pandas-datareader.\n"
            f"Last error: {e2}\n"
            f"Set FRED_API_KEY env var or install pandas-datareader."
        ) from e2


# ──────────────────────────────────────────────────────────────────────────────
# Transform raw levels → log series
# ──────────────────────────────────────────────────────────────────────────────

def _transform(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Apply log-transform (and for CPI: first-difference → log-inflation).
    Align all series to the same length.
    """
    series_raw: Dict[str, np.ndarray] = {}

    for model_name, ticker in FRED_SERIES.items():
        col = df[ticker].dropna()

        if model_name == "pi":
            log_cpi = np.log(col.values.astype(float))
            arr = np.diff(log_cpi)
        else:
            arr = np.log(col.values.astype(float))

        series_raw[model_name] = arr

    # Align lengths (trim to shortest)
    min_len = min(len(v) for v in series_raw.values())
    return {k: v[-min_len:] for k, v in series_raw.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_empirical_moments(
    start: str = "1960-01-01",
    end: Optional[str] = None,
    fred_api_key: Optional[str] = None,
    force_refresh: bool = False,
    cache_ttl_seconds: Optional[int] = DEFAULT_CACHE_TTL_SECONDS,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Return empirical HP-filtered moments and the underlying series DataFrame.

    Reads from disk cache if available and not force_refresh.
    Caches to disk after a fresh pull.
    """
    # ---- try cache ----
    cache_is_fresh = False
    if CACHE_PATH.exists() and cache_ttl_seconds is not None:
        cache_age = time.time() - CACHE_PATH.stat().st_mtime
        cache_is_fresh = cache_age <= cache_ttl_seconds
    elif CACHE_PATH.exists():
        cache_is_fresh = True

    if not force_refresh and CACHE_PATH.exists() and cache_is_fresh:
        try:
            cached = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
            series_hp_dict = {col: cached[col].dropna().values for col in cached.columns}
            moments = compute_moments(series_hp_dict, output_key="y", apply_hp=False)
            return moments, cached
        except Exception as e:
            warnings.warn(f"Cache read failed ({e}); re-fetching FRED ...")

    # ---- fresh pull ----
    raw_df   = _fetch_from_fred(start=start, end=end, fred_api_key=fred_api_key)
    raw_dict = _transform(raw_df)

    # HP-filter each series
    series_hp = {k: hp_cycle(v, lamb=HP_LAMBDA) for k, v in raw_dict.items()}

    min_len  = min(len(v) for v in series_hp.values())
    ref_ticker = FRED_SERIES["y"]
    ref_dates  = raw_df[ref_ticker].dropna().index[-min_len:]
    series_df  = pd.DataFrame(
        {k: v[-min_len:] for k, v in series_hp.items()},
        index=ref_dates,
    )

    # Cache to disk
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    series_df.to_csv(CACHE_PATH)

    moments = compute_moments(
        {col: series_df[col].values for col in series_df.columns},
        output_key="y",
        apply_hp=False,
    )
    return moments, series_df


def get_empirical_moments_cached(
    start: str = "1960-01-01",
    fred_api_key: Optional[str] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Thin wrapper intended to be decorated with @st.cache_data(ttl=86400)
    in app.py.
    """
    return load_empirical_moments(start=start, fred_api_key=fred_api_key)


# ──────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison table
# ──────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
    model_moments: Dict[str, float],
    empirical_moments: Dict[str, float],
) -> pd.DataFrame:
    """
    Merge model and empirical moment dicts into a single comparison DataFrame.

    Returns DataFrame with columns: Moment | Model | Empirical | Difference | Ratio
    """
    all_keys = sorted(
        set(model_moments) | set(empirical_moments),
        key=lambda k: (k.split("_")[0], k.split("_")[1] if "_" in k else ""),
    )

    rows = []
    for key in all_keys:
        mod = model_moments.get(key, float("nan"))
        emp = empirical_moments.get(key, float("nan"))
        diff  = mod - emp if not (np.isnan(mod) or np.isnan(emp)) else float("nan")
        ratio = mod / emp if (not np.isnan(emp) and emp != 0) else float("nan")
        rows.append(
            {
                "Moment":     key,
                "Model":      round(mod,  6),
                "Empirical":  round(emp,  6),
                "Difference": round(diff, 6),
                "Ratio":      round(ratio, 4),
            }
        )

    return pd.DataFrame(rows)
