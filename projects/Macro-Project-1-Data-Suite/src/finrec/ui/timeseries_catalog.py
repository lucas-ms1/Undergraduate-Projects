"""
Curated time series catalogs for Finance (tickers) and Econ (FRED ids).
Supports ordering by recents + category priority + alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SeriesOption:
    id: str
    label: str
    category: str
    priority: int


# Category priorities: lower = higher in list
_FINANCE_CATEGORY_PRIORITY = {
    "Indices": 0,
    "Magnificent7": 1,
    "TopSP500": 2,
    "Commodities": 3,
    "Rates": 4,
    "Crypto": 5,
    "Other": 99,
}

_ECON_CATEGORY_PRIORITY = {
    "Inflation": 0,
    "Labor": 1,
    "Rates": 2,
    "Growth": 3,
    "Recession": 4,
    "Other": 99,
}

_FINANCE_CATALOG: List[SeriesOption] = [
    # Indices
    SeriesOption("SPY", "S&P 500 (SPY)", "Indices", 0),
    SeriesOption("QQQ", "Nasdaq-100 (QQQ)", "Indices", 1),
    SeriesOption("DIA", "Dow Jones (DIA)", "Indices", 2),
    SeriesOption("IWM", "Russell 2000 (IWM)", "Indices", 3),
    SeriesOption("VTI", "Total US (VTI)", "Indices", 4),
    SeriesOption("VOO", "S&P 500 (VOO)", "Indices", 5),
    # Magnificent 7
    SeriesOption("AAPL", "Apple (AAPL)", "Magnificent7", 10),
    SeriesOption("MSFT", "Microsoft (MSFT)", "Magnificent7", 11),
    SeriesOption("GOOGL", "Alphabet (GOOGL)", "Magnificent7", 12),
    SeriesOption("AMZN", "Amazon (AMZN)", "Magnificent7", 13),
    SeriesOption("NVDA", "Nvidia (NVDA)", "Magnificent7", 14),
    SeriesOption("META", "Meta (META)", "Magnificent7", 15),
    SeriesOption("TSLA", "Tesla (TSLA)", "Magnificent7", 16),
    # Top S&P 500 by market cap
    SeriesOption("JPM", "JPMorgan Chase (JPM)", "TopSP500", 20),
    SeriesOption("JNJ", "Johnson & Johnson (JNJ)", "TopSP500", 21),
    SeriesOption("V", "Visa (V)", "TopSP500", 22),
    SeriesOption("UNH", "UnitedHealth (UNH)", "TopSP500", 23),
    SeriesOption("WMT", "Walmart (WMT)", "TopSP500", 24),
    SeriesOption("PG", "Procter & Gamble (PG)", "TopSP500", 25),
    SeriesOption("HD", "Home Depot (HD)", "TopSP500", 26),
    SeriesOption("MA", "Mastercard (MA)", "TopSP500", 27),
    SeriesOption("BAC", "Bank of America (BAC)", "TopSP500", 28),
    SeriesOption("XOM", "Exxon Mobil (XOM)", "TopSP500", 29),
    SeriesOption("CVX", "Chevron (CVX)", "TopSP500", 30),
    SeriesOption("ABBV", "AbbVie (ABBV)", "TopSP500", 31),
    SeriesOption("ORCL", "Oracle (ORCL)", "TopSP500", 32),
    SeriesOption("COST", "Costco (COST)", "TopSP500", 33),
    SeriesOption("PEP", "PepsiCo (PEP)", "TopSP500", 34),
    SeriesOption("KO", "Coca-Cola (KO)", "TopSP500", 35),
    SeriesOption("AVGO", "Broadcom (AVGO)", "TopSP500", 36),
    SeriesOption("ADBE", "Adobe (ADBE)", "TopSP500", 37),
    SeriesOption("LLY", "Eli Lilly (LLY)", "TopSP500", 38),
    SeriesOption("NFLX", "Netflix (NFLX)", "TopSP500", 39),
    # Commodities
    SeriesOption("GLD", "Gold (GLD)", "Commodities", 40),
    SeriesOption("SLV", "Silver (SLV)", "Commodities", 41),
    SeriesOption("USO", "Oil (USO)", "Commodities", 42),
    SeriesOption("UNG", "Natural Gas (UNG)", "Commodities", 43),
    SeriesOption("DBA", "Ag basket (DBA)", "Commodities", 44),
    # Rates / fixed income
    SeriesOption("TLT", "20+ Treas (TLT)", "Rates", 50),
    SeriesOption("IEF", "7-10 Treas (IEF)", "Rates", 51),
    SeriesOption("SHY", "1-3 Treas (SHY)", "Rates", 52),
    SeriesOption("HYG", "High Yield (HYG)", "Rates", 53),
    # Crypto
    SeriesOption("BTC-USD", "Bitcoin (BTC-USD)", "Crypto", 60),
    SeriesOption("ETH-USD", "Ethereum (ETH-USD)", "Crypto", 61),
]

_ECON_CATALOG: List[SeriesOption] = [
    # Inflation
    SeriesOption("CPIAUCSL", "CPI All Urban (CPIAUCSL)", "Inflation", 0),
    SeriesOption("CPILFESL", "Core CPI (CPILFESL)", "Inflation", 1),
    SeriesOption("PCEPI", "PCE Price Index (PCEPI)", "Inflation", 2),
    SeriesOption("T5YIE", "5Y Breakeven Inflation (T5YIE)", "Inflation", 3),
    # Labor
    SeriesOption("UNRATE", "Unemployment Rate (UNRATE)", "Labor", 10),
    SeriesOption("PAYEMS", "Nonfarm Payrolls (PAYEMS)", "Labor", 11),
    SeriesOption("ICSA", "Initial Claims (ICSA)", "Labor", 12),
    SeriesOption("CES0500000003", "Avg Hourly Earnings (CES0500000003)", "Labor", 13),
    # Rates
    SeriesOption("FEDFUNDS", "Fed Funds Rate (FEDFUNDS)", "Rates", 20),
    SeriesOption("DGS10", "10Y Treasury (DGS10)", "Rates", 21),
    SeriesOption("DGS2", "2Y Treasury (DGS2)", "Rates", 22),
    SeriesOption("T10Y2Y", "10Y-2Y Spread (T10Y2Y)", "Rates", 23),
    # Growth
    SeriesOption("GDP", "GDP (GDP)", "Growth", 30),
    SeriesOption("GDPC1", "Real GDP (GDPC1)", "Growth", 31),
    SeriesOption("INDPRO", "Industrial Production (INDPRO)", "Growth", 32),
    SeriesOption("RSXFS", "Retail Sales (RSXFS)", "Growth", 33),
    # Recession
    SeriesOption("USREC", "Recession Indicator (USREC)", "Recession", 40),
]

_FINANCE_BY_ID = {o.id: o for o in _FINANCE_CATALOG}
_ECON_BY_ID = {o.id: o for o in _ECON_CATALOG}


def _build_options(
    catalog: List[SeriesOption],
    by_id: dict,
    category_priority: dict,
    recent_ids: List[str],
    custom_ids: List[str],
) -> List[SeriesOption]:
    seen: set[str] = set()
    ordered: List[SeriesOption] = []

    # 1. Recents first (preserve recency order)
    for rid in recent_ids:
        rid = rid.strip().upper()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        if rid in by_id:
            ordered.append(by_id[rid])
        else:
            cat = "Other"
            ordered.append(SeriesOption(rid, rid, cat, category_priority.get(cat, 99)))

    # 2. Custom IDs not yet in catalog
    for cid in custom_ids:
        cid = cid.strip().upper()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if cid in by_id:
            ordered.append(by_id[cid])
        else:
            ordered.append(SeriesOption(cid, cid, "Other", 99))

    # 3. Remaining catalog items by priority then alpha
    remaining = [o for o in catalog if o.id not in seen]
    remaining.sort(key=lambda o: (category_priority.get(o.category, 99), o.priority, o.id))
    ordered.extend(remaining)

    return ordered


def build_finance_options(
    recent_ids: List[str],
    custom_ids: List[str],
) -> List[SeriesOption]:
    return _build_options(
        _FINANCE_CATALOG,
        _FINANCE_BY_ID,
        _FINANCE_CATEGORY_PRIORITY,
        recent_ids,
        custom_ids,
    )


def build_econ_options(
    recent_ids: List[str],
    custom_ids: List[str],
) -> List[SeriesOption]:
    return _build_options(
        _ECON_CATALOG,
        _ECON_BY_ID,
        _ECON_CATEGORY_PRIORITY,
        recent_ids,
        custom_ids,
    )
