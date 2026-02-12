from __future__ import annotations

from finrec.providers.macro.fred_stub import FREDStubProvider
from finrec.providers.market.yfinance_stub import YFinanceStubProvider
from finrec.providers.news.gdelt_stub import GDELTStubProvider


class _Ctx:
    def log(self, level: str, message: str) -> None:
        pass


def test_yfinance_stub_honors_date_range():
    p = YFinanceStubProvider()
    df = p.fetch({"symbol": "AAPL", "start_date": "2020-01-01", "end_date": "2020-01-10", "n": 5}, ctx=_Ctx())
    assert df["date"].min() >= "2020-01-01"
    assert df["date"].max() <= "2020-01-10"


def test_fred_stub_honors_date_range():
    p = FREDStubProvider()
    df = p.fetch({"series_id": "CPIAUCSL", "start_date": "2020-01-01", "end_date": "2020-06-30", "n": 5}, ctx=_Ctx())
    assert df["date"].min() >= "2020-01-01"
    assert df["date"].max() <= "2020-06-30"


def test_gdelt_stub_runs_with_date_range():
    p = GDELTStubProvider()
    df = p.fetch({"query": "inflation", "start_date": "2020-01-01", "end_date": "2020-01-10", "n": 5}, ctx=_Ctx())
    assert "ts" in df.columns
    assert len(df) == 5

