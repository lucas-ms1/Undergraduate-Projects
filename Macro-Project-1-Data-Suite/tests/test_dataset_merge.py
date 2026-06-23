from __future__ import annotations

from pathlib import Path

import pandas as pd

from finrec.datasets.merge import build_merged_dataset


def test_build_merged_dataset_outer(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"

    pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "value": [1.0, 2.0]}).to_csv(a, index=False)
    pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "close": [10.0, 11.0]}).to_csv(b, index=False)

    out = build_merged_dataset(
        inputs=[
            {"path": str(a), "date_col": "date", "value_col": "value", "alias": "macro"},
            {"path": str(b), "date_col": "date", "value_col": "close", "alias": "mkt"},
        ],
        merge_how="outer",
        ffill=False,
    )

    assert list(out.columns) == ["date", "macro", "mkt"]
    assert out["date"].tolist() == ["2020-01-01", "2020-01-02", "2020-01-03"]


def test_build_merged_dataset_ffill(tmp_path: Path):
    a = tmp_path / "a.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-01-03"], "value": [1.0, 3.0]}).to_csv(a, index=False)

    out = build_merged_dataset(
        inputs=[{"path": str(a), "date_col": "date", "value_col": "value", "alias": "x"}],
        merge_how="outer",
        ffill=True,
    )
    # single series should remain same dates; ffill doesn't add new rows
    assert out["date"].tolist() == ["2020-01-01", "2020-01-03"]

