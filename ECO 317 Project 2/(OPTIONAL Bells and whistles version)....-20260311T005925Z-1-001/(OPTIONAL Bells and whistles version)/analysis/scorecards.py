"""Model-vs-data scorecard helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.moments import compute_moments


def flatten_moments(moments: dict) -> dict[str, float]:
    """Flatten nested moment dictionaries into dot-separated keys."""
    flat: dict[str, float] = {}
    for variable, stats in moments.items():
        for stat_name, value in stats.items():
            flat[f"{variable}.{stat_name}"] = float(value)
    return flat


def get_default_targets(model_key: str) -> pd.DataFrame:
    """Illustrative course-style targets, kept intentionally lightweight."""
    defaults = {
        "model1": [
            ("c.variance", 0.12, 1.0),
            ("c.autocorrelation", 0.80, 1.0),
            ("c.corr_with_income", 0.55, 1.0),
            ("a.mean", 5.00, 0.5),
        ],
        "model2": [
            ("k.autocorrelation", 0.95, 1.0),
            ("c.corr_with_income", 0.75, 1.0),
            ("investment.variance", 0.20, 1.0),
            ("y.variance", 0.15, 0.75),
        ],
        "model3": [
            ("labor.mean", 0.35, 1.0),
            ("labor.autocorrelation", 0.70, 1.0),
            ("earnings.variance", 0.08, 1.0),
            ("c.corr_with_income", 0.40, 0.75),
        ],
    }
    rows = defaults.get(model_key, [])
    return pd.DataFrame(rows, columns=["moment", "target", "weight"])


def get_fred_series_presets(model_key: str) -> pd.DataFrame:
    """Recommended FRED series by model for empirical scorecards/calibration."""
    presets = {
        "model1": [
            ("c", "Real Personal Consumption Expenditures", "PCEC96", "pct_change"),
            ("y", "Real Disposable Personal Income", "DSPIC96", "pct_change"),
        ],
        "model2": [
            ("y", "Real GDP", "GDPC1", "pct_change"),
            ("c", "Real Personal Consumption Expenditures", "PCEC96", "pct_change"),
            ("investment", "Real Gross Private Domestic Investment", "GPDIC1", "pct_change"),
        ],
        "model3": [
            ("earnings", "Average Hourly Earnings: Total Private", "CES0500000003", "pct_change"),
            ("labor", "Average Weekly Hours: Total Private", "AWHAETP", "level"),
            ("c", "Real Personal Consumption Expenditures", "PCEC96", "pct_change"),
        ],
    }
    rows = presets.get(model_key, [])
    return pd.DataFrame(rows, columns=["variable", "label", "series_id", "transform"])


def get_default_empirical_moment_list(model_key: str) -> list[str]:
    """Reasonable moment targets to compute from downloaded FRED preset data."""
    defaults = {
        "model1": [
            "c.variance",
            "c.autocorrelation",
            "c.corr_with_income",
            "y.variance",
            "y.autocorrelation",
        ],
        "model2": [
            "y.variance",
            "y.autocorrelation",
            "c.corr_with_income",
            "investment.variance",
            "investment.autocorrelation",
        ],
        "model3": [
            "earnings.variance",
            "earnings.autocorrelation",
            "labor.variance",
            "labor.autocorrelation",
            "c.corr_with_income",
        ],
    }
    return defaults.get(model_key, [])


def targets_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize target-moment input."""
    required = {"moment", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    if "weight" not in out.columns:
        out["weight"] = 1.0
    out = out[["moment", "target", "weight"]]
    out["target"] = out["target"].astype(float)
    out["weight"] = out["weight"].astype(float)
    return out


def _transform_series(values: pd.Series, transform: str) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    if transform == "pct_change":
        arr = arr.pct_change()
    elif transform == "log_diff":
        arr = np.log(arr.replace(0, np.nan)).diff()
    elif transform == "diff":
        arr = arr.diff()
    return arr


def build_targets_from_empirical_bundle(
    downloaded_bundle: dict[str, pd.DataFrame],
    preset_df: pd.DataFrame,
    income_key: str,
    selected_moments: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create target moments from downloaded FRED series mapped to model variables."""
    aligned = None
    used_rows = []
    for row in preset_df.to_dict(orient="records"):
        series_id = row["series_id"]
        variable = row["variable"]
        transform = row.get("transform", "level")
        frame = downloaded_bundle.get(series_id)
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local["date"] = pd.to_datetime(local["date"], errors="coerce")
        local["value"] = _transform_series(local["value"], transform)
        local = local[["date", "value"]].dropna().rename(columns={"value": variable})
        used_rows.append(
            {
                "variable": variable,
                "series_id": series_id,
                "transform": transform,
            }
        )
        if aligned is None:
            aligned = local
        else:
            aligned = aligned.merge(local, on="date", how="inner")

    if aligned is None or aligned.empty:
        raise ValueError("No downloaded FRED preset data were available to build target moments.")

    series = {
        col: aligned[col].to_numpy(dtype=float)
        for col in aligned.columns
        if col != "date"
    }
    empirical_moments = compute_moments(series, income_key=income_key)
    flat = flatten_moments(empirical_moments)
    desired = selected_moments or sorted(flat)
    rows = []
    for moment in desired:
        if moment in flat:
            rows.append(
                {
                    "moment": moment,
                    "target": float(flat[moment]),
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows), aligned, empirical_moments


def build_moment_scorecard(
    sim_moments: dict,
    target_moments: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Compare simulated moments to targets and return a score table."""
    flat = flatten_moments(sim_moments)
    rows = []
    for row in target_moments.to_dict(orient="records"):
        moment = row["moment"]
        target = float(row["target"])
        weight = float(row.get("weight", 1.0))
        model_value = float(flat.get(moment, np.nan))
        gap = model_value - target
        pct_gap = np.nan if abs(target) < 1e-12 else gap / target
        weighted_sq_error = np.nan if np.isnan(model_value) else weight * (gap ** 2)
        rows.append(
            {
                "moment": moment,
                "model": model_value,
                "target": target,
                "gap": gap,
                "pct_gap": pct_gap,
                "weight": weight,
                "weighted_sq_error": weighted_sq_error,
            }
        )
    scorecard = pd.DataFrame(rows)
    objective = float(np.nansum(scorecard["weighted_sq_error"].to_numpy(dtype=float)))
    return scorecard, objective
