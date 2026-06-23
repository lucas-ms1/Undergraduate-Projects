from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from finrec.providers.utils.optional import require_optional


def build_recession_intervals(usrec_df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build recession intervals from a USREC time series.

    Parameters
    ----------
    usrec_df:
        DataFrame containing a recession indicator over time.
        Supported shapes:
          - columns: ["date", "USREC"] (wide)
          - columns: ["date", "value"] (long-form FRED output)
        Values are treated as recession when == 1 (after numeric coercion).

    Returns
    -------
    list[tuple[pd.Timestamp, pd.Timestamp]]
        A list of (start, end) timestamps for contiguous recession periods.
        End is inclusive in the data but intended to be used as x1 for plotly vrect.
    """
    if usrec_df is None or usrec_df.empty:
        return []

    if "date" not in usrec_df.columns:
        raise ValueError(f"USREC dataframe missing 'date' column. Columns: {list(usrec_df.columns)}")

    if "USREC" in usrec_df.columns:
        y = pd.to_numeric(usrec_df["USREC"], errors="coerce")
    elif "value" in usrec_df.columns:
        y = pd.to_numeric(usrec_df["value"], errors="coerce")
    else:
        raise ValueError(
            "USREC dataframe must have either a 'USREC' column or a 'value' column. "
            f"Columns: {list(usrec_df.columns)}"
        )

    d = pd.to_datetime(usrec_df["date"], errors="coerce")
    s = pd.DataFrame({"date": d, "flag": y}).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if s.empty:
        return []

    s["in_rec"] = (s["flag"] == 1).astype(int)

    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_rec = False
    start: Optional[pd.Timestamp] = None
    last_date: Optional[pd.Timestamp] = None

    for row in s.itertuples(index=False):
        dt = getattr(row, "date")
        flag = int(getattr(row, "in_rec"))
        if flag == 1 and not in_rec:
            in_rec = True
            start = dt
        if flag == 0 and in_rec:
            # recession ended at previous observation date
            end = last_date if last_date is not None else dt
            if start is not None and end is not None and end >= start:
                intervals.append((pd.Timestamp(start), pd.Timestamp(end)))
            in_rec = False
            start = None
        last_date = dt

    if in_rec and start is not None and last_date is not None and last_date >= start:
        intervals.append((pd.Timestamp(start), pd.Timestamp(last_date)))

    return intervals


def plot_timeseries(
    df: pd.DataFrame,
    left_cols: Sequence[str],
    right_cols: Optional[Sequence[str]] = None,
    *,
    title: str = "",
    x_col: str = "date",
    y_left_label: str = "",
    y_right_label: str = "",
    recession_intervals: Optional[Iterable[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    log_y_left: bool = False,
    log_y_right: bool = False,
):
    """
    Build a Plotly time series figure with optional secondary axis and recession shading.

    Parameters
    ----------
    df:
        Input dataframe.
    left_cols:
        Columns to plot on left axis.
    right_cols:
        Optional columns to plot on right axis.
    title:
        Figure title.
    x_col:
        X-axis column name (date-like).
    y_left_label, y_right_label:
        Axis titles.
    recession_intervals:
        Optional iterable of (start, end) timestamps. Each interval is shaded via add_vrect.
    log_y_left, log_y_right:
        Whether to use log scaling on the corresponding y-axis.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or df.empty:
        raise ValueError("df is empty")
    if x_col not in df.columns:
        raise ValueError(f"x_col '{x_col}' not found. Columns: {list(df.columns)}")

    left_cols = [c for c in left_cols if c in df.columns]
    if not left_cols:
        raise ValueError("No left_cols found in dataframe")

    right_cols = list(right_cols or [])
    right_cols = [c for c in right_cols if c in df.columns and c not in left_cols]

    go = require_optional("plotly.graph_objects", extra_hint="dev")
    subplots = require_optional("plotly.subplots", extra_hint="dev")

    plot_df = df.copy()
    x_raw = plot_df[x_col]
    is_datetime_x = False
    if pd.api.types.is_numeric_dtype(x_raw):
        # Keep numeric x as-is (e.g., horizons in IRFs).
        x = x_raw
    else:
        x = pd.to_datetime(x_raw, errors="coerce")
        if x.notna().any():
            is_datetime_x = True
        else:
            # If datetime coercion fails entirely, keep raw x.
            x = x_raw
    plot_df[x_col] = x
    plot_df = plot_df.dropna(subset=[x_col]).sort_values(x_col)

    fig = subplots.make_subplots(specs=[[{"secondary_y": bool(right_cols)}]])

    x_hover = "%{x|%Y-%m-%d}" if is_datetime_x else "%{x}"
    for c in left_cols:
        y = pd.to_numeric(plot_df[c], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=y,
                mode="lines",
                name=str(c),
                hovertemplate=f"{x_hover}<br>{c}: %{{y}}<extra></extra>",
            ),
            secondary_y=False,
        )

    for c in right_cols:
        y = pd.to_numeric(plot_df[c], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=y,
                mode="lines",
                name=str(c),
                hovertemplate=f"{x_hover}<br>{c}: %{{y}}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=title or None,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=50 if title else 20, b=40),
    )

    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_left_label or None, secondary_y=False)
    if right_cols:
        fig.update_yaxes(title_text=y_right_label or None, secondary_y=True)

    if log_y_left:
        fig.update_yaxes(type="log", secondary_y=False)
    if right_cols and log_y_right:
        fig.update_yaxes(type="log", secondary_y=True)

    if recession_intervals and is_datetime_x:
        for (start, end) in recession_intervals:
            if start is None or end is None:
                continue
            fig.add_vrect(
                x0=pd.Timestamp(start),
                x1=pd.Timestamp(end),
                fillcolor="rgba(120,120,120,0.20)",
                line_width=0,
                layer="below",
            )

    return fig

