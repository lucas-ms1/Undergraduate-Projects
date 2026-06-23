"""Advanced policy and value-function plots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_phase_diagram(model_key: str, result: dict, grid: np.ndarray, shock_labels: list[str], colors: list[str], layout_factory, style_axes):
    """Drift plot using existing policy levels."""
    fig = go.Figure(layout=layout_factory(
        title="Phase Diagram / Drift",
        xaxis_title="Current state",
        yaxis_title="Next state minus current state",
    ))

    if model_key in ("model1", "model2"):
        policies = np.asarray(result["policy_levels"], dtype=float)
        for s in range(policies.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=policies[:, s] - grid,
                    mode="lines",
                    name=shock_labels[s],
                    line=dict(width=1.6, color=colors[s]),
                )
            )
    elif model_key == "model3" and "savings" in result["policy_levels"]:
        policies = np.asarray(result["policy_levels"]["savings"], dtype=float)
        for s in range(policies.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=policies[:, s] - grid,
                    mode="lines",
                    name=shock_labels[s],
                    line=dict(width=1.6, color=colors[s]),
                )
            )
    else:
        labor = np.asarray(result["policy_levels"]["labor"], dtype=float)
        fig.add_trace(
            go.Bar(x=shock_labels, y=labor, marker_color=colors[: len(shock_labels)], name="Labor")
        )
        fig.update_layout(yaxis_title="Labor level")
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    return style_axes(fig)


def make_value_surface(result: dict, grid: np.ndarray, shock_labels: list[str], layout_factory, style_axes):
    """Heatmap of the value function over state and shock."""
    vf = np.asarray(result["value_function"], dtype=float)
    fig = go.Figure(
        data=go.Heatmap(
            x=grid,
            y=shock_labels,
            z=vf.T,
            colorscale="Viridis",
            colorbar=dict(title="Value"),
        ),
        layout=layout_factory(
            title="Value Function Surface",
            xaxis_title="State",
            yaxis_title="Shock state",
        ),
    )
    return style_axes(fig)


def make_policy_surfaces(model_key: str, result: dict, grid: np.ndarray, shock_labels: list[str], layout_factory, style_axes):
    """Heatmaps for policies or a simple extension for labor-only Model 3."""
    if model_key in ("model1", "model2"):
        arrays = [
            ("Consumption policy surface", np.asarray(result["c_policy"], dtype=float)),
            ("Choice policy surface", np.asarray(result["policy_levels"], dtype=float)),
        ]
    elif model_key == "model3" and "savings" in result["policy_levels"]:
        arrays = [
            ("Consumption policy surface", np.asarray(result["c_policy"], dtype=float)),
            ("Savings policy surface", np.asarray(result["policy_levels"]["savings"], dtype=float)),
            ("Labor policy surface", np.asarray(result["policy_levels"]["labor"], dtype=float)),
        ]
    else:
        labor = np.asarray(result["policy_levels"]["labor"], dtype=float)
        consumption = np.asarray(result["c_policy"], dtype=float)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Labor by shock", "Consumption by shock"),
        )
        fig.add_trace(go.Bar(x=shock_labels, y=labor, name="Labor"), row=1, col=1)
        fig.add_trace(go.Bar(x=shock_labels, y=consumption, name="Consumption"), row=1, col=2)
        fig.update_layout(**layout_factory(title="Labor-Only Policy Summary", height=450))
        return [style_axes(fig)]

    figs = []
    for title, arr in arrays:
        fig = go.Figure(
            data=go.Heatmap(
                x=grid,
                y=shock_labels,
                z=arr.T,
                colorscale="Plasma",
                colorbar=dict(title=title),
            ),
            layout=layout_factory(
                title=title,
                xaxis_title="State",
                yaxis_title="Shock state",
            ),
        )
        figs.append(style_axes(fig))
    return figs
