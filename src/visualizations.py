"""Visualization helpers for the app."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_histogram(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a histogram for tensile strength with a light engineering theme."""
    fig = px.histogram(
        df,
        x="tensile_strength",
        nbins=20,
        title=title,
        color_discrete_sequence=["#1F4E79"],
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_x=0.5,
        font=dict(size=12),
        xaxis_title="Tensile strength (MPa)",
        yaxis_title="Count",
    )
    return fig


def plot_tensile_vs_parameter(
    df: pd.DataFrame,
    parameter: str,
    title: str
) -> go.Figure:
    """Create a scatter plot of tensile strength vs selected parameter."""

    required_columns = [parameter, "tensile_strength", "material_type"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for plotting: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    plot_df = df[required_columns].copy()

    plot_df[parameter] = pd.to_numeric(
        plot_df[parameter], errors="coerce"
    )

    plot_df["tensile_strength"] = pd.to_numeric(
        plot_df["tensile_strength"], errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=[parameter, "tensile_strength", "material_type"]
    )

    if plot_df.empty:
        raise ValueError(
            "No valid numeric data is available for this plot."
        )

    fig = px.scatter(
        plot_df,
        x=parameter,
        y="tensile_strength",
        color="material_type",
        title=title,
        labels={
            parameter: f"{parameter} ({_unit_for(parameter)})",
            "tensile_strength": "Tensile strength (MPa)",
            "material_type": "Material Type",
        },
    )

    return fig


def _unit_for(parameter: str) -> str:
    """Map parameter names to units for chart labels."""
    mapping = {
        "layer_height": "mm",
        "nozzle_temperature": "degC",
        "print_speed": "mm/s",
        "infill_density": "%",
    }
    return mapping.get(parameter, "")


def plot_metric_importance(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a feature-importance bar chart with aggregated material influence."""
    fig = px.bar(
        df,
        x="importance",
        y="feature_name",
        orientation="h",
        title=title,
        color_discrete_sequence=["#E8710A"],
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_x=0.5,
        font=dict(size=12),
        xaxis_title="Importance",
        yaxis_title="Feature",
    )
    return fig


def plot_actual_vs_predicted(df: pd.DataFrame, title: str) -> go.Figure:
    """Scatter plot actual vs predicted values for the test set."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["actual"],
        y=df["predicted"],
        mode="markers",
        marker=dict(color="#1F4E79", size=8),
        name="Observed vs predicted",
    ))
    fig.add_trace(go.Scatter(
        x=[min(df["actual"].min(), df["predicted"].min()), max(df["actual"].max(), df["predicted"].max())],
        y=[min(df["actual"].min(), df["predicted"].min()), max(df["actual"].max(), df["predicted"].max())],
        mode="lines",
        line=dict(color="#E8710A", dash="solid"),
        name="y = x",
    ))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=title,
        title_x=0.5,
        xaxis_title="Actual tensile strength (MPa)",
        yaxis_title="Predicted tensile strength (MPa)",
    )
    return fig


def plot_residuals(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a residual histogram."""
    residuals = df["actual"] - df["predicted"]
    fig = px.histogram(
        residuals,
        nbins=20,
        title=title,
        color_discrete_sequence=["#1F4E79"],
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_x=0.5,
        xaxis_title="Residual (actual - predicted), MPa",
        yaxis_title="Count",
    )
    return fig


def plot_response_surface(prediction_grid: pd.DataFrame, x_param: str, y_param: str, title: str) -> go.Figure:
    """Plot a 2D heatmap from the real model prediction grid."""
    pivot = prediction_grid.pivot(index=y_param, columns=x_param, values="predicted_strength")
    fig = px.imshow(
        pivot.values,
        labels=dict(x=x_param, y=y_param, color="Predicted strength (MPa)"),
        title=title,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_x=0.5,
        xaxis_title=f"{x_param} ({_unit_for(x_param)})",
        yaxis_title=f"{y_param} ({_unit_for(y_param)})",
    )
    return fig


def plot_optimization_comparison(original: dict[str, float], optimized: dict[str, float], title: str) -> go.Figure:
    """Create a bar chart comparing original vs optimized strength."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Original", "Optimized"], y=[original["strength"], optimized["strength"]], marker_color=["#6B7C8F", "#E8710A"]))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=title,
        title_x=0.5,
        xaxis_title="Condition",
        yaxis_title="Predicted tensile strength (MPa)",
    )
    return fig


def plot_ga_convergence(history: pd.DataFrame, title: str) -> go.Figure:
    """Create a convergence chart for the best and mean GA fitness over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["generation"], y=history["best"], mode="lines", name="Best", line=dict(color="#1F4E79")))
    fig.add_trace(go.Scatter(x=history["generation"], y=history["mean"], mode="lines", name="Mean", line=dict(color="#E8710A", dash="dot")))
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=title,
        title_x=0.5,
        xaxis_title="Generation",
        yaxis_title="Predicted tensile strength (MPa)",
    )
    return fig
