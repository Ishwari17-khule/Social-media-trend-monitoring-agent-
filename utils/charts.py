"""Plotly chart builders for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_dark"
COLORS = {
    "Positive": "#22c55e",
    "Negative": "#ef4444",
    "Neutral": "#94a3b8",
    "Irrelevant": "#38bdf8",
}


def apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the shared dark dashboard styling to a Plotly figure."""
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font={"color": "#e5edf7", "size": 12},
        margin={"l": 25, "r": 20, "t": 45, "b": 25},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(gridcolor="#263244")
    fig.update_yaxes(gridcolor="#263244")
    return fig


def sentiment_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map=COLORS,
        hole=0.35,
    )
    return apply_layout(fig, "Sentiment Distribution")


def sentiment_line(df: pd.DataFrame) -> go.Figure:
    dated = df.dropna(subset=["published_at"]).copy()
    if dated.empty:
        return apply_layout(go.Figure(), "Sentiment Over Time")
    dated["date"] = dated["published_at"].dt.floor("D")
    grouped = dated.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig = px.line(
        grouped,
        x="date",
        y="count",
        color="sentiment",
        color_discrete_map=COLORS,
        markers=True,
    )
    return apply_layout(fig, "Sentiment Over Time")


def top_keywords_bar(keywords_df: pd.DataFrame) -> go.Figure:
    plot_df = keywords_df.sort_values("frequency", ascending=True)
    fig = px.bar(plot_df, x="frequency", y="keyword", orientation="h", text="frequency")
    fig.update_traces(marker_color="#2f8cff", textposition="outside")
    return apply_layout(fig, "Top Keywords")


def trend_chart(df: pd.DataFrame) -> go.Figure:
    dated = df.dropna(subset=["published_at"]).copy()
    if dated.empty:
        return apply_layout(go.Figure(), "Trend Intensity")
    dated["date"] = dated["published_at"].dt.floor("D")
    trend = dated.groupby("date")["trend_score"].mean().reset_index()
    fig = px.area(trend, x="date", y="trend_score", markers=True)
    fig.update_traces(line_color="#a855f7", fillcolor="rgba(168,85,247,0.25)")
    return apply_layout(fig, "Trend Intensity")


def hourly_activity(df: pd.DataFrame) -> go.Figure:
    dated = df.dropna(subset=["published_at"]).copy()
    if dated.empty:
        return apply_layout(go.Figure(), "Hourly Activity")
    dated["hour"] = dated["published_at"].dt.hour
    hourly = dated.groupby("hour").size().reset_index(name="articles")
    fig = px.bar(hourly, x="hour", y="articles")
    fig.update_traces(marker_color="#f59e0b")
    return apply_layout(fig, "Hourly Activity")


def sentiment_distribution_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.bar(counts, x="sentiment", y="count", color="sentiment", color_discrete_map=COLORS)
    return apply_layout(fig, "Sentiment Distribution")
