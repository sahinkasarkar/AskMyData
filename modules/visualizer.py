"""
modules/visualizer.py
Adaptive Plotly visualization generator matching the SaaS dashboard theme.
Automatically selects line, area, bar, or donut charts depending on dataset semantics.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Theme palette matching AskMyData SaaS UI
COLOR_PRIMARY = "#2563EB"
COLOR_ACCENT = "#7C3AED"
COLOR_EMERALD = "#10B981"
COLOR_AMBER = "#F59E0B"
COLOR_GRID = "#F1F5F9"
COLOR_TEXT = "#475569"
PALETTE = [
    "#2563EB", "#7C3AED", "#06B6D4", "#10B981",
    "#F59E0B", "#EC4899", "#8B5CF6", "#64748B"
]


def _format_value_axis(fig, val_col):
    """Formats values cleanly with currency or numeric separators."""
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color=COLOR_TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color=COLOR_TEXT,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            color=COLOR_TEXT,
            tickfont=dict(size=10)
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="#FFFFFF"
        )
    )
    return fig


def create_visualization(data, plan=None) -> go.Figure | None:
    """
    Constructs an optimal, polished Plotly chart for the given result data.
    """
    if data is None:
        return None

    # Handle Scalar single values
    if isinstance(data, (int, float, np.number)):
        fig = go.Figure(go.Indicator(
            mode="number",
            value=data,
            number={"prefix": "₹", "valueformat": ",.0f", "font": {"size": 42, "color": "#0F172A"}},
            title={"text": plan.get("value_column", "Metric Total") if plan else "Total", "font": {"size": 14, "color": COLOR_TEXT}}
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        return fig

    if not isinstance(data, pd.DataFrame) or data.empty:
        return None

    df = data.copy()
    if len(df.columns) < 2:
        return None

    x_col = df.columns[0]
    y_col = df.columns[1]

    # Clean numeric format on y-axis
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0)

    # 1. TIME-SERIES / TREND (Line / Area Chart)
    is_time = any(w in str(x_col).lower() for w in ["date", "period", "month", "year", "quarter", "time"])
    if is_time or (plan and plan.get("operation") == "timeseries"):
        fig = px.area(
            df,
            x=x_col,
            y=y_col,
            markers=True,
            color_discrete_sequence=[COLOR_PRIMARY]
        )
        fig.update_traces(
            line=dict(width=2.5, color=COLOR_PRIMARY),
            marker=dict(size=6, color=COLOR_PRIMARY, line=dict(width=1.5, color="#FFFFFF")),
            fillcolor="rgba(37, 99, 235, 0.08)",
            hovertemplate="<b>%{x}</b><br>" + f"{y_col}: ₹%{{y:,.0f}}<extra></extra>"
        )
        return _format_value_axis(fig, y_col)

    # 2. PROPORTION / SHARE (Donut Chart for 5 or fewer categories)
    if len(df) <= 4 and df[y_col].min() >= 0:
        fig = px.pie(
            df,
            names=x_col,
            values=y_col,
            hole=0.6,
            color_discrete_sequence=PALETTE
        )
        fig.update_traces(
            textinfo="percent+label",
            textfont=dict(size=11, family="Plus Jakarta Sans"),
            hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>"
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    # 3. HORIZONTAL BAR (When category names are long or top ranked)
    max_label_len = df[x_col].astype(str).str.len().max()
    if len(df) > 6 or max_label_len > 12:
        df_sorted = df.sort_values(by=y_col, ascending=True)
        fig = px.bar(
            df_sorted,
            x=y_col,
            y=x_col,
            orientation="h",
            color=y_col,
            color_continuous_scale=[[0, "#93C5FD"], [1, COLOR_PRIMARY]]
        )
        fig.update_traces(
            marker=dict(line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>" + f"{y_col}: ₹%{{x:,.0f}}<extra></extra>"
        )
        fig.update_layout(coloraxis_showscale=False)
        return _format_value_axis(fig, y_col)

    # 4. STANDARD VERTICAL BAR
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color_discrete_sequence=[COLOR_PRIMARY]
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), opacity=0.9),
        hovertemplate="<b>%{x}</b><br>" + f"{y_col}: ₹%{{y:,.0f}}<extra></extra>"
    )
    return _format_value_axis(fig, y_col)