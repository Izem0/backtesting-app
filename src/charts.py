import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_cum_returns_graph(data: pd.DataFrame, **layout_kwargs) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    for col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        yaxis={
            "title": "% cumulative return",
            "tickformat": ".1%",
        },
        yaxis2={"showgrid": False},
        legend={
            "orientation": "h",
            "xanchor": "left",
            "x": 0,
            "yanchor": "bottom",
            "y": 1.02,
        },
        **layout_kwargs,
    )
    return fig


def create_monthly_bargraph(data: list[pd.Series], **layout_kwargs) -> go.Figure:
    """Use monthly benchmark and strategy returns to make a bargraph."""
    bars = []
    for d in data:
        bar = go.Bar(
            x=d.index,
            y=d.values,
            name=d.name,
            texttemplate="%{y}",
        )
        bars.append(bar)
    fig = go.Figure(bars)
    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
    fig.update_layout(
        yaxis={"title": "% monthly return", "tickformat": ".1%"},
        barmode="group",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        **layout_kwargs,
    )
    return fig
