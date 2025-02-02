import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_cum_returns_graph(
    cum_returns: pd.DataFrame, market: str, **layout_kwargs
) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns["benchmark_cum_return"],
            name=market,
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns["strategy_cum_return"],
            name="Strategy",
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


def create_monthly_bargraph(
    monthly_returns: pd.DataFrame, market: str, **layout_kwargs
) -> go.Figure:
    """Use monthly benchmark and strategy returns to make a bargraph."""
    fig = go.Figure(
        [
            go.Bar(
                x=monthly_returns["date"],
                y=monthly_returns["benchmark_return"],
                name=market,
                texttemplate="%{y}",
            ),
            go.Bar(
                x=monthly_returns["date"],
                y=monthly_returns["strategy_return"],
                name="Strategy",
                texttemplate="%{y}",
            ),
        ]
    )
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
