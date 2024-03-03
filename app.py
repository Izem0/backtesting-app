import json
from datetime import datetime, timezone, time, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import pandas.io.formats.style
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap  # type: ignore
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta

import strategies
from utils import get_binance_ohlcv, get_functions, compute_returns


BASE_DIR = Path(__file__).resolve().parent
MARKET_MAP_PATH = BASE_DIR / "markets_mapping.json"
COMMON_LAYOUT = dict(margin=dict(l=0, r=0, t=25, b=0))
CMAP = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)


@st.cache_data
def load_markets_mapping(file_path: str) -> dict:
    with open(file_path, "r") as f:
        markets_map = json.load(f)
        # reverse key, value for yfinance
        markets_map["yfinance"] = {
            value: key for key, value in markets_map["yfinance"].items()
        }
    return markets_map


@st.cache_data
def load_binance_data(
    market: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    df = get_binance_ohlcv(
        market=market,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    df.set_index("date", inplace=True)
    return df


@st.cache_data
def load_yfinance_data(
    market: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
):
    df = yf.Ticker(market).history(interval=timeframe, start=start_date, end=end_date)
    df.index.rename("date", inplace=True)
    df.columns = df.columns.str.lower()
    df.drop(columns=["dividends", "stock splits"], inplace=True)
    return df


@st.cache_data
def compute_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """Use daily returns (with datetime as index) to compute monthly returns."""
    group = daily_returns.groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).transform(lambda x: (x + 1).cumprod() - 1)
    group = group.groupby([group.index.year, group.index.month]).last()
    group.index.set_names(["year", "month"], inplace=True)
    group = group.to_frame().reset_index()
    group["date"] = [
        datetime(year, int(month), 1)
        for year, month in zip(group["year"], group["month"])
    ]
    group.sort_values(["year", "month"], inplace=True)
    return group


@st.cache_data
def make_monthly_bargraph(monthly_returns: pd.DataFrame) -> go.Figure:
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
        yaxis=dict(title="% monthly return", tickformat=".1%"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **COMMON_LAYOUT,
    )
    return fig


def pivot_monthly(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    melt = monthly_returns.melt(
        id_vars=["year", "month"],
        value_vars=["benchmark_return", "strategy_return"],
        var_name="strategy",
        value_name="return",
    )
    melt["strategy"].replace({"_return": ""}, regex=True, inplace=True)
    pivot = melt.pivot_table(
        index=["year"], columns=["month", "strategy"], values=["return"]
    )
    pivot = pivot.droplevel(level=0, axis=1)
    return pivot


def rename_to_month_names(midx):
    """Rename month of multiindex to month names
    Example: [(1, benchmark), (1, strategy), (2, benchmark) ...] -> [(Jan., benchmark), (Jan., strategy), (Feb., benchmark) ...]
    """
    new_cols = [(datetime(2023, col[0], 1).strftime("%b."), col[1]) for col in midx]
    return pd.MultiIndex.from_tuples(new_cols, names=["month", "strategy"])


def pretty_ohlcv(
    styler: pandas.io.formats.style.Styler,
) -> pandas.io.formats.style.Styler:
    """Style ohlcv dataframe"""
    styler.background_gradient(
        axis=1,
        cmap=CMAP,
        subset=["benchmark_cum_return", "strategy_cum_return"],
        vmin=-1,
        vmax=1,
    )
    # styler.background_gradient(
    #     axis=1,
    #     cmap="Reds",
    #     subset=["signal"],
    #     vmin=0,
    #     vmax=1,
    # )
    format_dict = {col: "{:.2%}" for col in styler.columns if "return" in col}
    format_dict.update(
        {
            "close": "${:.2f}",
            # "signal": "{:.1f}",
        }
    )
    styler.format(format_dict)
    return styler


def pretty_pivot(
    styler: pandas.io.formats.style.Styler,
) -> pandas.io.formats.style.Styler:
    """Style pivot (monthly returns) dataframe"""
    styler.background_gradient(axis=None, cmap=CMAP, vmin=-1, vmax=1)
    styler.format({col: "{:.2%}" for col in styler.columns})
    return styler


def flatten_multiindex(midx, joiner: str = " - "):
    return [joiner.join(col).strip() for col in midx]


def get_max_length(fn: Callable) -> int:
    args = fn.__defaults__
    if not args:
        return 0
    return max(args)


##############
# APP CONFIG #
##############
st.set_page_config(layout="centered")
st.title("Backtesting app")

with open(BASE_DIR / "style.css", "r") as f:
    st.markdown(f"<style>{f.read()}<style/>", unsafe_allow_html=True)


########################
# LOAD MARKETS MAPPING #
########################
markets_map = load_markets_mapping(MARKET_MAP_PATH)


###########
# FILTERS #
###########
col1, col2, col3, col4, col5 = st.columns([2.25, 2.25, 2.25, 2.25, 3])

with col1:
    source = st.selectbox("Source", options=["binance", "yfinance"], index=0)

with col2:
    market = st.selectbox("Market", options=markets_map[source].keys())

with col3:
    strategies_list = get_functions(strategies)
    strategy = st.selectbox(
        "Strategy",
        options=strategies_list,
        index=strategies_list.index("secret_sauce_crypto"),
    )

with col4:
    fees = st.number_input(
        "Fee",
        min_value=0.0,
        max_value=1.0,
        value=0.001,
        step=0.001,
        format="%f",
        help="Fees to apply to trades (0.01=1%)",
    )

with col5:
    end_date = datetime.today().replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    start_date = end_date - relativedelta(years=1)
    d = st.date_input(
        "Period",
        value=(start_date, end_date),
        min_value=datetime(2017, 1, 1, tzinfo=timezone.utc),
        max_value=end_date,
        format="YYYY-MM-DD",
    )
    try:
        start_date, end_date = d
        # convert to datetime with timezone
        start_date = datetime.combine(start_date, time()).replace(tzinfo=timezone.utc)
        end_date = datetime.combine(end_date, time()).replace(tzinfo=timezone.utc)
    except ValueError:
        pass

####################
# STRATEGY RETURNS #
####################
# load market data with offset
# (the calculation of signal requires data before start_date)
max_length = get_max_length(getattr(strategies, strategy))
ohlcv = eval(f"load_{source}_data")(
    markets_map[source][market],
    timeframe="1d",
    start_date=start_date - timedelta(days=max_length),
    end_date=end_date,
)
# get signal from strategy
signal = getattr(strategies, strategy)(ohlcv)
ohlcv = ohlcv.join(signal)
# query only date range needed (-1 day to have returns on day 1 as well)
ohlcv = ohlcv.loc[ohlcv.index >= start_date - timedelta(days=1)]
# compute returns
returns = compute_returns(ohlcv["close"], signal=signal, fees=fees)
ohlcv = ohlcv.join(returns)
# clean a bit df
ohlcv.drop(
    columns=[
        "open",
        "high",
        "low",
        "volume",
    ],
    inplace=True,
)
ohlcv.sort_index(ascending=False, inplace=True)

# display df
st.dataframe(
    ohlcv.style.pipe(pretty_ohlcv),
    height=350,
    use_container_width=True,
    # column_config={"benchmark_return": None, "strategy_return": None},
    column_config={
        "benchmark_return": None,
        "strategy_return": None,
        "signal": None,
        "date": st.column_config.Column(label="Date"),
        "close": st.column_config.Column(label="Close Price"),
        "benchmark_cum_return": st.column_config.Column(
            label="Benchmark Cumulative Return"
        ),
        "strategy_cum_return": st.column_config.Column(
            label="Strategy Cumulative Return"
        ),
    },
)

############################################
# BENCHMARK VS STRATEGY CUMULATIVE RETURNS #
############################################
st.header("Benchmark vs Strategy Cumulative return")

fig = make_subplots(
    rows=1,
    cols=1,
    # row_heights=[0.7, 0.3],
    # shared_xaxes=True,
)
fig.add_trace(
    go.Scatter(
        x=ohlcv.index,
        y=ohlcv["benchmark_cum_return"],
        name=market,
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ohlcv.index,
        y=ohlcv["strategy_cum_return"],
        name="Strategy",
        showlegend=True,
    ),
    row=1,
    col=1,
)
# fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["signal"], name="Signal"), row=2, col=1)
fig.update_layout(
    yaxis=dict(
        title="% cumulative return",
        tickformat=".1%",
    ),
    yaxis2=dict(showgrid=False),
    legend=dict(orientation="h", xanchor="left", x=0, yanchor="bottom", y=1.02),
    **COMMON_LAYOUT,
)
# fig.update_yaxes(type="log")
st.plotly_chart(fig, use_container_width=True)

###################
# MONTHLY RETURNS #
###################
st.header("Monthly returns")

# get monthly returns
monthly_benchmark = compute_monthly_returns(ohlcv["benchmark_return"])
monthly_strategy = compute_monthly_returns(ohlcv["strategy_return"])
monthly = monthly_benchmark.merge(
    monthly_strategy, how="inner", on=["date", "year", "month"]
)

st.subheader("Table")
# compute pivot table
pivot = pivot_monthly(monthly)
pivot.columns = rename_to_month_names(pivot.columns)
# flatten multiindex columns as streamlit currently does not support dataframes with multiple header rows
pivot.columns = flatten_multiindex(pivot.columns, joiner="/")
# display pretty dataframe
st.dataframe(
    pivot.style.pipe(pretty_pivot),
    column_config={
        "year": st.column_config.NumberColumn(format="%d"),
    },
)

st.subheader("Bar graph")
monthly_bargraph = make_monthly_bargraph(monthly)
st.plotly_chart(monthly_bargraph, use_container_width=True)
