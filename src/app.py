import datetime as dt
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pendulum
import streamlit as st
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

import strategies
from charts import create_cum_returns_graph, create_monthly_bargraph
from compute import compute_monthly_returns, compute_returns, pivot_monthly
from data import get_binance_ohlcv, get_binance_top_markets, load_binance_data
from style import pretty_ohlcv, pretty_pivot
from utils import flatten_multiindex, get_module_functions, rename_to_month_names

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
COMMON_LAYOUT = {"margin": {"l": 0, "r": 0, "t": 25, "b": 0}}
CMAP = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
COLUMN_CONFIG = {
    "benchmark_return": None,
    "strategy_return": None,
    "signal": st.column_config.Column(label="Signal"),
    "date": st.column_config.Column(label="Date"),
    "open": st.column_config.Column(label="Open Price"),
    "benchmark_cum_return": st.column_config.Column(
        label="Benchmark Cumulative Return"
    ),
    "strategy_cum_return": st.column_config.Column(label="Strategy Cumulative Return"),
}


@st.cache_data
def load_binance_data_cache(
    market: str,
    timeframe: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
) -> pd.DataFrame:
    return load_binance_data(
        market=market,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )


@st.cache_data
def get_binance_top_markets_cache() -> list[str]:
    cmc_api_key = os.getenv("COINMARKETCAP_API_KEY")
    return get_binance_top_markets(cmc_api_key)


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

# set css styles
with open(BASE_DIR / "style.css") as f:
    styles = f.read()
st.markdown(f"<style>{styles}<style/>", unsafe_allow_html=True)


###########
# FILTERS #
###########
col1, col2, col3, col4, col5 = st.columns([2.25, 2.25, 2.25, 2.25, 3])

with col1:
    source = st.selectbox("Source", options=["binance"], index=0)

with col2:
    market = st.selectbox("Market", options=get_binance_top_markets_cache())

with col3:
    strategies_list = get_module_functions(strategies)
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
    end_date = pendulum.now(tz="UTC").start_of("day")
    start_date = end_date.subtract(years=3)
    date_input = st.date_input(
        "Period",
        value=(start_date, end_date),
        min_value=pendulum.datetime(2017, 1, 1, tz="UTC"),
        max_value=end_date,
        format="YYYY-MM-DD",
    )
    try:
        start_date, end_date = date_input
        # convert dates to datetime with timezone
        start_date = pendulum.datetime(
            start_date.year, start_date.month, start_date.day, tz="UTC"
        )
        end_date = pendulum.datetime(
            end_date.year, end_date.month, end_date.day, tz="UTC"
        )
    except ValueError:
        pass

##########################
# TABLE STRATEGY RETURNS #
##########################
# load market data with offset
# (the calculation of signal requires data before start_date)
ohlcv = get_binance_ohlcv(
    market=market,
    timeframe="1d",
    start_date=pendulum.datetime(2017, 8, 17, tz="UTC"),
    end_date=end_date,
)
ohlcv.set_index("date", inplace=True)

# check for missing data
missing_data: bool = start_date < ohlcv.index[0]
if missing_data:
    st.warning(
        "This market's available data does not cover the range specified "
        f"(oldest available date: {ohlcv.index[0]:%Y-%m-%d})"
    )
    ohlcv.drop(index=ohlcv.index[0], inplace=True)

# get signal from strategy
signal = getattr(strategies, strategy)(ohlcv)
ohlcv = ohlcv.join(signal)

# query only date range needed
ohlcv = ohlcv.loc[ohlcv.index >= start_date]

# compute returns
returns = compute_returns(ohlcv["open"], signal=signal, fees=fees)
ohlcv = ohlcv.join(returns)

# clean a bit df
ohlcv.drop(
    columns=[
        "close",
        "high",
        "low",
        "volume",
    ],
    inplace=True,
)
ohlcv.sort_index(ascending=False, inplace=True)

# display returns
st.dataframe(
    ohlcv.style.pipe(pretty_ohlcv, cmap=CMAP),
    height=350,
    use_container_width=True,
    column_config=COLUMN_CONFIG,
)

############################
# CHART CUMULATIVE RETURNS #
############################
st.header("Benchmark vs Strategy Cumulative return")

cum_returns_graph = create_cum_returns_graph(ohlcv, market=market, **COMMON_LAYOUT)
st.plotly_chart(cum_returns_graph, use_container_width=True)

#########################
# CHART MONTHLY RETURNS #
#########################
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
# flatten multiindex columns as streamlit currently does not support
# dataframes with multiple header rows
pivot.columns = flatten_multiindex(pivot.columns, joiner="/")
# display pretty dataframe
st.dataframe(
    pivot.style.pipe(pretty_pivot, cmap=CMAP),
    column_config={
        "year": st.column_config.NumberColumn(format="%d"),
    },
)

st.subheader("Bar graph")
monthly_bargraph = create_monthly_bargraph(monthly, market=market, **COMMON_LAYOUT)
st.plotly_chart(monthly_bargraph, use_container_width=True)
