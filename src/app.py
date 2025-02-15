import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pendulum
import streamlit as st
import vectorbt as vbt
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

import strategies
from charts import create_cum_returns_graph, create_monthly_bargraph
from compute import compute_monthly_returns
from data import get_binance_top_markets
from logger import setup_logger
from style import pretty_ohlcv
from utils import get_module_functions

load_dotenv()

LOG = setup_logger("app")
BASE_DIR = Path(__file__).resolve().parent
COMMON_LAYOUT = {"margin": {"l": 0, "r": 0, "t": 25, "b": 0}}
CMAP = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
RETURNS_COLUMN_CONFIG = {
    "open": st.column_config.NumberColumn(label="Open Price", format="$%d"),
}
METRICS_TO_EXCLUDE = [
    "expectancy",
    "profit_factor",
    "calmar_ratio",
    "omega_ratio",
    "sortino_ratio",
    "max_gross_exposure",
]


@st.cache_data
def load_data_cache(
    symbols: list[str],
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    data = vbt.BinanceData.download(
        symbols=symbols,
        interval=interval,
        start=start,
        end=end,
    )
    data.data[market].columns = data.data[market].columns.str.lower()
    return data.data[symbols[0]]


@st.cache_data
def get_binance_top_markets_cache() -> list[str]:
    cmc_api_key = os.getenv("COINMARKETCAP_API_KEY")
    return get_binance_top_markets(cmc_api_key)


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
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    source = st.selectbox("Source", options=["binance"], index=0)

with col2:
    market = st.selectbox("Market", options=get_binance_top_markets_cache())

with col3:
    fees = st.number_input(
        "Fee",
        min_value=0.0,
        max_value=1.0,
        value=0.001,
        step=0.001,
        format="%f",
        help="Fees to apply to trades (0.01=1%)",
    )

with col4:
    end_date = pendulum.now(tz="UTC").start_of("day")
    start_date = end_date.subtract(years=3)
    date_input = st.date_input(
        "Period",
        value=(start_date, end_date),
        min_value=None,
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

# select strategies
strategies_list = get_module_functions(strategies)
strategies_choice = st.multiselect(
    "Strategies",
    options=strategies_list,
    default=["buy_and_hold", "secret_sauce_crypto"],
)

if not strategies_choice:
    st.stop()

##########################
# TABLE STRATEGY RETURNS #
##########################
# load market data with offset
# (the calculation of signal requires data before start_date)
ohlcv_full = load_data_cache(
    symbols=[market],
    interval="1d",
    start=start_date.subtract(days=90),
    end=end_date.add(days=1),
)
ohlcv = ohlcv_full.loc[ohlcv_full.index >= start_date]

# check for missing data
missing_data: bool = start_date < ohlcv_full.index[0]
if missing_data:
    st.warning(
        "This market's available data does not cover the range specified "
        f"(oldest available date: {ohlcv.index[0]:%Y-%m-%d})"
    )
    ohlcv = ohlcv.drop(index=ohlcv.index[0])

# get signals & size for strategies
size_df = pd.DataFrame()
min_size_df = pd.DataFrame(index=ohlcv.index)
for strat in strategies_choice:
    signal = getattr(strategies, strat)(ohlcv_full)
    # query only date range needed
    size_df[strat] = signal.loc[signal.index >= ohlcv.index[0]]

    if strat == "secret_sauce_crypto":
        min_size_df[strat] = 100 / ohlcv["open"].values
        continue

    # set min size as 0 for other startegies
    min_size_df[strat] = [0] * ohlcv.shape[0]


# instantiate portfolio
pf = vbt.Portfolio.from_orders(
    ohlcv["open"],
    size=size_df,
    fees=fees,
    size_type="targetpercent",
    freq="1D",
    min_size=min_size_df,
    init_cash=1000,
)

# display stats
metrics = [m for m in vbt.Portfolio.metrics.keys() if m not in METRICS_TO_EXCLUDE]
stats = pf.stats(metrics=metrics, agg_func=None).round(4).T
stats = stats.astype(str)
st.dataframe(
    stats,
    height=350,
    use_container_width=True,
)

# display cumulative returns
returns = pd.DataFrame(data=ohlcv["open"], index=ohlcv.index)
format_dict = {"open": "${:.2f}"}
for strat in strategies_choice:
    signal_col_name = f"{strat}_signal"
    returns = returns.join(size_df[strat].to_frame(signal_col_name))
    returns = returns.join(pf.cumulative_returns()[strat])
    # df formatting
    format_dict.update({signal_col_name: "{:.1f}", strat: "{:.2%}"})

st.dataframe(
    returns[::-1].style.pipe(
        pretty_ohlcv,
        cmap=CMAP,
        format_dict=format_dict,
        exclude_regex=r".*signal|open.*",
    ),
    height=350,
    use_container_width=True,
)

# ############################
# # CHART CUMULATIVE RETURNS #
# ############################
st.header("Benchmark vs Strategy Cumulative return")

cum_returns_graph = create_cum_returns_graph(stats[strategies_choice], **COMMON_LAYOUT)
st.plotly_chart(cum_returns_graph, use_container_width=True)


# #########################
# # CHART MONTHLY RETURNS #
# #########################
st.header("Monthly returns")
# get monthly returns
monthly_returns = []
for strat in strategies_choice:
    returns = pf.asset_returns()[strat]
    returns = returns.replace({-np.inf: 0})
    monthly = compute_monthly_returns(returns)
    monthly_returns.append(monthly)

monthly_bargraph = create_monthly_bargraph(monthly_returns, **COMMON_LAYOUT)
st.plotly_chart(monthly_bargraph, use_container_width=True)
