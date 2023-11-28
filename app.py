import json
from datetime import datetime, timezone, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta

import strategies
from utils import get_binance_ohlcv, get_functions, compute_returns


# TODO:
# - add strategies description
# - add custom strategy field
# - add report (difference in returns vs benchmark) below in app
# - implement strategy pattern for sources (with methods like load_data, get_tickers ...)
# - add more tickers for yfinance


BASE_DIR = Path(__file__).resolve().parent
MARKET_MAP_PATH = BASE_DIR / "markets_mapping.json"
COMMON_LAYOUT = dict(margin=dict(l=0, r=0, t=25, b=0))


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
col1, col2, col3, col4 = st.columns([3, 3, 3, 3])

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
ohlcv = eval(f"load_{source}_data")(
    markets_map[source][market],
    timeframe="1d",
    start_date=start_date - timedelta(days=365),
    end_date=end_date,
)
# get signal from strategy
signal = getattr(strategies, strategy)(ohlcv)
ohlcv = ohlcv.join(signal)
# query only date range needed (-1 day to have returns on day 1 as well)
ohlcv = ohlcv.loc[ohlcv.index >= start_date - timedelta(days=1)]
# compute returns
returns = compute_returns(ohlcv["close"], signal=signal, fees=0.001)
ohlcv = ohlcv.join(returns)
# clean a bit df
ohlcv.drop(columns=["open", "high", "low", "volume"], inplace=True)
ohlcv.sort_index(ascending=False, inplace=True)

# display df
st.dataframe(
    ohlcv,
    height=350,
    use_container_width=True,
    column_config={"strategy_return": None},
)

############################################
# BENCHMARK VS STRATEGY CUMULATIVE RETURNS #
############################################
st.header("Benchmark vs Strategy Cumulative return")

fig = make_subplots(
    rows=2,
    cols=1,
    row_heights=[0.7, 0.3],
    shared_xaxes=True,
)
fig.add_trace(
    go.Scatter(
        x=ohlcv.index,
        y=ohlcv["benchmark_cum_return"],
        name=market,
        showlegend=True,
    )
)
fig.add_trace(
    go.Scatter(
        x=ohlcv.index,
        y=ohlcv["strategy_cum_return"],
        name="Strategy",
        showlegend=True,
    )
)
fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["signal"], name="Signal"), row=2, col=1)
fig.update_layout(
    yaxis=dict(
        title="% cumulative return",
        tickformat=".1%",
    ),
    yaxis2=dict(showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
monthly = monthly_benchmark.merge(monthly_strategy, how="inner", on=["date", "year", "month"])

monthly_bargraph = make_monthly_bargraph(monthly)
st.plotly_chart(monthly_bargraph, use_container_width=True)
