import json
from datetime import datetime
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
# - implement strategy pattern for sources (with methods like load_data, get_tickers ...)
# - add more tickers for yfinance
# - add custom strategy field
# - add stats below dataframe in app


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
    timeframe = st.selectbox("Timeframe", options=["1h", "4h", "1d", "1w"], index=2)

with col5:
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - relativedelta(years=1)
    d = st.date_input(
        "Period",
        value=(start_date, end_date),
        min_value=datetime(2017, 1, 1),
        max_value=end_date,
        format="YYYY-MM-DD",
    )
    try:
        start_date, end_date = d
    except ValueError:
        pass


####################
# STRATEGY RETURNS #
####################
# load market data
ohlcv = eval(f"load_{source}_data")(
    markets_map[source][market],
    timeframe=timeframe,
    start_date=start_date - pd.Timedelta(days=1),
    end_date=end_date,
)
# get signal from strategy
signal = getattr(strategies, strategy)(ohlcv)
ohlcv = ohlcv.join(signal)
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
st.subheader("Benchmark vs Strategy Cumulative return")

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
st.subheader("Monthly returns")

ohlcv2 = ohlcv.copy()
ohlcv2.loc[ohlcv.index.day == 1, "benchmark_return"] = 0
group = ohlcv2.groupby([ohlcv2.index.year, ohlcv2.index.month])[
    ["benchmark_return", "strategy_return"]
].transform(lambda x: (x + 1).cumprod() - 1)
group2 = group.groupby([group.index.year, group.index.month])[
    ["benchmark_return", "strategy_return"]
].last()
group2.index.set_names(["year", "month"], inplace=True)
group2.reset_index(inplace=True)
group2["date"] = [
    datetime(year, month, 1) for year, month in zip(group2["year"], group2["month"])
]
fig = go.Figure(
    [
        go.Bar(
            x=group2["date"],
            y=group2["benchmark_return"],
            name=market,
            texttemplate="%{y}",
        ),
        go.Bar(
            x=group2["date"],
            y=group2["strategy_return"],
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

st.plotly_chart(fig, use_container_width=True)
