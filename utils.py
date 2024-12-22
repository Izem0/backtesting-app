import os
from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import requests


def _chunks_dates(dates, n=1000):
    """Return 2 lists of start dates & end dates given a series of dates."""
    start_dates, end_dates = [], []
    for i in range(0, len(dates), n):
        chunk = dates[i : i + n]
        start_dates.append(chunk[0])
        end_dates.append(chunk[-1])

    return start_dates, end_dates


def get_binance_ohlcv(
    market: str,
    timeframe: str = "1d",
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
) -> pd.DataFrame:
    tf_mapping = {
        "1w": "1W",
        "1d": "1D",
        "4h": "4H",
        "1h": "1H",
        "15m": "15T",
        "5m": "5T",
    }

    dates = pd.date_range(start=start_date, end=end_date, freq=tf_mapping[timeframe])
    start_dates, end_dates = _chunks_dates(dates, n=1000)

    df = pd.DataFrame()
    for start_date, end_date in zip(start_dates, end_dates):
        params = {
            "symbol": market,
            "interval": timeframe,
            "startTime": int(start_date.timestamp() * 1e3),
            "endTime": int(end_date.timestamp() * 1e3),
            "limit": 1000,
        }
        r = requests.get("https://api.binance.com/api/v3/klines", params=params)
        dfx = pd.DataFrame(r.json())
        df = pd.concat([df, dfx], ignore_index=True)

    df = df[[0, 1, 2, 3, 4, 5]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"], utc=True, unit="ms")
    df = df.astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    return df


def get_functions(module) -> list[str]:
    """Returns list of all functions present in a module"""
    functions = getmembers(module, isfunction)
    return [x[0] for x in functions]


def compute_returns(price, signal, fees=0.001) -> pd.DataFrame:
    """`price` should be the open price"""
    # make dataframe
    df = pd.DataFrame({"price": price, "signal": signal}, index=price.index)
    # Calculate the benchmark daily returns
    df["benchmark_return"] = df["price"].pct_change()
    # Calculate the strategy returns with fees
    df["strategy_return"] = df["signal"].shift() * df["benchmark_return"]
    df["strategy_return"] = df["strategy_return"] - fees * df["signal"].diff().abs()

    # calculate the benchmark cumulative returns
    df["benchmark_cum_return"] = (1 + df["benchmark_return"]).cumprod() - 1
    # Calculate the cumulative returns
    df["strategy_cum_return"] = (1 + df["strategy_return"]).cumprod() - 1
    return df[
        [
            "benchmark_return",
            "benchmark_cum_return",
            "strategy_return",
            "strategy_cum_return",
        ]
    ]


def get_binance_markets(exclude: list[str] = None) -> list[str]:
    """Return binance's USDT & active markets. Optionally provide a list of markets to exclude."""
    if exclude is None:
        exclude = []
    r = requests.get("https://api.binance.com/api/v3/exchangeInfo")
    return [
        x["symbol"]
        for x in r.json()["symbols"]
        if (x["status"] != "BREAK")
        and (x["quoteAsset"] == "USDT")
        and x["symbol"] not in exclude
    ]


def get_cmc_listings(limit=1000, exclude_stables: bool = True):
    """Returns a paginated list of all active cryptocurrencies with latest market data.
    https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyListingsLatest"""

    r = requests.get(
        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
        headers={"X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY")},
        params={"limit": limit},
    )
    cmc = pd.DataFrame(r.json()["data"])
    cmc["stablecoin"] = cmc["tags"].apply(lambda x: "stablecoin" in x)

    if exclude_stables:
        return cmc.loc[cmc["stablecoin"] == False, :]
    return cmc


def get_binance_top_markets(top: int = 500) -> list[str]:
    # get binance markets
    markets = pd.DataFrame(dict(symbol=get_binance_markets()))

    # get cmc listing
    cmc = get_cmc_listings()
    cmc["symbol"] = cmc["symbol"] + "USDT"

    # keep only top x markets
    top_x = (
        cmc[["symbol"]]
        .merge(markets, how="inner", on="symbol")
        .loc[:top, "symbol"]
        .to_list()
    )
    return top_x
