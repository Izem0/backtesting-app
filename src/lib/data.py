import datetime as dt

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


def _format_binance_ohlcv(ohlcv: list):
    df = pd.DataFrame(ohlcv)
    df = df[[0, 1, 2, 3, 4, 5]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"], utc=True, unit="ms")
    df = df.astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    return df


def get_binance_ohlcv(
    market: str,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    timeframe: str = "1d",
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

    ohlcv = []
    for start_date, end_date in zip(start_dates, end_dates, strict=True):
        params = {
            "symbol": market,
            "interval": timeframe,
            "startTime": int(start_date.timestamp() * 1e3),
            "endTime": int(end_date.timestamp() * 1e3),
            "limit": 1000,
        }
        r = requests.get("https://api.binance.com/api/v3/klines", params=params)
        data = r.json()
        ohlcv.extend(data)

    df = _format_binance_ohlcv(ohlcv)
    return df


def load_binance_data(
    market: str,
    timeframe: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
) -> pd.DataFrame:
    df = get_binance_ohlcv(
        market=market,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    df = df.set_index("date")
    return df


def get_binance_markets(exclude: list[str] = None) -> list[str]:
    """Return binance's USDT & active markets.
    Optionally provide a list of markets to exclude
    """
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


def get_cmc_listings(api_key: str, limit=1000, exclude_stables: bool = True):
    """Returns a paginated list of all active cryptocurrencies with latest market data.
    https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyListingsLatest
    """

    r = requests.get(
        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
        headers={"X-CMC_PRO_API_KEY": api_key},
        params={"limit": limit},
    )
    cmc = pd.DataFrame(r.json()["data"])
    cmc["stablecoin"] = cmc["tags"].apply(lambda x: "stablecoin" in x)

    if exclude_stables:
        return cmc.loc[~cmc["stablecoin"], :]
    return cmc


def get_binance_top_markets(cmc_api_key: str, top: int = 500) -> list[str]:
    # get binance markets
    markets = pd.DataFrame({"symbol": get_binance_markets()})

    # get cmc listing
    cmc = get_cmc_listings(api_key=cmc_api_key)
    cmc["symbol"] = cmc["symbol"] + "USDT"

    # keep only top x markets
    top_x = (
        cmc[["symbol"]]
        .merge(markets, how="inner", on="symbol")
        .loc[:top, "symbol"]
        .to_list()
    )
    return top_x
