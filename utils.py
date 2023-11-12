from inspect import getmembers, isfunction

import pandas as pd
import requests


def _chunks_dates(dates, n=1000):
    """Return 2 lists of start dates & end dates given a series of dates."""
    start_dates, end_dates = [], []
    for i in range(0, len(dates), n):
        chunk = dates[i:i + n]
        start_dates.append(chunk[0])
        end_dates.append(chunk[-1])

    return start_dates, end_dates


def get_binance_ohlcv(market: str, timeframe: str = '1d', start_date: pd.Timestamp = None, end_date: pd.Timestamp = None) -> pd.DataFrame:
    tf_mapping = {
        '1w': '1W',
        '1d': '1D',
        '4h': '4H',
        '1h': '1H',
        '15m': '15T',
        '5m': '5T',
    }

    dates = pd.date_range(start=start_date, end=end_date, freq=tf_mapping[timeframe])
    start_dates, end_dates = _chunks_dates(dates, n=1000)

    df = pd.DataFrame()
    for start_date, end_date in zip(start_dates, end_dates):
        params = {
            'symbol': market,
            'interval': timeframe,
            'startTime': int(start_date.timestamp() * 1e3),
            'endTime': int(end_date.timestamp() * 1e3),
            'limit': 1000
            }
        r = requests.get('https://api.binance.com/api/v3/klines', params=params)
        dfx = pd.DataFrame(r.json())
        df = pd.concat([df, dfx], ignore_index=True)

    df = df[[0, 1, 2, 3, 4, 5]]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date'], utc=True, unit='ms')
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    return df


def get_functions(module) -> list[str]:
    """Returns list of all functions present in a module"""
    functions = getmembers(module, isfunction)
    return [x[0] for x in functions]


def compute_returns(price, signal, fees=0.001) -> pd.DataFrame:
    # make dataframe
    df = pd.DataFrame({"price": price, "signal": signal}, index=price.index)
    # Calculate the benchmark daily returns
    df["benchmark_return"] = df["price"].pct_change()
    # Calculate the strategy returns with fees
    df["strategy_return"] = df["signal"] * df["benchmark_return"]
    entry_exit_days = df['signal'].diff().fillna(0)  # 1 when entering or exiting, 0 otherwise
    df['strategy_return'] = df['strategy_return'] - fees * entry_exit_days

    # calculate the benchmark cumulative returns
    df["benchmark_cum_return"] = (1 + df["benchmark_return"]).cumprod() - 1
    # Calculate the cumulative returns
    df["strategy_cum_return"] = (1 + df["strategy_return"]).cumprod() - 1
    return df[["benchmark_return", "benchmark_cum_return", "strategy_return", "strategy_cum_return"]]
