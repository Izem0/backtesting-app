import datetime as dt

import pandas as pd


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


def compute_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Use daily returns (with datetime as index) to compute monthly returns
    """
    group = daily_returns.groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).transform(lambda x: (x + 1).cumprod() - 1)
    group = group.groupby([group.index.year, group.index.month]).last()
    group.index.set_names(["year", "month"], inplace=True)
    group = group.to_frame().reset_index()
    group["date"] = [
        dt.datetime(year, int(month), 1)
        for year, month in zip(group["year"], group["month"], strict=True)
    ]
    group.sort_values(["year", "month"], inplace=True)
    return group


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
