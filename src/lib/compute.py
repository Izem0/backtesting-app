import datetime as dt

import pandas as pd


def compute_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """
    Use daily returns (with datetime as index) to compute monthly returns
    """
    name = daily_returns.name
    group = daily_returns.groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).transform(lambda x: (x + 1).cumprod() - 1)
    group = group.groupby([group.index.year, group.index.month]).last()
    group.index.set_names(["year", "month"], inplace=True)
    group = group.to_frame(name).reset_index()
    group["date"] = [
        dt.datetime(year, int(month), 1)
        for year, month in zip(group["year"], group["month"], strict=True)
    ]
    return pd.Series(data=group[name].values, index=group["date"], name=name)
