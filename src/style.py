import re

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def pretty_ohlcv(
    styler: pd.io.formats.style.Styler,
    cmap: LinearSegmentedColormap,
    format_dict: dict,
    exclude_regex: str,
) -> pd.io.formats.style.Styler:
    """Style ohlcv dataframe"""
    styler.background_gradient(
        axis=1,
        cmap=cmap,
        subset=[col for col in styler.columns if not re.match(exclude_regex, col)],
        vmin=-1,
        vmax=1,
    )
    styler.format(format_dict)
    return styler


def pretty_pivot(
    styler: pd.io.formats.style.Styler, cmap: dict
) -> pd.io.formats.style.Styler:
    """Style pivot (monthly returns) dataframe"""
    styler.background_gradient(axis=None, cmap=cmap, vmin=-1, vmax=1)
    styler.format({col: "{:.2%}" for col in styler.columns})
    return styler
