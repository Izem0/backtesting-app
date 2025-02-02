import pandas as pd
import pandas.io.formats.style
from matplotlib.colors import LinearSegmentedColormap


def pretty_ohlcv(
    styler: pd.io.formats.style.Styler, cmap: LinearSegmentedColormap
) -> pd.io.formats.style.Styler:
    """Style ohlcv dataframe"""
    styler.background_gradient(
        axis=1,
        cmap=cmap,
        subset=["benchmark_cum_return", "strategy_cum_return"],
        vmin=-1,
        vmax=1,
    )
    # styler.background_gradient(
    #     axis=1,
    #     cmap="Reds",
    #     subset=["signal"],
    #     vmin=0,
    #     vmax=1,
    # )
    format_dict = {col: "{:.2%}" for col in styler.columns if "return" in col}
    format_dict.update(
        {
            "open": "${:.2f}",
            "signal": "{:.1f}",
        }
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
