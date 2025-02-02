import datetime as dt
from inspect import getmembers, isfunction

import pandas as pd


def get_module_functions(module) -> list[str]:
    """Returns list of all functions present in a module"""
    functions = getmembers(module, isfunction)
    return [x[0] for x in functions]


def rename_to_month_names(midx):
    """Rename month of multiindex to month names
    Example: [(1, benchmark), (1, strategy), (2, benchmark) ...]
    -> [(Jan., benchmark), (Jan., strategy), (Feb., benchmark) ...]
    """
    new_cols = [(dt.datetime(2023, col[0], 1).strftime("%b."), col[1]) for col in midx]
    return pd.MultiIndex.from_tuples(new_cols, names=["month", "strategy"])


def flatten_multiindex(midx, joiner: str = " - ") -> list[str]:
    return [joiner.join(col).strip() for col in midx]
