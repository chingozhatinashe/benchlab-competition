"""Minute speeds -> hourly speeds, the same way for training data and live runs."""

import numpy as np
import pandas as pd

STATUS_OK = "ok"
STATUS_MISSING = "missing"


def to_hourly(minutes, start, end):
    """Hourly mean of the valid minutes over [start, end), labelled by the hour's start.

    `minutes` is a Series of 1-minute speeds on a UTC index, NaN where invalid.
    An hour with no valid minutes stays missing; nothing is imputed. Returns a
    DataFrame with Time | status | value | n_valid.
    """
    hours = pd.date_range(start, end, freq="h", inclusive="left", name="Time")
    grouped = minutes.groupby(minutes.index.floor("h"))
    df = pd.DataFrame({"value": grouped.mean(), "n_valid": grouped.count()}).reindex(hours)
    df["n_valid"] = df["n_valid"].fillna(0).astype(int)
    df["status"] = np.where(df["n_valid"] > 0, STATUS_OK, STATUS_MISSING)
    return df.reset_index()[["Time", "status", "value", "n_valid"]]


def hourly_series(minutes, start, end):
    """to_hourly as a float Series of speeds on a complete hourly index (NaN = missing)."""
    hourly = to_hourly(minutes, start, end)
    return pd.Series(
        hourly["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(hourly["Time"], name="Time"),
        name="speed",
    )
