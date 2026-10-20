"""Pinned train/eval split of the hourly real-time ACE speed series (see docs/evaluation.md).

    train  [DATA_START, TRAIN_END)   4 years, for fitting models
    eval   [TRAIN_END, EVAL_END)     1 year, for TIMSE scoring only

The dates are fixed so that re-pulling the data never moves the eval year.
"""

from pathlib import Path

import pandas as pd

DATA_START = pd.Timestamp("2021-10-20T00:00Z")
TRAIN_END = pd.Timestamp("2025-10-20T00:00Z")
EVAL_END = pd.Timestamp("2026-10-20T00:00Z")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ace_rtsw"
HOURLY_CSV = DATA_DIR / "speed_hourly.csv"


def load_hourly(path=HOURLY_CSV, start=DATA_START, end=EVAL_END):
    """Hourly speed (km/s) for [start, end) as a float Series on a UTC hourly index.

    Missing hours are NaN; the index is complete, with no gaps.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m training.pull_ace_rtsw`")
    df = pd.read_csv(path, usecols=["Time", "value"])
    series = pd.Series(
        df["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(df["Time"], utc=True), name="Time"),
        name="speed",
    )
    expected = pd.date_range(start, end, freq="h", inclusive="left", name="Time")
    if not expected.isin(series.index).all():
        raise ValueError(
            f"{path} does not cover {start:%Y-%m-%d} .. {end:%Y-%m-%d}; "
            "re-run `python -m training.pull_ace_rtsw`"
        )
    return series.reindex(expected)


def load_train(path=HOURLY_CSV):
    """The training years. The only data a model may be fitted on."""
    return load_hourly(path, DATA_START, TRAIN_END)


def load_eval(path=HOURLY_CSV):
    """The held-out year. Use only for scoring, never for fitting."""
    return load_hourly(path, TRAIN_END, EVAL_END)
