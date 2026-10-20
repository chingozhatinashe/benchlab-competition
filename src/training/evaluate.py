"""Score a model with TIMSE on the held-out eval year (see docs/evaluation.md).

    from training.evaluate import evaluate
    evaluate(MyModel())   # -> {"timse": ..., "n_forecasts": ..., ...}

A model has two methods:

    fit(train)            train: hourly speed Series for the training years only
    predict(history, t0)  history: every hour fully observed by t0 (labels <= t0 - 1h)
                          returns 72 forecasts for the hours labelled t0+1h .. t0+72h

The harness calls fit itself, with the training years only, so a model is never
fitted on eval data. Eval hours before t0 do reach predict as inputs, exactly as
recent observations would in the live run.
"""

import numpy as np
import pandas as pd

from training.datasets import EVAL_END, HOURLY_CSV, TRAIN_END, load_hourly
from training.metrics import timse

HORIZON = 72
CYCLE = pd.Timedelta(hours=6)


def forecast_times(start=TRAIN_END, end=EVAL_END):
    """Every 6-hourly t0 whose whole 72-hour target window lies inside [start, end)."""
    last = end - pd.Timedelta(hours=HORIZON + 1)
    return pd.date_range(start, last, freq=CYCLE)


def evaluate(model, path=HOURLY_CSV):
    """Fit `model` on the training years, forecast every eval t0, and return TIMSE."""
    series = load_hourly(path)
    model.fit(series[series.index < TRAIN_END].copy())

    forecasts, actuals = [], []
    for t0 in forecast_times():
        pos = series.index.get_loc(t0)
        history = series.iloc[:pos].copy()
        forecast = np.asarray(model.predict(history, t0), dtype=float).ravel()
        if forecast.shape != (HORIZON,) or not np.isfinite(forecast).all():
            raise ValueError(f"predict at {t0} must return {HORIZON} finite values")
        forecasts.append(forecast)
        actuals.append(series.iloc[pos + 1 : pos + 1 + HORIZON].to_numpy())

    forecasts, actuals = np.concatenate(forecasts), np.concatenate(actuals)
    return {
        "timse": timse(actuals, forecasts),
        "n_forecasts": len(forecasts) // HORIZON,
        "n_scored": int(np.isfinite(actuals).sum()),
        "n_missing_actual": int(np.isnan(actuals).sum()),
    }
