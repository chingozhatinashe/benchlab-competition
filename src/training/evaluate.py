"""Score a model with TIMSE on the held-out eval year (see docs/evaluation.md).

    from training.evaluate import evaluate
    evaluate(MyModel())   # -> {"timse": ..., "n_forecasts": ..., ...}

A model has two methods:

    fit(train)            train: hourly speed Series for the training years only
    predict(history, t0)  history: every hour fully observed by t0 (labels <= t0 - 1h)
                          returns 72 forecasts for the hours labelled t0+1h .. t0+72h

The harness calls fit itself, with the training years only. Eval hours before t0
reach predict, exactly as recent observations would in the live run, so a model
may also refit inside predict on that history (see training.baselines.Rolling).
It never sees the hours being forecast.
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


MIN_RUN_POINTS = 2  # a run needs two observations for TIMSE to have a threshold gap


def run_forecasts(model, path=HOURLY_CSV):
    """Fit `model` on the training years and forecast every eval t0.

    Returns (t0s, forecasts, actuals); the arrays are (runs x 72), row i is the
    run at t0s[i], and NaN in `actuals` marks a missing observation.
    """
    series = load_hourly(path)
    model.fit(series[series.index < TRAIN_END].copy())

    t0s = forecast_times()
    forecasts, actuals = [], []
    for t0 in t0s:
        pos = series.index.get_loc(t0)
        history = series.iloc[:pos].copy()
        forecast = np.asarray(model.predict(history, t0), dtype=float).ravel()
        if forecast.shape != (HORIZON,) or not np.isfinite(forecast).all():
            raise ValueError(f"predict at {t0} must return {HORIZON} finite values")
        forecasts.append(forecast)
        actuals.append(series.iloc[pos + 1 : pos + 1 + HORIZON].to_numpy())
    return t0s, np.stack(forecasts), np.stack(actuals)


def run_timse(forecasts, actuals):
    """TIMSE of each forecast run on its own 72 hours.

    NaN for a run with fewer than MIN_RUN_POINTS observations.
    """
    scores = np.full(len(forecasts), np.nan)
    for i, (f, a) in enumerate(zip(forecasts, actuals, strict=True)):
        if np.isfinite(a).sum() >= MIN_RUN_POINTS:
            scores[i] = timse(a, f)
    return scores


def score(forecasts, actuals):
    """Pooled TIMSE, the distribution of per-run TIMSE, RMSE and counts."""
    observed = np.isfinite(actuals)
    per_run = run_timse(forecasts, actuals)
    scored = per_run[np.isfinite(per_run)]
    return {
        "timse": timse(actuals, forecasts),
        "run_timse_mean": float(scored.mean()),
        "run_timse_median": float(np.median(scored)),
        "run_timse_std": float(scored.std(ddof=1)),
        "run_timse_p10": float(np.percentile(scored, 10)),
        "run_timse_p90": float(np.percentile(scored, 90)),
        "run_timse_max": float(scored.max()),
        # Secondary diagnostic only; models are ranked by TIMSE.
        "rmse": float(np.sqrt(np.mean((actuals[observed] - forecasts[observed]) ** 2))),
        "n_forecasts": len(forecasts),
        "n_runs_scored": len(scored),
        "n_scored": int(observed.sum()),
        "n_missing_actual": int((~observed).sum()),
    }


def evaluate(model, path=HOURLY_CSV):
    """Fit `model` on the training years, forecast every eval t0, and score it.

    `timse` pools every run into one score. The `run_timse_*` entries summarise
    the TIMSE of each 72-hour run on its own.
    """
    _, forecasts, actuals = run_forecasts(model, path)
    return score(forecasts, actuals)
