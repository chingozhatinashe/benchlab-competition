"""Forecast orchestration: fetch inputs -> build the hourly series -> run the model.

Every step degrades instead of failing: if the model asset can't be loaded the
forecast is persistence, if no data arrives it is the training mean (or a fixed
default), and a non-finite model output falls back the same way. Logs go to
stderr; stdout belongs to model.py's JSON.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from connectors import ACEArchiveConnector, SWPCConnector
from connectors.base import STATUS_OK, to_utc
from feature_engineering.hourly import hourly_series
from models.final_baseline import ASSET, FinalBaseline

FORECAST_HOURS = 72
DEFAULT_SPEED = 420.0  # used only with no model asset and no data at all
FALLBACK_HISTORY_HOURS = 48
REALTIME_SPAN = pd.Timedelta(hours=24)  # the SWPC 1-minute feed covers about a day
OFFLINE_ENV = "SWP_OFFLINE"  # "1" skips all network access (contract tests)


def log(msg):
    print(f"[inference] {msg}", file=sys.stderr, flush=True)


def minute_speeds(frame):
    """Valid 1-minute speeds from a connector frame, as a Series on a UTC index."""
    ok = frame[frame["status"] == STATUS_OK]
    return pd.Series(
        ok["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(ok["Time"], name="Time"),
        name="speed",
    )


def realtime_minutes(frame):
    """ACE minutes from the SWPC feed; every spacecraft's average if ACE has none."""
    ok = frame[frame["status"] == STATUS_OK]
    ace = ok[ok["source"].astype(str).str.upper() == "ACE"]
    use = ace if len(ace) else ok
    return use.groupby("Time")["value"].mean()


def empty_history(t0, hours):
    index = pd.date_range(t0 - pd.Timedelta(hours=hours), t0, freq="h", inclusive="left")
    return pd.Series(np.nan, index=index.rename("Time"), name="speed")


def fetch_history(t0, hours, archive=None, realtime=None):
    """Hourly speed for the `hours` hours before t0 (labels t0 - hours .. t0 - 1h).

    Only minutes stamped before t0 are used, so the series holds exactly what
    was observed by t0. The daily archive (the training source) gives the bulk;
    the SWPC real-time feed fills the recent minutes the archive doesn't have yet.
    """
    archive = archive or ACEArchiveConnector()
    realtime = realtime or SWPCConnector()
    start = t0 - pd.Timedelta(hours=hours)
    last_minute = t0 - pd.Timedelta(minutes=1)

    minutes = minute_speeds(archive.fetch(start, last_minute))
    recent = realtime_minutes(realtime.fetch(max(start, t0 - REALTIME_SPAN), last_minute))
    top_up = recent[~recent.index.isin(minutes.index)]
    log(f"minutes: {len(minutes)} from the archive, {len(top_up)} added from SWPC real time")
    combined = pd.concat([minutes, top_up]).sort_index() if len(top_up) else minutes
    return hourly_series(combined, start, t0)


def load_model(path=ASSET):
    try:
        return FinalBaseline.load(path)
    except Exception as e:
        log(f"model asset {path} unusable ({e!r}); falling back to persistence")
        return None


def fallback(history, model):
    observed = history.dropna()
    if len(observed):
        return np.full(FORECAST_HOURS, float(observed.iloc[-1]))
    return np.full(FORECAST_HOURS, model.mean if model is not None else DEFAULT_SPEED)


def run_forecast(t0, model=None, history=None):
    """72 hourly speeds (km/s) for t0+1h .. t0+72h, from data observed by t0."""
    started = time.monotonic()
    t0 = to_utc(t0)
    model = model or load_model()
    hours = model.config.history_hours if model is not None else FALLBACK_HISTORY_HOURS

    if history is None:
        if os.environ.get(OFFLINE_ENV) == "1":
            log(f"{OFFLINE_ENV}=1: no network, empty history")
            history = empty_history(t0, hours)
        else:
            try:
                history = fetch_history(t0, hours)
            except Exception as e:
                log(f"fetching history failed ({e!r}); continuing without data")
                history = empty_history(t0, hours)

    observed = history.dropna()
    latest = f"{observed.index[-1]:%Y-%m-%d %H:%M}" if len(observed) else "none"
    log(f"t0 {t0:%Y-%m-%d %H:%M}: {len(observed)}/{len(history)} hours observed, latest {latest}")

    forecast = None
    if model is not None:
        try:
            forecast = np.asarray(model.predict(history, t0), dtype=float)
        except Exception as e:
            log(f"model failed ({e!r}); falling back to persistence")
    if forecast is None or forecast.shape != (FORECAST_HOURS,) or not np.isfinite(forecast).all():
        forecast = fallback(history, model)
        log("using the fallback forecast")

    log(f"done in {time.monotonic() - started:.1f} s; "
        f"forecast {forecast.min():.0f}..{forecast.max():.0f} km/s")
    return [float(v) for v in forecast]


# ---------------------------------------------------------------------------
# Linear model (models.linear_model), opt-in with SWP_MODEL=linear.
# The submitted forecast stays the final baseline unless that is set.

MODEL_ENV = "SWP_MODEL"


def load_linear(path=None):
    from models.linear_model import ASSET as LINEAR_ASSET
    from models.linear_model import LinearModel

    try:
        return LinearModel.load(path or LINEAR_ASSET)
    except Exception as e:
        log(f"linear model asset unusable ({e!r}); falling back to the final baseline")
        return None


def linear_features(linear, t0, inputs=None):
    """The selected regressors at t0 as {name: value} (NaN where unknown; imputed later)."""
    from feature_engineering import windows
    from inference.fetch_inputs import fetch_inputs

    specs = [windows.WindowSpec.parse(n) for n in linear.config.features]
    if inputs is None:
        inputs, _ = fetch_inputs(t0, ch_lag_hours=windows.ch_lags(specs))
    row = windows.window_table(inputs, [t0], specs).iloc[0]
    return {k: float(v) for k, v in row.items()}


def run_linear_forecast(t0, linear=None, baseline=None, history=None, features=None):
    """Linear-model forecast; the final baseline (and its fallbacks) if anything fails.

    Live pipeline: regressors fetched for t0 -> imputation with the training medians
    -> the selected regressors -> GLM. The speed history and the regressors are
    fetched in parallel.
    """
    started = time.monotonic()
    t0 = to_utc(t0)
    linear = linear or load_linear()
    baseline = baseline or load_model()
    if linear is None:
        return run_forecast(t0, baseline, history)
    hours = max(linear.config.history_hours,
                baseline.config.history_hours if baseline is not None else 0)
    offline = os.environ.get(OFFLINE_ENV) == "1"
    if history is None and offline:
        history = empty_history(t0, hours)
    if features is None and offline:
        features = {}
    if history is None or features is None:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            h = pool.submit(fetch_history, t0, hours) if history is None else None
            f = pool.submit(linear_features, linear, t0) if features is None else None
            if h is not None:
                try:
                    history = h.result()
                except Exception as e:
                    log(f"fetching history failed ({e!r}); continuing without data")
                    history = empty_history(t0, hours)
            if f is not None:
                try:
                    features = f.result()
                except Exception as e:
                    log(f"regressors failed ({e!r}); all imputed")
                    features = {}

    missing = [n for n in linear.config.features
               if not np.isfinite(features.get(n, np.nan))]
    log(f"linear model: {len(linear.config.features) - len(missing)}/"
        f"{len(linear.config.features)} regressors observed; imputed: {missing or 'none'}")
    try:
        forecast = np.asarray(linear.predict(history, features), dtype=float)
    except Exception as e:
        log(f"linear model failed ({e!r})")
        forecast = None
    if forecast is None or forecast.shape != (FORECAST_HOURS,) or not np.isfinite(forecast).all():
        log("falling back to the final baseline")
        return run_forecast(t0, baseline, history)
    log(f"linear model done in {time.monotonic() - started:.1f} s; "
        f"forecast {forecast.min():.0f}..{forecast.max():.0f} km/s")
    return [float(v) for v in forecast]


def run_selected_forecast(t0):
    """The forecast model.py submits: the final baseline, or the linear model if opted in."""
    if os.environ.get(MODEL_ENV, "").lower() == "linear":
        return run_linear_forecast(t0)
    return run_forecast(t0)
