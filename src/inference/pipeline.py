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
