"""Eruptive activity (GOES X-ray flares) and the solar activity level (F10.7) at t0.

Flares: a flare is counted once its peak has been observed, i.e. peak time <=
t0 - FLARE_LATENCY. Large flares often launch CMEs, which reach Earth 1-4 days later.
The flare list holds flares of class B and above, so a window with no flares gets
the floor FLUX_FLOOR (B1); a source that could not be read gives NaN instead.

F10.7: Penticton measures three times a day (17, 20, 23 UT). A forecast on UTC day
D uses only days up to D-1, so every value it sees was published before t0. This
also keeps OMNI history (one value spread over the whole day) safe to train on.
"""

import numpy as np
import pandas as pd

FLARE_LATENCY = pd.Timedelta(minutes=10)
FLUX_FLOOR = 1e-7  # W/m^2, a B1 flare
M_CLASS = 1e-5  # W/m^2
SINCE_CAP = 168.0  # hours
F107_CARRY_DAYS = 3
F107_MEAN_DAYS = 27
F107_MIN_DAYS = 14

FLARE_FEATURES = ["xray_peak_24h", "xray_peak_72h", "m_flares_72h", "hours_since_m_flare"]
F107_FEATURES = ["f107_prev_day", "f107_mean_27d"]


def flare_features(flares, t0s):
    """One row per t0; all NaN when `flares` is None (source unavailable)."""
    t0s = pd.DatetimeIndex(t0s)
    out = pd.DataFrame(np.nan, index=t0s, columns=FLARE_FEATURES)
    if flares is None:
        return out
    times = pd.DatetimeIndex(flares["peak_time"])
    order = np.argsort(times.asi8, kind="stable")
    peaks = times[order]
    flux = flares["peak_flux"].to_numpy(dtype=float)[order]
    big = flux >= M_CLASS
    for t0 in t0s:
        cut = t0 - FLARE_LATENCY
        hi = peaks.searchsorted(cut, side="right")
        lo24 = peaks.searchsorted(t0 - pd.Timedelta(hours=24), side="right")
        lo72 = peaks.searchsorted(t0 - pd.Timedelta(hours=72), side="right")
        f24, f72 = flux[lo24:hi], flux[lo72:hi]
        out.loc[t0, "xray_peak_24h"] = np.log10(max(f24.max(initial=0.0), FLUX_FLOOR))
        out.loc[t0, "xray_peak_72h"] = np.log10(max(f72.max(initial=0.0), FLUX_FLOOR))
        out.loc[t0, "m_flares_72h"] = float(big[lo72:hi].sum())
        last_big = np.flatnonzero(big[:hi])
        since = (cut - peaks[last_big[-1]]) / pd.Timedelta(hours=1) if len(last_big) else SINCE_CAP
        out.loc[t0, "hours_since_m_flare"] = min(float(since), SINCE_CAP)
    return out


def f107_features(daily, t0s):
    """Previous-day F10.7 (carried back up to 3 days) and its 27-day mean; NaN if unknown."""
    t0s = pd.DatetimeIndex(t0s)
    out = pd.DataFrame(np.nan, index=t0s, columns=F107_FEATURES)
    if daily is None or len(daily) == 0:
        return out
    daily = daily.dropna()
    daily.index = pd.DatetimeIndex(daily.index).floor("D")
    daily = daily.groupby(level=0).mean()
    for t0 in t0s:
        today = (t0 - FLARE_LATENCY).floor("D")
        past = daily[daily.index < today]
        recent = past[past.index >= today - pd.Timedelta(days=F107_CARRY_DAYS)]
        if len(recent):
            out.loc[t0, "f107_prev_day"] = float(recent.iloc[-1])
        window = past[past.index >= today - pd.Timedelta(days=F107_MEAN_DAYS)]
        if len(window) >= F107_MIN_DAYS:
            out.loc[t0, "f107_mean_27d"] = float(window.mean())
    return out
