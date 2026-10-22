"""Regressors with a tunable lookback window: (signal, statistic, window) specs.

The 40 fixed regressors in feature_engineering.registry use hand-picked windows
(last hour, 6 h, 24 h). Here the window is a choice: each candidate is a
WindowSpec, named like "bt_mean_12h", and model selection picks among them. The
same `window_table` builds the training table and the live row, under the same
availability rules as feature_engineering.build:

    in-situ   hourly rows labelled before t0 (shifted by insitu.LATENCY)
    AIA       the image for cut-off t0 - CH_LATENCY - lag
    flares    peaks by t0 - FLARE_LATENCY
    F10.7     days before the UTC day of t0

A window statistic needs at least half its hours (or days), otherwise it is NaN.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_engineering import build, solar_activity

HOUR = pd.Timedelta(hours=1)
INSITU_SIGNALS = ("speed", "density", "pressure", "log_temperature", "log_t_ratio", "bt",
                  "polarity", "b_var", "icme", "log_p47", "log_p310")
MEAN_HOURS = (1, 3, 6, 12, 24, 48)
CH_LAG_HOURS = tuple(range(0, 121, 6))
FLARE_HOURS = (24, 48, 72, 120)
F107_DAYS = (1, 7, 27)
LAG_HOURS = {"lag27": 648, "lag13": 324}


@dataclass(frozen=True)
class WindowSpec:
    signal: str
    stat: str  # mean | max | min | slope | lag27 | lag13 | at | count
    hours: int

    @property
    def name(self):
        return f"{self.signal}_{self.stat}_{self.hours}h"

    @property
    def family(self):
        """Specs of one family differ only in their window; selection keeps one of each."""
        return f"{self.signal}_{self.stat}"

    @classmethod
    def parse(cls, name):
        signal, stat, hours = name.rsplit("_", 2)
        return cls(signal, stat, int(hours.rstrip("h")))


def candidates():
    """Every candidate regressor, in a fixed order."""
    specs = [WindowSpec(s, "mean", h) for s in INSITU_SIGNALS for h in MEAN_HOURS]
    specs += [WindowSpec(s, "max", h) for s in ("pressure", "bt") for h in (6, 24, 48)]
    specs += [WindowSpec("bz", "min", h) for h in (3, 6, 24)]
    specs += [WindowSpec(s, "slope", h) for s in ("density", "log_p47") for h in (6, 12, 24)]
    specs += [WindowSpec(s, "lag27", 72) for s in ("density", "pressure", "polarity")]
    specs += [WindowSpec("density", "lag13", 72)]
    specs += [WindowSpec("ch_cm", "at", h) for h in CH_LAG_HOURS]
    specs += [WindowSpec("xray_peak", "max", h) for h in FLARE_HOURS]
    specs += [WindowSpec("m_flares", "count", h) for h in FLARE_HOURS]
    specs += [WindowSpec("f107", "mean", 24 * d) for d in F107_DAYS]
    return specs


def ch_lags(specs):
    """Image lags (hours before t0 - CH_LATENCY) that the specs need."""
    return sorted({s.hours for s in specs if s.signal == "ch_cm"})


def _at(series, labels):
    return series.reindex(pd.DatetimeIndex(labels)).to_numpy(dtype=float)


def _insitu(hourly, spec, last):
    col = hourly.get(spec.signal)
    if col is None:
        return np.full(len(last), np.nan)
    h = spec.hours
    if spec.stat in ("mean", "max", "min"):
        roll = col.rolling(h, min_periods=math.ceil(h / 2))
        return _at(getattr(roll, spec.stat)(), last)
    if spec.stat == "slope":
        half = h // 2
        recent = col.rolling(half, min_periods=math.ceil(half / 2)).mean()
        return (_at(recent, last) - _at(recent, last - half * HOUR)) / half
    if spec.stat in LAG_HOURS:
        roll = col.rolling(h, min_periods=h // 2).mean()
        return _at(roll, last - (LAG_HOURS[spec.stat] - h) * HOUR)
    raise ValueError(f"unknown in-situ statistic {spec.stat!r}")


def _flares(flares, spec, t0s):
    out = np.full(len(t0s), np.nan)
    if flares is None:
        return out
    times = pd.DatetimeIndex(flares["peak_time"])
    order = np.argsort(times.asi8, kind="stable")
    peaks = times[order]
    flux = flares["peak_flux"].to_numpy(dtype=float)[order]
    hi = peaks.searchsorted(t0s - solar_activity.FLARE_LATENCY, side="right")
    lo = peaks.searchsorted(t0s - spec.hours * HOUR, side="right")
    for i in range(len(t0s)):
        window = flux[lo[i] : hi[i]]
        if spec.signal == "xray_peak":
            out[i] = np.log10(max(window.max(initial=0.0), solar_activity.FLUX_FLOOR))
        else:
            out[i] = float((window >= solar_activity.M_CLASS).sum())
    return out


def _f107(daily, spec, t0s):
    out = np.full(len(t0s), np.nan)
    if daily is None or len(daily) == 0:
        return out
    daily = daily.dropna()
    daily.index = pd.DatetimeIndex(daily.index).floor("D")
    daily = daily.groupby(level=0).mean()
    days = spec.hours // 24
    for i, t0 in enumerate(t0s):
        today = (t0 - solar_activity.FLARE_LATENCY).floor("D")
        past = daily[daily.index < today]
        if days == 1:  # the previous day, carried back a few days like the fixed regressor
            recent = past[past.index >= today - pd.Timedelta(days=solar_activity.F107_CARRY_DAYS)]
            out[i] = float(recent.iloc[-1]) if len(recent) else np.nan
        else:
            window = past[past.index >= today - pd.Timedelta(days=days)]
            out[i] = float(window.mean()) if len(window) >= math.ceil(days / 2) else np.nan
    return out


def window_table(inputs, t0s, specs):
    """One column per spec (NaN where unknown), one row per t0."""
    t0s = pd.DatetimeIndex(t0s)
    last = t0s - HOUR
    cols = {}
    for spec in specs:
        if spec.signal in INSITU_SIGNALS or spec.signal == "bz":
            hourly = inputs.hourly
            cols[spec.name] = (np.full(len(t0s), np.nan) if hourly is None or hourly.empty
                               else _insitu(hourly, spec, last))
        elif spec.signal == "ch_cm":
            ch = inputs.coronal_holes
            cut = t0s - build.CH_LATENCY - spec.hours * HOUR
            cols[spec.name] = (np.full(len(t0s), np.nan) if ch is None or ch.empty
                               else _at(ch["ch_cm"], cut))
        elif spec.signal in ("xray_peak", "m_flares"):
            cols[spec.name] = _flares(inputs.flares, spec, t0s)
        elif spec.signal == "f107":
            cols[spec.name] = _f107(inputs.f107, spec, t0s)
        else:
            raise ValueError(f"unknown signal {spec.signal!r}")
    table = pd.DataFrame(cols, index=t0s)
    table.index.name = "t0"
    return table
