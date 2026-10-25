"""Inputs -> one row of regressors per forecast time t0 (registry.NAMES order).

The same functions build the training table (many t0s over years of history) and
the live row (one t0), so a feature can't mean something different in the two.
Every value at t0 comes from data available by t0:
    in-situ   hourly rows labelled before t0 (insitu.hourly_frame shifts by its latency)
    shocks    confirmed by data measured by t0 - LATENCY
    AIA       the image chosen for cut-off t0 - CH_LATENCY - d days
    flares    peaks by t0 - FLARE_LATENCY
    F10.7     days before the UTC day of t0
Values that are unknown stay NaN here; feature_engineering.impute fills them.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_engineering import coronal_holes, insitu, registry, solar_activity

HOUR = pd.Timedelta(hours=1)
CH_LATENCY = pd.Timedelta(hours=1)
CH_LAG_DAYS = range(6)
CARRY_HOURS = 3
SHOCK_CAP = 168.0
HISTORY = pd.Timedelta(days=29)  # in-situ history a live row needs (27 d lag + 72 h window)


@dataclass
class Inputs:
    """Everything the builders need. None means the source was unavailable."""

    hourly: pd.DataFrame | None = None  # insitu.hourly_frame output
    shocks: pd.DatetimeIndex | None = None  # insitu.detect_shocks output
    coronal_holes: pd.DataFrame | None = None  # coronal_holes.NAMES, indexed by cut-off
    flares: pd.DataFrame | None = None  # peak_time | peak_flux | flare_class
    f107: pd.Series | None = None  # daily, indexed by UTC day
    coronagraph: pd.DataFrame | None = None  # coronagraph.SIGNALS, indexed by image time


def ch_cutoffs(t0):
    """Image cut-off times for the coronal-hole features of a forecast at t0."""
    return [t0 - CH_LATENCY - pd.Timedelta(days=d) for d in CH_LAG_DAYS]


def _at(series, labels):
    return series.reindex(pd.DatetimeIndex(labels)).to_numpy(dtype=float)


def insitu_features(hourly, shocks, t0s):
    t0s = pd.DatetimeIndex(t0s)
    names = [f.name for f in registry.FEATURES if f.group in ("plasma", "field", "cme")]
    out = pd.DataFrame(np.nan, index=t0s, columns=names)
    if hourly is None or hourly.empty:
        return out
    h = hourly
    last = t0s - HOUR  # the newest hourly row a forecast at t0 may use

    def carried(col):
        return _at(h[col].ffill(limit=CARRY_HOURS - 1), last)

    def mean(col, hours, at=last):
        return _at(h[col].rolling(hours, min_periods=hours // 2).mean(), at)

    def slope(col):
        recent = h[col].rolling(3, min_periods=2).mean()
        return (_at(recent, last) - _at(recent, last - 3 * HOUR)) / 3.0

    def lagged(col, lag_hours, how="mean"):
        # the 72 h starting one lag before t0, i.e. the forecast hours one lag earlier
        roll = h[col].rolling(72, min_periods=36)
        series = roll.mean() if how == "mean" else roll.max()
        return _at(series, last - (lag_hours - 72) * HOUR)

    out["density_1h"] = carried("density")
    out["density_6h"] = mean("density", 6)
    out["density_24h"] = mean("density", 24)
    out["density_slope_6h"] = slope("density")
    out["log_temperature_6h"] = mean("log_temperature", 6)
    out["log_t_ratio_6h"] = mean("log_t_ratio", 6)
    out["pressure_1h"] = carried("pressure")
    out["pressure_6h"] = mean("pressure", 6)
    out["pressure_max_24h"] = _at(h["pressure"].rolling(24, min_periods=12).max(), last)
    out["density_lag27d"] = lagged("density", 648)
    out["pressure_lag27d"] = lagged("pressure", 648)
    out["density_lag13d"] = lagged("density", 324)
    out["bt_1h"] = carried("bt")
    out["bt_6h"] = mean("bt", 6)
    out["bt_24h"] = mean("bt", 24)
    out["bz_1h"] = carried("bz")
    out["bz_min_6h"] = _at(h["bz"].rolling(6, min_periods=3).min(), last)
    out["polarity_6h"] = mean("polarity", 6)
    out["polarity_lag27d"] = lagged("polarity", 648)
    out["b_var_6h"] = mean("b_var", 6)
    out["icme_6h"] = mean("icme", 6)
    out["icme_24h"] = mean("icme", 24)
    out["log_p47_1h"] = carried("log_p47")
    out["log_p47_slope_6h"] = slope("log_p47")
    out["log_p310_1h"] = carried("log_p310")

    shocks = pd.DatetimeIndex([] if shocks is None else shocks)
    if shocks.tz is None:
        shocks = shocks.tz_localize("UTC")
    since = np.full(len(t0s), SHOCK_CAP)
    for i, t0 in enumerate(t0s):
        seen = insitu.confirmed_by(shocks, t0 - insitu.LATENCY)
        if len(seen):
            since[i] = min((t0 - seen[-1]) / HOUR, SHOCK_CAP)
    has_field = np.isfinite(out["bt_24h"].to_numpy())
    out["hours_since_shock"] = np.where(has_field, since, np.nan)
    return out


def coronal_hole_features(ch, t0s):
    t0s = pd.DatetimeIndex(t0s)
    names = [f.name for f in registry.FEATURES if f.group == "coronal hole"]
    out = pd.DataFrame(np.nan, index=t0s, columns=names)
    if ch is None or ch.empty:
        return out
    for d in CH_LAG_DAYS:
        cut = t0s - CH_LATENCY - pd.Timedelta(days=d)
        out[f"ch_cm_d{d}"] = _at(ch["ch_cm"], cut)
    cut = t0s - CH_LATENCY
    out["ch_east_d0"] = _at(ch["ch_east"], cut)
    out["ch_disk_d0"] = _at(ch["ch_disk"], cut)
    return out


def build_table(inputs, t0s):
    """Raw regressors (NaN where unknown) for each t0, columns in registry order."""
    t0s = pd.DatetimeIndex(t0s)
    parts = [
        insitu_features(inputs.hourly, inputs.shocks, t0s),
        coronal_hole_features(inputs.coronal_holes, t0s),
        solar_activity.flare_features(inputs.flares, t0s),
        solar_activity.f107_features(inputs.f107, t0s),
    ]
    table = pd.concat(parts, axis=1)[registry.NAMES]
    table.index.name = "t0"
    return table


def coronal_hole_frame(images):
    """{cut-off: (taken, jpeg)} from connectors.sdo -> metrics frame indexed by cut-off."""
    rows = {}
    for cut, (taken, jpeg) in images.items():
        if taken is None:
            rows[cut] = dict.fromkeys(coronal_holes.NAMES, np.nan)
        else:
            rows[cut] = coronal_holes.metrics(jpeg, taken)
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=list(coronal_holes.NAMES))
    frame.index = pd.DatetimeIndex(frame.index, name="cutoff")
    return frame.sort_index()
