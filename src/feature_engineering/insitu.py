"""In-situ conditions at L1 (ACE plasma, field and energetic particles) -> hourly inputs.

Training and live runs both go through `hourly_frame`, so a feature means the same
thing in both. Minutes are binned with an availability shift: an hour labelled h
holds the minutes measured in [h - latency, h + 1h - latency), so every value in
the rows before t0 was published by t0 (the real-time lists run a few minutes
behind). Nothing is interpolated; an hour without valid minutes stays NaN.

Hourly columns:
    speed, density, log_temperature    plasma means (log10 K)
    pressure                           dynamic pressure, nPa (1.6726e-6 n V^2)
    log_t_ratio                        log10 of T / T_expected(V); below log10(0.5) marks CME ejecta
    bt, bz                             field magnitude and GSM Bz, nT
    polarity                           -Bx/Bt: +1 field pointing away from the Sun, -1 towards it
    b_var                              field-direction variability within the hour (std of the
                                       components / Bt); low inside smooth CME flux ropes
    icme                               share of minutes with T ratio < 0.5 and Bt >= 8 nT
    log_p47, log_p310                  log10 EPAM proton flux, 47-68 keV and 310-580 keV
"""

import numpy as np
import pandas as pd

LATENCY = pd.Timedelta(minutes=10)
PROTON_MASS_FACTOR = 1.6726e-6  # n [cm^-3] * V^2 [km/s]^2 -> nPa

# Physical ranges; anything outside is treated as bad data.
VALID = {
    "density": (0.05, 150.0),
    "speed": (200.0, 2500.0),
    "temperature": (3e3, 1e7),
    "bx": (-150.0, 150.0),
    "by": (-150.0, 150.0),
    "bz": (-150.0, 150.0),
    "bt": (0.05, 150.0),
    "p47": (1e-2, 1e9),
    "p310": (1e-3, 1e8),
}
HOURLY_COLUMNS = ["speed", "density", "log_temperature", "pressure", "log_t_ratio", "bt", "bz",
                  "polarity", "b_var", "icme", "log_p47", "log_p310"]
MIN_MINUTES_VAR = 10

# CME signatures (the thresholds training.solar_wind_events uses for its catalogue)
TEMP_RATIO = 0.5
ICME_MIN_FIELD = 8.0
SHOCK_DV = 40.0
SHOCK_RATIO = 1.3
SHOCK_SEPARATION = pd.Timedelta(hours=6)
FIELD_SHOCK_RATIO = 2.0
FIELD_SHOCK_DB = 8.0
SHOCK_LOOKAHEAD = pd.Timedelta(minutes=20)  # a shock at t is confirmed by data up to t + 20 min


def expected_temperature(speed):
    """Typical proton temperature (K) for a speed (km/s) in ambient wind (Lopez 1987)."""
    v = np.asarray(speed, dtype=float)
    slow = (0.031 * v - 5.1) ** 2 * 1e3
    fast = (0.51 * v - 142.0) * 1e3
    return np.where(v < 500, slow, fast)


def clean(df):
    """Out-of-range values -> NaN, for the columns VALID knows."""
    df = df.copy()
    for col, (lo, hi) in VALID.items():
        if col in df:
            v = df[col]
            df[col] = v.where((v >= lo) & (v <= hi))
    return df


def combine(archive, realtime):
    """Archive minutes, with real-time minutes only where the archive has no row."""
    if realtime is None or realtime.empty:
        return archive
    if archive is None or archive.empty:
        return realtime
    extra = realtime[~realtime.index.isin(archive.index)]
    return pd.concat([archive, extra]).sort_index()


def minute_frame(plasma, mag, epam=None):
    """Plasma, field and derived per-minute quantities on one minute index."""
    plasma = clean(plasma) if plasma is not None else None
    mag = clean(mag) if mag is not None else None
    cols = {}
    if plasma is not None and len(plasma):
        cols.update({c: plasma[c] for c in ("speed", "density", "temperature")})
    if mag is not None and len(mag):
        cols.update({c: mag[c] for c in ("bx", "by", "bz", "bt")})
    df = pd.DataFrame(cols)
    for c in ("speed", "density", "temperature", "bx", "by", "bz", "bt"):
        if c not in df:
            df[c] = np.nan
    df["pressure"] = PROTON_MASS_FACTOR * df["density"] * df["speed"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["temperature"] / expected_temperature(df["speed"])
        df["log_t_ratio"] = np.log10(ratio.where(ratio > 0))
        df["log_temperature"] = np.log10(df["temperature"])
        df["polarity"] = (-df["bx"] / df["bt"]).clip(-1, 1)
    known = ratio.notna() & df["bt"].notna()
    df["icme"] = ((ratio < TEMP_RATIO) & (df["bt"] >= ICME_MIN_FIELD)).astype(float).where(known)
    if epam is not None and len(epam):
        e = clean(epam)
        df = df.join(np.log10(e[["p47", "p310"]]).rename(columns=lambda c: f"log_{c}"),
                     how="outer")
    for c in ("log_p47", "log_p310"):
        if c not in df:
            df[c] = np.nan
    return df.sort_index()


def hourly_frame(minutes, start, end, latency=LATENCY):
    """Hourly means of `minute_frame` output over [start, end), shifted by `latency`.

    Row h holds minutes measured in [h - latency, h + 1h - latency). Returns a frame on
    the complete hourly index with HOURLY_COLUMNS.
    """
    index = pd.date_range(start, end, freq="h", inclusive="left", name="Time")
    if minutes is None or minutes.empty:
        return pd.DataFrame(np.nan, index=index, columns=HOURLY_COLUMNS)
    label = (minutes.index + latency).floor("h")
    grouped = minutes.groupby(label)
    means = grouped[["speed", "density", "log_temperature", "pressure", "log_t_ratio", "bt",
                     "bz", "polarity", "icme", "log_p47", "log_p310"]].mean()
    var = grouped[["bx", "by", "bz"]].var(ddof=0).sum(axis=1, min_count=3)
    n_field = grouped["bx"].count()
    b_var = np.sqrt(var) / means["bt"]
    means["b_var"] = b_var.where(n_field >= MIN_MINUTES_VAR)
    return means.reindex(index)[HOURLY_COLUMNS]


def detect_shocks(minutes):
    """Times of fast-forward shocks: V, n, T and Bt jump within 20 minutes (or Bt alone).

    `minutes` needs speed, density, temperature and bt. A shock at t uses data up to
    t + SHOCK_LOOKAHEAD, so a forecast may count it once t + SHOCK_LOOKAHEAD has
    been observed (see `confirmed_by`).
    """
    if minutes is None or minutes.empty:
        return pd.DatetimeIndex([], tz="UTC", name="shock")
    ten = minutes[["speed", "density", "temperature", "bt"]].resample("10min").mean()
    before = ten.rolling(2, min_periods=1).mean().shift(1)  # the 20 min before
    after = ten.rolling(2, min_periods=1).mean().shift(-1)  # this and the next 10 min
    plasma = (
        (after["speed"] - before["speed"] >= SHOCK_DV)
        & (after["density"] >= SHOCK_RATIO * before["density"])
        & (after["temperature"] >= SHOCK_RATIO * before["temperature"])
        & (after["bt"] >= SHOCK_RATIO * before["bt"])
    )
    field = (after["bt"] >= FIELD_SHOCK_RATIO * before["bt"]) & (
        after["bt"] - before["bt"] >= FIELD_SHOCK_DB
    )
    jump = plasma | field
    kept = []
    for t in ten.index[jump.to_numpy()]:
        if not kept or t - kept[-1] > SHOCK_SEPARATION:
            kept.append(t)
    if not kept:
        return pd.DatetimeIndex([], tz="UTC", name="shock")
    return pd.DatetimeIndex(kept, name="shock")


def confirmed_by(shocks, cutoff):
    """The shocks whose detection only needed data measured by `cutoff`."""
    shocks = pd.DatetimeIndex(shocks)
    return shocks[shocks + SHOCK_LOOKAHEAD <= cutoff]
