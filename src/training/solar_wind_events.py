"""Identify CMEs, shocks and high-speed-stream arrivals in the ACE real-time data.

Dev-only, for analysing forecasts (src/experiments/final_baseline_analysis.ipynb).
No external CME catalogue is used: CMEs are found from their in-situ signatures,
the way catalogues such as Richardson & Cane's are built.

- **Shock:** within 20 minutes, speed rises by >= 40 km/s while density, temperature and
  field strength each rise by >= 30%. A fast-forward shock moves all four together;
  a stream interface lowers the density, so it doesn't qualify. Real-time plasma
  can fail during the largest events (in January 2026 it read a flat 290 km/s
  through the shock), so a field jump alone also counts: Bt at least doubles and
  rises by >= 8 nT.
- **ICME (ejecta):** proton temperature below half the temperature expected for the
  speed (Lopez 1987, as used by Richardson & Cane 1995) for >= 8 h, gaps <= 3 h
  bridged, with mean field >= 8 nT.
- **Possible ICME:** the same cold interval with a weaker field. These may be weak
  CMEs or cold slow wind; they are kept out of the clean no-CME set.
- **CME-affected hours:** each ICME, plus its sheath (from a shock up to 24 h before
  the ejecta), plus 24 h after a shock with no ejecta found (a CME flank).
- **Stream arrival:** outside CME-affected hours, a rise of >= 120 km/s within 48 h
  that reaches >= 500 km/s. The arrival time is the hour the speed crosses the
  midpoint of the rise.

The thresholds are standard literature values, fixed before looking at forecast
errors. Plasma comes from the cached SWEPAM files (training.pull_ace_rtsw) and
the field from the matching MAG files, cached in src/data/ace_mag/raw/.
"""

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from connectors.ace_archive import parse_daily
from connectors.base import HTTPNotFound, http_get_bytes
from feature_engineering.insitu import (  # noqa: F401  (re-exported for the notebooks)
    FIELD_SHOCK_DB,
    FIELD_SHOCK_RATIO,
    ICME_MIN_FIELD,
    SHOCK_DV,
    SHOCK_RATIO,
    SHOCK_SEPARATION,
    TEMP_RATIO,
    detect_shocks,
    expected_temperature,
)
from training.datasets import DATA_DIR

PLASMA_DIR = DATA_DIR / "raw"
MAG_DIR = DATA_DIR.parent / "ace_mag" / "raw"
ARCHIVE = "https://soho.nascom.nasa.gov/sdb/ace/daily"
PLASMA_FILE = "{day:%Y%m%d}_ace_swepam_1m.txt"
MAG_FILE = "{day:%Y%m%d}_ace_mag_1m.txt"

# ICME (the shock and cold-plasma thresholds live in feature_engineering.insitu)
ICME_MIN_HOURS = 8
ICME_GAP_HOURS = 3
SHEATH_MAX = pd.Timedelta(hours=24)
FLANK_HOURS = pd.Timedelta(hours=24)
# Stream arrival
STREAM_RISE = 120.0
STREAM_WINDOW = 48
STREAM_PEAK = 500.0
STREAM_SEPARATION = pd.Timedelta(days=3)


def log(msg):
    print(f"[solar_wind_events] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------- data
def parse_plasma(text):
    """1-minute density (cm^-3), speed (km/s) and temperature (K); NaN where flagged."""
    return parse_daily(text, "swepam")


def parse_mag(text):
    """1-minute field components, magnitude Bt and angles (nT, GSM); NaN where flagged."""
    return parse_daily(text, "mag")


def _cached(day, directory, pattern):
    path = Path(directory) / pattern.format(day=day)
    if path.exists():
        return path.read_text()
    try:
        text = http_get_bytes(f"{ARCHIVE}/{pattern.format(day=day)}").decode("ascii")
    except HTTPNotFound:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def load_minutes(start, end, kind):
    """1-minute plasma or mag data for [start, end), downloading days not yet cached."""
    directory, pattern, parse = {
        "plasma": (PLASMA_DIR, PLASMA_FILE, parse_plasma),
        "mag": (MAG_DIR, MAG_FILE, parse_mag),
    }[kind]
    days = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end), freq="D",
                         inclusive="left")
    with ThreadPoolExecutor(max_workers=8) as pool:
        texts = list(pool.map(lambda d: _cached(d, directory, pattern), days))
    parts = [parse(t) for t in texts if t]
    log(f"{kind}: {len(parts)}/{len(days)} days")
    df = pd.concat(parts).sort_index()
    return df[(df.index >= start) & (df.index < end)]


def load_conditions(start, end):
    """Minute plasma + field on one index, for [start, end)."""
    return load_minutes(start, end, "plasma").join(load_minutes(start, end, "mag"), how="outer")


def _runs(mask):
    """(start, end) index pairs of consecutive True values, end exclusive."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(edges[::2], edges[1::2], strict=True))


def detect_icmes(hourly, shocks):
    """ICME ejecta from hourly conditions, with sheaths from preceding shocks.

    `hourly` has speed, temperature and bt on a complete hourly index. Returns a
    DataFrame with one row per CME: start (shock or ejecta start), ejecta_start,
    end, shock, peak speed, max field, min temperature ratio.
    """
    shocks = pd.DatetimeIndex(shocks)
    if len(shocks) == 0:
        shocks = pd.DatetimeIndex([], tz="UTC")
    ratio = hourly["temperature"] / expected_temperature(hourly["speed"])
    cold = (ratio < TEMP_RATIO).to_numpy()
    # bridge short gaps (missing or briefly warmer hours) inside a cold interval
    for s, e in _runs(~cold):
        if s > 0 and e < len(cold) and e - s <= ICME_GAP_HOURS:
            cold[s:e] = True
    rows = []
    for s, e in _runs(cold):
        if e - s < ICME_MIN_HOURS:
            continue
        block = hourly.iloc[s:e]
        strong = block["bt"].mean() >= ICME_MIN_FIELD
        start, end = hourly.index[s], hourly.index[e - 1] + pd.Timedelta(hours=1)
        lead = shocks[(shocks <= start + pd.Timedelta(hours=1)) & (shocks >= start - SHEATH_MAX)]
        shock = lead[0] if len(lead) and strong else pd.NaT
        rows.append({
            "kind": "ICME" if strong else "possible ICME",
            "start": shock.floor("h") if pd.notna(shock) else start,
            "ejecta_start": start,
            "end": end,
            "shock": shock,
            "peak_speed": block["speed"].max(),
            "max_bt": block["bt"].max(),
            "min_temp_ratio": ratio.iloc[s:e].min(),
        })
    # shocks with no ejecta behind them: a CME flank or a shock-only passage
    used = {r["shock"] for r in rows if pd.notna(r["shock"])}
    flanks = [
        {"kind": "shock only", "start": t.floor("h"), "ejecta_start": pd.NaT,
         "end": t.floor("h") + FLANK_HOURS, "shock": t, "peak_speed": np.nan,
         "max_bt": np.nan, "min_temp_ratio": np.nan}
        for t in shocks if t not in used
    ]
    columns = ["kind", "start", "ejecta_start", "end", "shock", "peak_speed", "max_bt",
               "min_temp_ratio"]
    out = pd.DataFrame(rows + flanks, columns=columns)
    return out.sort_values("start").reset_index(drop=True)


def cme_mask(index, events, kinds=("ICME", "shock only")):
    """True for the hours in `index` inside any event of the given kinds."""
    mask = np.zeros(len(index), dtype=bool)
    events = events[events["kind"].isin(kinds)]
    for start, end in zip(events["start"], events["end"], strict=True):
        mask |= (index >= start) & (index < end)
    return pd.Series(mask, index=index, name="cme")


def detect_stream_arrivals(speed, cme):
    """Arrival times of high-speed streams outside CME-affected hours.

    A rise of >= STREAM_RISE km/s within STREAM_WINDOW hours reaching >= STREAM_PEAK;
    the arrival is the hour the smoothed speed first crosses the rise's midpoint.
    """
    smooth = speed.interpolate(limit=6).rolling(6, center=True, min_periods=3).median()
    v = smooth.to_numpy()
    n = len(v)
    arrivals = []
    i = 0
    while i < n - 1:
        if not np.isfinite(v[i]):
            i += 1
            continue
        window = v[i + 1 : i + 1 + STREAM_WINDOW]
        if window.size == 0 or not np.isfinite(window).any():
            i += 1
            continue
        j = int(np.nanargmax(window)) + i + 1
        if v[j] - v[i] >= STREAM_RISE and v[j] >= STREAM_PEAK:
            # follow the whole rise: from its lowest point to its peak
            segment = v[i : j + 1]
            base = i + len(segment) - 1 - int(np.nanargmin(segment[::-1]))  # last minimum
            while True:
                ahead = v[j + 1 : base + 1 + STREAM_WINDOW]
                if ahead.size == 0 or not np.isfinite(ahead).any() or np.nanmax(ahead) <= v[j]:
                    break
                j = j + 1 + int(np.nanargmax(ahead))
            mid = v[base] + 0.5 * (v[j] - v[base])
            k = base + int(np.argmax(v[base : j + 1] >= mid))
            t = speed.index[k]
            near_cme = cme.iloc[max(k - 12, 0) : k + 13].any()
            if not near_cme and (not arrivals or t - arrivals[-1] > STREAM_SEPARATION):
                arrivals.append(t)
            i = j
        else:
            i += 1
    return pd.DatetimeIndex(arrivals, name="arrival")


# ---------------------------------------------------------------------- run labels
REGIMES = ["CME", "possible CME", "stream arrival", "fast stream", "slow wind", "intermediate"]
NO_CME = REGIMES[2:]


def classify_runs(t0s, actual, cme, arrivals, possible=None, horizon=72):
    """One regime per forecast run, from what happened during its 72 target hours.

    CME            any target hour is CME-affected
    possible CME   otherwise, any target hour is in a weak-field cold interval
    stream arrival a non-CME stream arrives during the target hours
    fast stream    otherwise, mean observed speed >= 550 km/s
    slow wind      otherwise, max observed speed < 450 km/s
    intermediate   everything else
    """
    labels = []
    for t0, a in zip(t0s, actual, strict=True):
        start, end = t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=horizon + 1)
        window = (cme.index >= start) & (cme.index < end)
        if cme[window].any():
            labels.append("CME")
        elif possible is not None and possible[window].any():
            labels.append("possible CME")
        elif ((arrivals >= start) & (arrivals < end)).any():
            labels.append("stream arrival")
        elif np.nanmean(a) >= 550:
            labels.append("fast stream")
        elif np.nanmax(a) < 450:
            labels.append("slow wind")
        else:
            labels.append("intermediate")
    return pd.Categorical(labels, categories=REGIMES)


# ---------------------------------------------------------------------- routing regimes
# The regime-model study (training.regime_models) routes on 5 groups: possible CMEs
# join the CMEs, since both are cold ejecta-like intervals.
ROUTES = ["CME", "stream arrival", "fast stream", "intermediate", "slow wind"]
ROUTE_OF = {"CME": "CME", "possible CME": "CME", "stream arrival": "stream arrival",
            "fast stream": "fast stream", "intermediate": "intermediate",
            "slow wind": "slow wind"}


def hourly_events(start, end, index):
    """CME hours, possible-CME hours (bool Series on `index`) and stream arrival times.

    Detected from the cached 1-minute ACE plasma and field over [start, end).
    """
    minutes = load_conditions(start, end)
    hourly = minutes.resample("h").mean().reindex(
        pd.date_range(start, end, freq="h", inclusive="left", name="Time"))
    events = detect_icmes(hourly, detect_shocks(minutes))
    cme = cme_mask(hourly.index, events)
    possible = cme_mask(hourly.index, events, ("possible ICME",))
    arrivals = detect_stream_arrivals(hourly["speed"], cme | possible)
    return (cme.reindex(index, fill_value=False), possible.reindex(index, fill_value=False),
            arrivals)


def label_runs(positions, values, cme, possible, arrivals, horizon=72):
    """classify_runs, vectorised: one regime per run at index `positions` of `cme`.

    `values` is the hourly speed on the same index as `cme` and `possible` (bool
    Series); a run at position p targets positions p+1 .. p+horizon.
    """
    positions = np.asarray(positions)
    n = len(cme)

    def any_in_window(mask):
        c = np.concatenate([[0], np.cumsum(np.asarray(mask, dtype=int))])
        lo = np.clip(positions + 1, 0, n)
        hi = np.clip(positions + horizon + 1, 0, n)
        return c[hi] - c[lo] > 0

    at = cme.index.get_indexer(pd.DatetimeIndex(arrivals))
    arrival_hours = np.zeros(n, dtype=bool)
    arrival_hours[at[at >= 0]] = True
    values = np.asarray(values, dtype=float)
    idx = positions[:, None] + np.arange(1, horizon + 1)
    actual = np.full(idx.shape, np.nan)
    inside = idx < len(values)
    actual[inside] = values[idx[inside]]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, peak = np.nanmean(actual, axis=1), np.nanmax(actual, axis=1)
    labels = np.select(
        [any_in_window(cme), any_in_window(possible), any_in_window(arrival_hours),
         mean >= 550, peak < 450],
        ["CME", "possible CME", "stream arrival", "fast stream", "slow wind"],
        "intermediate")
    return pd.Categorical(labels, categories=REGIMES)
