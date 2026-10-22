"""Fetch every regressor source for a forecast at t0, in parallel, within a time budget.

Each source is one job; a job that fails or runs out of time leaves its input as
None, and the features it feeds become NaN (filled later by
feature_engineering.impute). Nothing here raises into the forecast.

Availability is enforced at the source: in-situ minutes before t0 - LATENCY,
AIA images taken before t0 - CH_LATENCY - d days, flares by peak time and F10.7
by day (both inside feature_engineering.solar_activity).
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait

import pandas as pd

from connectors.ace_archive import ACEDailyArchive
from connectors.base import bytes_downloaded, to_utc
from connectors.sdo import SDOConnector
from connectors.swpc import SWPCF107Connector, SWPCFlareConnector, SWPCRealtime
from feature_engineering import build, coronal_holes, insitu

BUDGET_SECONDS = 300.0


def log(msg):
    print(f"[fetch_inputs] {msg}", file=sys.stderr, flush=True)


def _accept(jpeg, taken):
    return coronal_holes.usable(coronal_holes.grayscale(jpeg), taken)


def default_sources(budget=BUDGET_SECONDS):
    archive_budget = max(budget - 60.0, 30.0)
    return {
        "swepam": ACEDailyArchive("swepam", budget=archive_budget),
        "mag": ACEDailyArchive("mag", budget=archive_budget),
        "epam": ACEDailyArchive("epam", budget=archive_budget),
        "swepam_rt": SWPCRealtime("swepam"),
        "mag_rt": SWPCRealtime("mag"),
        "sdo": SDOConnector(budget=min(120.0, archive_budget)),
        "flares": SWPCFlareConnector(),
        "f107": SWPCF107Connector(),
    }


def _before(df, cutoff):
    return None if df is None else df[df.index < cutoff]


def fetch_inputs(t0, sources=None, budget=BUDGET_SECONDS, ch_lag_hours=None):
    """(build.Inputs, report) for a forecast at t0, using data available by t0.

    `ch_lag_hours` lists the AIA image lags (hours before t0 - CH_LATENCY) to fetch;
    by default the daily lags of the fixed regressors (build.ch_cutoffs).
    """
    t0 = to_utc(t0)
    if ch_lag_hours is None:
        ch_cutoffs = build.ch_cutoffs(t0)
    else:
        ch_cutoffs = [t0 - build.CH_LATENCY - pd.Timedelta(hours=h) for h in ch_lag_hours]
    started, bytes_before = time.monotonic(), bytes_downloaded()
    sources = sources or default_sources(budget)
    cutoff = t0 - insitu.LATENCY
    start = t0 - build.HISTORY - pd.Timedelta(hours=1)
    jobs = {
        "swepam": lambda: sources["swepam"].minutes(start, cutoff),
        "mag": lambda: sources["mag"].minutes(start, cutoff),
        "epam": lambda: sources["epam"].minutes(start, cutoff),
        "swepam_rt": lambda: sources["swepam_rt"].minutes(start, cutoff),
        "mag_rt": lambda: sources["mag_rt"].minutes(start, cutoff),
        "sdo": lambda: sources["sdo"].latest(ch_cutoffs, accept=_accept),
        "flares": lambda: sources["flares"].flares(),
        "f107": lambda: sources["f107"].daily(),
    }
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    futures = {pool.submit(fn): name for name, fn in jobs.items()}
    done, pending = wait(futures, timeout=budget)
    pool.shutdown(wait=False, cancel_futures=True)
    got, report = {}, {"t0": f"{t0:%Y-%m-%dT%H:%MZ}", "sources": {}}
    for future, name in futures.items():
        if future in pending:
            status = "timed out"
            log(f"{name}: timed out after {budget:.0f} s")
        else:
            try:
                got[name] = future.result()
                status = "ok"
            except Exception as e:
                status = f"failed: {e!r}"
        report["sources"][name] = status

    plasma = _before(insitu.combine(got.get("swepam"), got.get("swepam_rt")), cutoff)
    mag = _before(insitu.combine(got.get("mag"), got.get("mag_rt")), cutoff)
    epam = _before(got.get("epam"), cutoff)
    minutes = insitu.minute_frame(plasma, mag, epam)
    hourly = insitu.hourly_frame(minutes, t0 - build.HISTORY, t0)
    inputs = build.Inputs(
        hourly=hourly,
        shocks=insitu.detect_shocks(minutes),
        coronal_holes=build.coronal_hole_frame(got["sdo"]) if "sdo" in got else None,
        flares=got.get("flares"),
        f107=got.get("f107"),
    )
    report["rows"] = {
        "plasma_minutes": 0 if plasma is None else int(plasma["speed"].notna().sum()),
        "mag_minutes": 0 if mag is None else int(mag["bt"].notna().sum()),
        "epam_rows": 0 if epam is None else int(epam["p47"].notna().sum()),
        "hours_with_plasma": int(hourly["density"].notna().sum()),
        "hours_with_field": int(hourly["bt"].notna().sum()),
        "aia_images": 0 if inputs.coronal_holes is None
        else int(inputs.coronal_holes["ch_cm"].notna().sum()),
        "flares": None if inputs.flares is None else len(inputs.flares),
        "f107_days": None if inputs.f107 is None else len(inputs.f107),
    }
    report["seconds"] = round(time.monotonic() - started, 1)
    report["megabytes"] = round((bytes_downloaded() - bytes_before) / 1e6, 1)
    log(f"inputs in {report['seconds']} s, {report['megabytes']} MB: {report['sources']}")
    return inputs, report
