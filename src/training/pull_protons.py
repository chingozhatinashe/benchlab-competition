"""GOES >=10 MeV integral proton flux history, hourly -> src/data/features/protons_hourly.csv.

    python -m training.pull_protons            # resumable; ~2,200 daily files, ~30 min
    python -m training.pull_protons --check    # compare the last days with SWPC's live feed

Sources (the live feed, services.swpc.noaa.gov/json/goes/primary/integral-protons-*.json,
keeps 7 days):
    to 2020-03   OMNI hourly PR-FLX_10 (GOES-13/15 >10 MeV, pfu); OMNI stops there
    2020-11 on   NCEI GOES-R SGPS L2 5-minute averages (sci_sgps-l2-avg5m), GOES-16 to
                 2025-04-06, GOES-18 after (the other one if a day is missing). SGPS has no
                 >10 MeV channel: the flux is the sum over the differential channels of flux
                 x the part of the channel above 10 MeV (protons/(cm2 sr keV s) x keV), the
                 mean of the two sensors. The >500 MeV channel is left out: with it the
                 quiet-time value is about twice SWPC's.
The gap 2020-03-04 .. 2020-10-31 (solar minimum) stays missing.

Each day is reduced to 24 hourly means (cache: src/data/features/protons_raw/, one CSV per
day; the netCDF files are not kept). Hour h holds the 5-minute values from h to h + 1 h.
"""

import argparse
import io
import json
import re
import socket
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from training.pull_features import HISTORY_START, OUT, TIME_FORMAT, log

NCEI = ("https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
        "{sat}/l2/data/sgps-l2-avg5m/{y:%Y}/{y:%m}/")
LIVE = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json"
OMNI_END = pd.Timestamp("2020-03-05", tz="UTC")
SGPS_START = pd.Timestamp("2020-11-01", tz="UTC")
G18_FROM = pd.Timestamp("2025-04-07", tz="UTC")
THRESHOLD_KEV = 10_000.0
RAW = OUT / "protons_raw"
OUTFILE = OUT / "protons_hourly.csv"
EPOCH = pd.Timestamp("2000-01-01T12:00:00Z")
WORKERS = 4


def _ipv4_only():
    """NCEI's IPv6 address stalls connections from the dev machine; resolve IPv4 only."""
    if getattr(socket.getaddrinfo, "ipv4_only", False):
        return
    original = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, *args, **kwargs):
        return original(host, port, socket.AF_INET, *args, **kwargs)

    getaddrinfo.ipv4_only = True
    socket.getaddrinfo = getaddrinfo


def _get(url, timeout=60, tries=3):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt == tries - 1:
                raise


def integral_10mev(diff, lower, upper):
    """>=10 MeV flux (pfu) from differential fluxes (time, sensor, channel) in /keV."""
    width = np.clip(upper - np.maximum(lower, THRESHOLD_KEV), 0, None)  # (sensor, channel)
    flux = np.where(diff > -1e30, diff, np.nan)
    per_sensor = np.nansum(flux * width[None], axis=2)
    per_sensor[np.isnan(flux).all(axis=2)] = np.nan
    n = np.isfinite(per_sensor).sum(axis=1)
    total = np.nansum(per_sensor, axis=1)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def reduce_file(content):
    """Hourly >=10 MeV means of one SGPS daily file."""
    import h5py

    with h5py.File(io.BytesIO(content), "r") as f:
        diff = f["AvgDiffProtonFlux"][:].astype(float)
        lower = f["DiffProtonLowerEnergy"][:].astype(float)
        upper = f["DiffProtonUpperEnergy"][:].astype(float)
        secs = f["time" if "time" in f else "L2_SciData_TimeStamp"][:]  # v3 | v1 files
    t = EPOCH + pd.to_timedelta(secs, unit="s")
    flux = pd.Series(integral_10mev(diff, lower, upper), index=t)
    return flux.resample("1h").mean()


_LISTINGS = {}


def day_url(day):
    sats = ("goes18", "goes16") if day >= G18_FROM else ("goes16", "goes18")
    for sat in sats:
        base = NCEI.format(sat=sat, y=day)
        if base not in _LISTINGS:
            try:
                _LISTINGS[base] = _get(base, timeout=60).decode()
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                _LISTINGS[base] = ""  # no such month for this satellite
        names = sorted(set(re.findall(rf"sci_sgps-l2-avg5m_g\d+_d{day:%Y%m%d}_v[\d-]+\.nc",
                                      _LISTINGS[base])))
        if names:
            return base + names[-1]
    return None


def pull_day(day):
    path = RAW / f"{day:%Y%m%d}.csv"
    if path.exists():
        return day, "cached"
    url = day_url(day)
    if url is None:
        path.write_text("time,flux\n", encoding="utf-8")  # known missing
        return day, "missing"
    hourly = reduce_file(_get(url))
    hourly.rename("flux").rename_axis("time").to_csv(path, date_format=TIME_FORMAT)
    return day, "ok"


def pull_sgps(start=SGPS_START, end=None):
    RAW.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
    days = pd.date_range(start, end, freq="D", tz="UTC")
    for month in sorted({d.replace(day=1) for d in days}):  # listings first, serially
        day_url(month)
    counts, failed = {}, []
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(pull_day, d): d for d in days}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                _, status = fut.result()
            except Exception as e:  # noqa: BLE001 - retried on the next run
                status = "failed"
                failed.append(f"{futures[fut]:%Y-%m-%d}: {type(e).__name__} {e}")
                if len(failed) <= 3:
                    log(f"  failed {failed[-1]}")
            counts[status] = counts.get(status, 0) + 1
            if i % 100 == 0:
                log(f"  SGPS {i}/{len(days)} days: {counts}")
    log(f"SGPS: {counts}" + (f"; failed (run again): {failed[:5]}" if failed else ""))


def pull_omni(start=HISTORY_START, end=OMNI_END):
    from training.pull_omni import pull_variable

    o = pull_variable("PR-FLX_10", start, end)
    ok = o[o["status"] == "ok"]
    return pd.Series(ok["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(ok["Time"]),
                     name="flux")


def combine():
    parts = [pd.read_csv(p, parse_dates=["time"]) for p in sorted(RAW.glob("*.csv"))]
    sgps = pd.concat([p for p in parts if len(p)], ignore_index=True)
    sgps = pd.Series(sgps["flux"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(sgps["time"], tz="UTC") if sgps["time"].dt.tz is None
                     else pd.DatetimeIndex(sgps["time"]), name="flux")
    flux = pd.concat([pull_omni(), sgps]).sort_index()
    flux = flux[~flux.index.duplicated(keep="last")]
    flux.rename_axis("time").to_csv(OUTFILE, date_format=TIME_FORMAT)
    log(f"protons: {flux.notna().sum()} hours, {flux.index.min():%Y-%m-%d} .. "
        f"{flux.index.max():%Y-%m-%d} -> {OUTFILE}")
    return flux


def load():
    df = pd.read_csv(OUTFILE, parse_dates=["time"])
    return pd.Series(df["flux"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["time"]),
                     name="flux")


def check():
    """The pulled history against SWPC's live >=10 MeV feed over their overlap."""
    live = pd.DataFrame([r for r in json.loads(_get(LIVE)) if r["energy"] == ">=10 MeV"])
    live = pd.Series(live["flux"].to_numpy(dtype=float),
                     index=pd.to_datetime(live["time_tag"], utc=True)).resample("1h").mean()
    both = pd.concat([load().rename("pulled"), live.rename("live")], axis=1).dropna()
    ratio = both["pulled"] / both["live"]
    log(f"overlap {len(both)} hours: median pulled/live {ratio.median():.3f} "
        f"(5-95%: {ratio.quantile(.05):.3f} .. {ratio.quantile(.95):.3f}); "
        f"live max {both['live'].max():.2f} pfu, pulled max {both['pulled'].max():.2f}")
    return both


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    _ipv4_only()
    if args.check:
        check()
        return
    pull_sgps()
    combine()


if __name__ == "__main__":
    sys.exit(main())
