"""SWPC real-time ACE lists (SWEPAM, MAG, EPAM), archived by day on the NASA SOHO mirror.

This is the source of the training data (training.pull_ace_rtsw, training.pull_features),
so a live run reads the same series the models were fitted on. Today's file is
updated through the day, a few minutes behind real time.

Products (one file per UTC day, YYYYMMDD_ace_<product>_<cadence>.txt):
    swepam  1-minute proton density (cm^-3), speed (km/s), temperature (K)
    mag     1-minute field Bx, By, Bz, Bt (nT, GSM) and its latitude/longitude (deg)
    epam    5-minute differential electron and proton fluxes (particles/cm2-s-sr-MeV)
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from connectors.base import (
    STATUS_MISSING,
    STATUS_OK,
    Connector,
    HTTPNotFound,
    http_get_bytes,
    log,
)

BASE_URL = "https://soho.nascom.nasa.gov/sdb/ace/daily"
_TIME = ["year", "month", "day", "hhmm", "mjd", "sod"]


@dataclass(frozen=True)
class Product:
    """One daily list: file name, column layout and how to read each field.

    `fields` maps a field to (status column, smallest valid value). A value is kept
    when its status column is 0 and the value is above the fill value.
    """

    name: str
    pattern: str
    columns: tuple
    fields: dict
    cadence: pd.Timedelta


PRODUCTS = {
    "swepam": Product(
        "swepam", "{day:%Y%m%d}_ace_swepam_1m.txt",
        (*_TIME, "status", "density", "speed", "temperature"),
        {"density": ("status", -9999.0), "speed": ("status", -9999.0),
         "temperature": ("status", 0.0)},
        pd.Timedelta(minutes=1),
    ),
    "mag": Product(
        "mag", "{day:%Y%m%d}_ace_mag_1m.txt",
        (*_TIME, "status", "bx", "by", "bz", "bt", "lat", "lon"),
        {f: ("status", -999.0) for f in ("bx", "by", "bz", "bt", "lat", "lon")},
        pd.Timedelta(minutes=1),
    ),
    "epam": Product(
        "epam", "{day:%Y%m%d}_ace_epam_5m.txt",
        (*_TIME, "status_e", "e38", "e175", "status_p", "p47", "p115", "p310", "p795",
         "p1060", "anisotropy"),
        {"e38": ("status_e", 0.0), "e175": ("status_e", 0.0), "p47": ("status_p", 0.0),
         "p115": ("status_p", 0.0), "p310": ("status_p", 0.0), "p795": ("status_p", 0.0),
         "p1060": ("status_p", 0.0)},
        pd.Timedelta(minutes=5),
    ),
}

# Kept for the speed-only callers (training.pull_ace_rtsw, inference).
FILE_PATTERN = PRODUCTS["swepam"].pattern
FILE_COLUMNS = list(PRODUCTS["swepam"].columns)
FILL_SPEED = -9999.9


def empty_minutes(product):
    fields = list(PRODUCTS[product].fields)
    index = pd.DatetimeIndex([], tz="UTC", name="Time")
    return pd.DataFrame({f: pd.Series(dtype=float, index=index) for f in fields})


def parse_daily(text, product):
    """One day's file as a DataFrame of its fields on a UTC minute index.

    A value is NaN where its status is not 0 or it is a fill value. Malformed lines
    are skipped, and a repeated timestamp keeps its first row.
    """
    spec = PRODUCTS[product]
    rows = [line.split() for line in text.splitlines() if line[:1].isdigit()]
    rows = [r for r in rows if len(r) == len(spec.columns)]
    if not rows:
        return empty_minutes(product)
    df = pd.DataFrame(rows, columns=spec.columns)
    stamp = df["year"] + df["month"] + df["day"] + df["hhmm"]
    times = pd.to_datetime(stamp, format="%Y%m%d%H%M", utc=True)
    out = pd.DataFrame(index=pd.DatetimeIndex(times, name="Time"))
    for field, (status, floor) in spec.fields.items():
        values = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=float)
        good = pd.to_numeric(df[status], errors="coerce").to_numpy() == 0
        out[field] = np.where(good & (values > floor), values, np.nan)
    return out[~out.index.duplicated(keep="first")]


def parse_minutes(text):
    """1-minute speeds from one SWEPAM file, NaN where the status is bad or the value is fill."""
    speed = parse_daily(text, "swepam")["speed"]
    return speed.rename("speed")


def day_url(day, product="swepam"):
    return f"{BASE_URL}/{PRODUCTS[product].pattern.format(day=day)}"


def download_days(days, download, workers, budget, label):
    """Texts of the days' files in parallel, as {day: text or None}.

    A missing file gives None, a failed download is logged and left out, and
    downloads still running after `budget` seconds are abandoned.
    """
    if len(days) == 0:
        return {}
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(download, day): day for day in days}
    done, pending = wait(futures, timeout=budget)
    pool.shutdown(wait=False, cancel_futures=True)
    texts, failed = {}, 0
    for future in done:
        try:
            texts[futures[future]] = future.result()
        except Exception as e:  # one bad day must not lose the others
            log(f"{label} {futures[future]:%Y-%m-%d}: {e!r}")
            failed += 1
    found = sum(t is not None for t in texts.values())
    log(f"{label}: {found}/{len(days)} days in {time.monotonic() - started:.1f} s"
        f" ({len(pending)} timed out, {failed} failed)")
    return texts


def archive_days(start, end):
    """The UTC days covering [start, end], up to today (later files can't exist)."""
    today = pd.Timestamp.now(tz="UTC").floor("D")
    return pd.date_range(start.floor("D"), min(end.floor("D"), today), freq="D")


class ACEDailyArchive:
    """Every field of one product for [start, end], on its native minute index.

    Days after today are skipped and a missing file (404) is a gap. With
    `cache_dir`, finished days are read from and written to disk (training only:
    a live run is stateless). Never raises: a failure gives an empty frame.
    """

    def __init__(self, product, workers=8, timeout=20, retries=2, budget=240.0, cache_dir=None):
        if product not in PRODUCTS:
            raise ValueError(f"unknown ACE product {product!r}; one of {sorted(PRODUCTS)}")
        self.product = product
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self.budget = budget
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def download(self, day):
        path = None
        if self.cache_dir is not None:
            path = self.cache_dir / PRODUCTS[self.product].pattern.format(day=day)
            if path.exists():
                return path.read_text(encoding="ascii", errors="replace")
        try:
            body = http_get_bytes(day_url(day, self.product), timeout=self.timeout,
                                  retries=self.retries)
        except HTTPNotFound:
            return None
        text = body.decode("ascii", errors="replace")
        # a day's file is final once the next day's file has been started
        if path is not None and day + pd.Timedelta(days=2) <= pd.Timestamp.now(tz="UTC"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="ascii")
        return text

    def minutes(self, start, end):
        try:
            texts = download_days(archive_days(start, end), self.download, self.workers,
                                  self.budget, f"ACE {self.product}")
            parts = [parse_daily(t, self.product) for t in texts.values() if t is not None]
        except Exception as e:
            log(f"ACE {self.product} failed for {start}..{end}: {e!r}")
            parts = []
        if not parts:
            return empty_minutes(self.product)
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df[(df.index >= start) & (df.index <= end)]


class ACEArchiveConnector(Connector):
    """ACE SWEPAM proton speed (km/s), 1-minute, as Time | status | value.

    Downloads one file per UTC day in [start, end], in parallel. Days after today
    are skipped, a missing file (404) is a gap, and downloads still running after
    `budget` seconds are abandoned, so a slow mirror can't use up the run's time.
    """

    def __init__(self, workers=8, timeout=20, retries=2, budget=240.0):
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self.budget = budget

    def download(self, day):
        try:
            body = http_get_bytes(day_url(day), timeout=self.timeout, retries=self.retries)
        except HTTPNotFound:
            return None
        return body.decode("ascii", errors="replace")

    def _fetch(self, start, end):
        texts = download_days(archive_days(start, end), self.download, self.workers,
                              self.budget, "ACE archive")
        parts = [parse_minutes(t) for t in texts.values() if t is not None]
        if not parts:
            return None
        minutes = pd.concat(parts).sort_index()
        minutes = minutes[~minutes.index.duplicated(keep="first")]
        return pd.DataFrame(
            {
                "Time": minutes.index,
                "status": np.where(minutes.notna(), STATUS_OK, STATUS_MISSING),
                "value": minutes.to_numpy(),
            }
        )
