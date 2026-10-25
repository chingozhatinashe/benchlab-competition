"""SOHO/LASCO C2 and C3 browse images (512-pixel JPEG), for the coronagraph regressors.

The quick-look images are posted about 15 minutes after they are taken, in one
directory per instrument and UTC day:
    https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/YYYY/c2/YYYYMMDD/YYYYMMDD_HHMM_c2_512.jpg
The file name carries the capture time. The same directories hold the history that
training.pull_lasco_images reduces, so live runs see the same product.

`cells(start, end)` lists the days, keeps the images taken in [start, end] at the
cadence of feature_engineering.coronagraph.select_times, downloads them in parallel
and reduces each to its sector-ring cells. The caller sets `end` to t0 - LATENCY.
"""

import re
from concurrent.futures import ThreadPoolExecutor, wait

import pandas as pd

from connectors.base import HTTPNotFound, http_get_bytes, log
from feature_engineering import coronagraph

BASE_URL = "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed"


def day_url(instrument, day):
    return f"{BASE_URL}/{day:%Y}/{instrument}/{day:%Y%m%d}/"


def image_url(instrument, taken):
    return f"{day_url(instrument, taken)}{taken:%Y%m%d_%H%M}_{instrument}_512.jpg"


def parse_listing(html, instrument, day):
    """Capture times (UTC) of the 512-pixel images in one day's listing, sorted."""
    d = f"{day:%Y%m%d}"
    stamps = sorted(set(re.findall(rf"{d}_(\d{{4}})_{instrument}_512\.jpg", html)))
    return pd.DatetimeIndex(pd.to_datetime([f"{d}{s}" for s in stamps], format="%Y%m%d%H%M",
                                           utc=True), name="taken")


class LASCOConnector:
    """C2/C3 cell tables over a time range. Never raises: failures give empty tables."""

    def __init__(self, timeout=20, retries=2, workers=8, budget=120.0):
        self.timeout = timeout
        self.retries = retries
        self.workers = workers
        self.budget = budget

    def _get(self, url):
        return http_get_bytes(url, timeout=self.timeout, retries=self.retries)

    def listing(self, instrument, day):
        try:
            html = self._get(day_url(instrument, day)).decode("utf-8", errors="replace")
        except HTTPNotFound:
            return pd.DatetimeIndex([], tz="UTC", name="taken")
        return parse_listing(html, instrument, day)

    def _cells(self, instrument, taken):
        return coronagraph.cells(self._get(image_url(instrument, taken)), instrument)

    def cells(self, start, end):
        """{instrument: cell table indexed by capture time} for images taken in [start, end]."""
        days = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
        out = {i: pd.DataFrame(columns=coronagraph.cell_columns(), dtype=float)
               for i in coronagraph.INSTRUMENTS}
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            jobs = {(i, d): pool.submit(self.listing, i, d)
                    for i in coronagraph.INSTRUMENTS for d in days}
            done, _ = wait(jobs.values(), timeout=self.budget)
            wanted = []
            for (instrument, _day), fut in jobs.items():
                if fut not in done or fut.exception() is not None:
                    continue
                times = fut.result()
                wanted += [(instrument, t) for t in times[(times >= start) & (times <= end)]]
            by_inst = {}
            for instrument, t in wanted:
                by_inst.setdefault(instrument, []).append(t)
            wanted = [(i, t) for i, ts in by_inst.items()
                      for t in coronagraph.select_times(ts, i)]
            futures = {pool.submit(self._cells, i, t): (i, t) for i, t in wanted}
            done, pending = wait(futures, timeout=self.budget)
            rows = {i: {} for i in coronagraph.INSTRUMENTS}
            for fut in done:
                instrument, t = futures[fut]
                try:
                    v = fut.result()
                except Exception as e:
                    log(f"LASCO {instrument} {t:%Y-%m-%d %H:%M}: {e!r}")
                    continue
                if v is not None:
                    rows[instrument][t] = v.ravel()
            for instrument, r in rows.items():
                if r:
                    out[instrument] = pd.DataFrame.from_dict(
                        r, orient="index", columns=coronagraph.cell_columns()).sort_index()
            if pending:
                log(f"LASCO: {len(pending)} images timed out")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        log("LASCO: " + ", ".join(f"{i} {len(t)}/{sum(1 for j, _ in wanted if j == i)} images"
                                   for i, t in out.items()))
        return out
