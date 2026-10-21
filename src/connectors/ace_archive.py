"""SWPC real-time ACE SWEPAM 1-minute lists, archived by day on the NASA SOHO mirror.

This is the source of the training data (training.pull_ace_rtsw), so a live run
reads the same series the model was fitted on. Today's file is updated through
the day, a few minutes behind real time.
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait

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
FILE_PATTERN = "{day:%Y%m%d}_ace_swepam_1m.txt"
FILL_SPEED = -9999.9
FILE_COLUMNS = ["year", "month", "day", "hhmm", "mjd", "sod", "status", "density", "speed", "temp"]


def parse_minutes(text):
    """1-minute speeds from one file, NaN where the status is bad or the value is fill."""
    rows = [line.split() for line in text.splitlines() if line[:1].isdigit()]
    rows = [r for r in rows if len(r) == len(FILE_COLUMNS)]
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC", name="Time"))
    df = pd.DataFrame(rows, columns=FILE_COLUMNS)
    stamp = df["year"] + df["month"] + df["day"] + df["hhmm"]
    times = pd.to_datetime(stamp, format="%Y%m%d%H%M", utc=True)
    speed = df["speed"].astype(float)
    valid = (df["status"].astype(int) == 0) & (speed != FILL_SPEED)
    return pd.Series(
        np.where(valid, speed, np.nan),
        index=pd.DatetimeIndex(times, name="Time"),
        name="speed",
    )


def day_url(day):
    return f"{BASE_URL}/{FILE_PATTERN.format(day=day)}"


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
        today = pd.Timestamp.now(tz="UTC").floor("D")
        days = pd.date_range(start.floor("D"), min(end.floor("D"), today), freq="D")
        if days.empty:
            return None
        started = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=self.workers)
        futures = {pool.submit(self.download, day): day for day in days}
        done, pending = wait(futures, timeout=self.budget)
        pool.shutdown(wait=False, cancel_futures=True)

        parts, failed = [], 0
        for future in done:
            try:
                text = future.result()
            except Exception as e:  # one bad day must not lose the others
                log(f"ACE archive {futures[future]:%Y-%m-%d}: {e!r}")
                failed += 1
                continue
            if text is not None:
                parts.append(parse_minutes(text))
        log(
            f"ACE archive: {len(parts)}/{len(days)} days in {time.monotonic() - started:.1f} s"
            f" ({len(pending)} timed out, {failed} failed)"
        )
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
