"""SDO full-disk browse images (JPEG): AIA 193 Å for coronal-hole maps, HMI magnetograms ("HMIB").

Images are posted about 15 minutes after they are taken, roughly every 10 minutes,
in one directory per UTC day:
    https://sdo.gsfc.nasa.gov/assets/img/browse/YYYY/MM/DD/YYYYMMDD_HHMMSS_512_0193.jpg
The file name carries the capture time. The `?P=` filter keeps a day's listing to
about 20 KB instead of 2 MB.

`latest(cutoffs)` picks, for each cut-off time, the newest image taken at or before
it (and no more than `max_age` earlier) and downloads it. The caller sets the
cut-off to t0 minus the posting latency, so an image posted after t0 is never used.
"""

import re
from concurrent.futures import ThreadPoolExecutor, wait

import pandas as pd

from connectors.base import HTTPNotFound, http_get_bytes, log

BASE_URL = "https://sdo.gsfc.nasa.gov/assets/img/browse"
SIZE = 512
CHANNEL = "0193"
MAX_AGE = pd.Timedelta(hours=24)
NAME = re.compile(r"(\d{8}_\d{6})_(\d+)_([0-9A-Z]+)\.jpg")  # channel "0193", "HMIB", ...


def day_listing_url(day, size=SIZE, channel=CHANNEL):
    return f"{BASE_URL}/{day:%Y/%m/%d}/?P=*_{size}_{channel}.jpg"


def image_url(taken, size=SIZE, channel=CHANNEL):
    return f"{BASE_URL}/{taken:%Y/%m/%d}/{taken:%Y%m%d_%H%M%S}_{size}_{channel}.jpg"


def parse_listing(html, size=SIZE, channel=CHANNEL):
    """Capture times (UTC) of the images in one day's directory listing, sorted."""
    times = {
        m.group(1)
        for m in NAME.finditer(html)
        if int(m.group(2)) == size and m.group(3) == channel
    }
    stamps = pd.to_datetime(sorted(times), format="%Y%m%d_%H%M%S", utc=True)
    return pd.DatetimeIndex(stamps, name="taken")


def candidates(times, cutoff, max_age, n):
    """Up to n times <= cutoff and >= cutoff - max_age, newest first."""
    ok = times[(times <= cutoff) & (times >= cutoff - max_age)]
    return list(ok[::-1][:n])


class SDOConnector:
    """SDO browse images (one channel) by cut-off time. Never raises: failures give None."""

    def __init__(self, size=SIZE, channel=CHANNEL, timeout=20, retries=2, workers=6,
                 budget=120.0):
        self.size = size
        self.channel = channel
        self.timeout = timeout
        self.retries = retries
        self.workers = workers
        self.budget = budget

    def _get(self, url):
        return http_get_bytes(url, timeout=self.timeout, retries=self.retries)

    def listing(self, day):
        try:
            html = self._get(day_listing_url(day, self.size, self.channel)).decode(
                "utf-8", errors="replace")
        except HTTPNotFound:
            return pd.DatetimeIndex([], tz="UTC", name="taken")
        return parse_listing(html, self.size, self.channel)

    def image(self, taken):
        return self._get(image_url(taken, self.size, self.channel))

    def latest(self, cutoffs, max_age=MAX_AGE, accept=None, tries=12):
        """{cutoff: (taken, jpeg bytes)} for each cut-off; (None, None) where none is usable.

        `accept(jpeg, taken)` can reject a frame (blank or corrupt ones happen); then
        the next older image is tried, up to `tries` per cut-off.
        """
        cutoffs = list(cutoffs)
        result = {c: (None, None) for c in cutoffs}
        if not cutoffs:
            return result
        days = sorted({d for c in cutoffs for d in (c.floor("D"), (c - max_age).floor("D"))})
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            listings = self._run(pool, {day: (self.listing, day) for day in days}, "listing")
            known = [t for times in listings.values() if times is not None for t in times]
            times = pd.DatetimeIndex(sorted(set(known)), name="taken")
            if times.tz is None:  # no listing answered: an empty, naive index
                times = times.tz_localize("UTC")
            queue = {c: candidates(times, c, max_age, tries) for c in cutoffs}
            images = {}
            for round_ in range(tries):
                wanted = {q[round_] for c, q in queue.items()
                          if result[c][0] is None and len(q) > round_} - set(images)
                if not wanted and round_ > 0:
                    break
                images.update(self._run(pool, {t: (self.image, t) for t in wanted}, "image"))
                for c, q in queue.items():
                    if result[c][0] is not None or len(q) <= round_:
                        continue
                    t = q[round_]
                    jpeg = images.get(t)
                    if jpeg is not None and (accept is None or accept(jpeg, t)):
                        result[c] = (t, jpeg)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        log(f"SDO {self.channel}: {sum(v[0] is not None for v in result.values())}"
            f"/{len(cutoffs)} images")
        return result

    def _run(self, pool, jobs, label):
        futures = {pool.submit(fn, arg): key for key, (fn, arg) in jobs.items()}
        done, pending = wait(futures, timeout=self.budget)
        out = {}
        for future in done:
            try:
                out[futures[future]] = future.result()
            except Exception as e:
                log(f"SDO {label} {futures[future]}: {e!r}")
        if pending:
            log(f"SDO {label}: {len(pending)} timed out")
        return out
