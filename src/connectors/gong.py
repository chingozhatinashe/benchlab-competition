"""GONG zero-point-corrected synoptic magnetograms ("mrzqs"), the input to the PFSS model.

NSO posts one full-Sun synoptic map of the photospheric radial field about every hour, in
one directory per UTC day of observation:
    https://gong2.nso.edu/oQR/zqs/YYYYMM/mrzqsYYMMDD/mrzqsYYMMDDtHHMMcCCCC_LLL.fits.gz
The name carries the observation time, the Carrington rotation and the central longitude.
Files are posted 30-45 min later, and the archive keeps each file's original upload time
(checked 2016-10, 2026-07, 2026-11). The listing shows it in MST (UTC-7, all year).

Availability is the upload time itself: a map is usable for a forecast at t0 only if it was
modified by t0 - MARGIN. History and live runs read the same listings and apply the same rule.
"""

import re

import pandas as pd

from connectors.base import HTTPNotFound, http_get_bytes, log

BASE_URL = "https://gong2.nso.edu/oQR/zqs"
LISTING_UTC_OFFSET = pd.Timedelta(hours=7)  # listing times are MST
MARGIN = pd.Timedelta(minutes=10)  # same margin as the in-situ data
MAX_AGE = pd.Timedelta(hours=72)  # real-time outages run for days; a synoptic map ages slowly
ROW = re.compile(
    r'href="(mrzqs(\d{6})t(\d{4})c(\d{4})_(\d{3})\.fits\.gz)".*?'
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})"
)
COLUMNS = ["name", "observed", "modified", "rotation", "longitude"]


def day_url(day):
    return f"{BASE_URL}/{day:%Y%m}/mrzqs{day:%y%m%d}/"


def map_url(name):
    day = pd.to_datetime(name[5:11], format="%y%m%d")
    return day_url(day) + name


def parse_listing(html):
    """One row per map in a day's listing: name, observed and modified (UTC), rotation, lon."""
    rows = []
    for m in ROW.finditer(html):
        name, ymd, hm, rot, lon, modified = m.groups()
        rows.append((name, ymd + hm, modified, int(rot), int(lon)))
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("name")
    df["observed"] = pd.to_datetime(df["observed"], format="%y%m%d%H%M", utc=True)
    df["modified"] = (pd.to_datetime(df["modified"], format="%Y-%m-%d %H:%M", utc=True)
                      + LISTING_UTC_OFFSET)
    return df.sort_values("observed").reset_index(drop=True)


def usable(listing, t0, margin=MARGIN, max_age=MAX_AGE):
    """Rows posted by t0 - margin and observed at most max_age before it, newest first.

    The listing's minute resolution is rounded up, so a map modified at hh:mm:59 is
    never counted as posted at hh:mm.
    """
    cut = pd.Timestamp(t0) - margin
    posted = listing["modified"] + pd.Timedelta(minutes=1)
    ok = listing[(posted <= cut) & (listing["observed"] >= cut - max_age)]
    return ok.sort_values("observed", ascending=False)


class GONGConnector:
    """Newest GONG synoptic map available at t0. Never raises: failures give None."""

    def __init__(self, timeout=20, retries=2, tries=3):
        self.timeout = timeout
        self.retries = retries
        self.tries = tries

    def _get(self, url):
        return http_get_bytes(url, timeout=self.timeout, retries=self.retries)

    def listing(self, day):
        try:
            html = self._get(day_url(day)).decode("utf-8", errors="replace")
        except HTTPNotFound:
            return pd.DataFrame(columns=COLUMNS)
        return parse_listing(html)

    def download(self, name):
        return self._get(map_url(name))

    def latest(self, t0, accept=None, margin=MARGIN, max_age=MAX_AGE):
        """(row, fits.gz bytes) of the newest usable map, or (None, None).

        `accept(data, row)` may reject a map (e.g. one that fails to parse); the next older
        one is tried, up to `tries` maps.
        """
        t0 = pd.Timestamp(t0)
        try:
            days = pd.date_range((t0 - margin - max_age).floor("D"), t0.floor("D"), freq="D")
            frames = [f for f in (self.listing(d) for d in days) if not f.empty]
            if not frames:
                return None, None
            rows = usable(pd.concat(frames, ignore_index=True), t0, margin, max_age)
            for _, row in rows.head(self.tries).iterrows():
                try:
                    data = self.download(row["name"])
                except Exception as e:  # try the next older map
                    log(f"GONG {row['name']} failed: {e!r}")
                    continue
                if accept is None or accept(data, row):
                    return row, data
        except Exception as e:
            log(f"GONG listing failed for {t0}: {e!r}")
        return None, None
