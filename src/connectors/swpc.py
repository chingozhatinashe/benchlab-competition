"""NOAA SWPC real-time solar wind (RTSW) 1-minute plasma data."""

import numpy as np
import pandas as pd

from connectors.base import (
    STATUS_FLAGGED,
    STATUS_MISSING,
    STATUS_OK,
    Connector,
    http_get_json,
    log,
)

RTSW_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"

# Physical range used to flag obviously bad speeds (km/s).
VALID_MIN, VALID_MAX = 0.0, 1200.0


class SWPCConnector(Connector):
    """Proton speed (km/s) from every spacecraft in the SWPC RTSW feed.

    The feed only covers roughly the last 24 hours, so older ranges come back
    empty. Each spacecraft (ACE, IMAP, SOLAR1, ...) has its own rows, named in
    the extra `source` column.
    """

    extra_columns = ("source",)

    def __init__(self, url=RTSW_WIND_URL, timeout=30):
        self.url = url
        self.timeout = timeout

    def _fetch(self, start, end):
        records = http_get_json(self.url, timeout=self.timeout)
        if not records:
            return None
        raw = pd.DataFrame.from_records(records)

        df = pd.DataFrame(
            {
                "Time": pd.to_datetime(raw["time_tag"], utc=True),
                "value": pd.to_numeric(raw["proton_speed"], errors="coerce"),
                "source": raw["source"].astype(str),
            }
        )
        quality = pd.to_numeric(raw.get("overall_quality"), errors="coerce").fillna(0)

        missing = df["value"].isna()
        flagged = ~missing & (
            (quality != 0) | (df["value"] < VALID_MIN) | (df["value"] > VALID_MAX)
        )
        df["status"] = np.select([missing, flagged], [STATUS_MISSING, STATUS_FLAGGED], STATUS_OK)

        if df["Time"].min() > start:
            log(f"SWPC feed starts at {df['Time'].min()}, after requested start {start}")
        return df
