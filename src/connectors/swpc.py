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
                "Time": pd.to_datetime(raw["time_tag"], utc=True, format="ISO8601"),
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


# ---------------------------------------------------------------------------
# Multi-field real-time feeds, solar flares and F10.7 (regressor inputs)

RTSW_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"

# Field name in our frames -> field name in the feed, matching connectors.ace_archive.
REALTIME = {
    "swepam": (RTSW_WIND_URL, {"density": "proton_density", "speed": "proton_speed",
                               "temperature": "proton_temperature"}),
    "mag": (RTSW_MAG_URL, {"bx": "bx_gsm", "by": "by_gsm", "bz": "bz_gsm", "bt": "bt",
                           "lat": "theta_gsm", "lon": "phi_gsm"}),
}


def _empty(fields):
    index = pd.DatetimeIndex([], tz="UTC", name="Time")
    return pd.DataFrame({f: pd.Series(dtype=float, index=index) for f in fields})


class SWPCRealtime:
    """Last ~24 h of 1-minute plasma ("swepam") or field ("mag") from the RTSW feed.

    Same field names as connectors.ace_archive, so it can fill in the minutes the
    daily archive doesn't have yet. ACE rows are used when there are any, otherwise
    the mean over the other spacecraft. Rows with a bad quality flag are dropped.
    Never raises: a failure gives an empty frame.
    """

    def __init__(self, product, timeout=30):
        if product not in REALTIME:
            raise ValueError(f"unknown real-time product {product!r}")
        self.product = product
        self.url, self.fields = REALTIME[product]
        self.timeout = timeout

    def minutes(self, start, end):
        try:
            records = http_get_json(self.url, timeout=self.timeout)
            return self.parse(records, start, end)
        except Exception as e:
            log(f"SWPC real-time {self.product} failed: {e!r}")
            return _empty(self.fields)

    def parse(self, records, start, end):
        if not records:
            return _empty(self.fields)
        raw = pd.DataFrame.from_records(records)
        quality = pd.to_numeric(raw.get("overall_quality"), errors="coerce").fillna(0)
        raw = raw[quality == 0]
        times = pd.to_datetime(raw["time_tag"], utc=True, format="ISO8601")
        df = pd.DataFrame({ours: pd.to_numeric(raw[theirs], errors="coerce").to_numpy()
                           for ours, theirs in self.fields.items()},
                          index=pd.DatetimeIndex(times, name="Time"))
        source = raw["source"].astype(str).str.upper().to_numpy()
        use = df[source == "ACE"] if (source == "ACE").any() else df
        out = use.groupby(level=0).mean().sort_index()
        return out[(out.index >= start) & (out.index <= end)]


FLARE_COLUMNS = ["peak_time", "peak_flux", "flare_class"]


def empty_flares():
    return pd.DataFrame({"peak_time": pd.Series(dtype="datetime64[ns, UTC]"),
                         "peak_flux": pd.Series(dtype=float),
                         "flare_class": pd.Series(dtype=object)})


def parse_flares(records):
    """GOES flare list (SWPC JSON) as peak_time | peak_flux (W/m^2, 0.1-0.8 nm) | flare_class."""
    if not records:
        return empty_flares()
    raw = pd.DataFrame.from_records(records)
    df = pd.DataFrame({
        "peak_time": pd.to_datetime(raw["max_time"], utc=True, errors="coerce", format="ISO8601"),
        "peak_flux": pd.to_numeric(raw["max_xrlong"], errors="coerce"),
        "flare_class": raw["max_class"].astype(str),
    })
    df = df.dropna(subset=["peak_time", "peak_flux"])
    return df.drop_duplicates("peak_time").sort_values("peak_time").reset_index(drop=True)


class SWPCFlareConnector:
    """GOES X-ray flares of the last 7 days. `flares()` gives None if the feed failed,
    so "no flares" (an empty frame) and "unknown" can be told apart."""

    def __init__(self, url=FLARES_URL, timeout=30):
        self.url = url
        self.timeout = timeout

    def flares(self):
        try:
            return parse_flares(http_get_json(self.url, timeout=self.timeout))
        except Exception as e:
            log(f"SWPC flares failed: {e!r}")
            return None


def parse_f107(records):
    """Daily F10.7 (sfu) by UTC day: the local-noon (20 UT) reading, else the day's mean."""
    if not records:
        return pd.Series(dtype=float, name="f107")
    raw = pd.DataFrame.from_records(records)
    times = pd.to_datetime(raw["time_tag"], utc=True, errors="coerce", format="ISO8601")
    df = pd.DataFrame({"day": times.dt.floor("D"),
                       "noon": raw["reporting_schedule"].astype(str).str.lower() == "noon",
                       "flux": pd.to_numeric(raw["flux"], errors="coerce")}).dropna()
    df = df[(df["flux"] > 0) & (df["flux"] < 1000)]
    noon = df[df["noon"]].groupby("day")["flux"].mean()
    mean = df.groupby("day")["flux"].mean()
    out = noon.reindex(mean.index).fillna(mean)
    out.index = pd.DatetimeIndex(out.index, name="day")
    return out.rename("f107").sort_index()


class SWPCF107Connector:
    """Daily 10.7 cm solar radio flux for about the last 40 days; None if the feed failed."""

    def __init__(self, url=F107_URL, timeout=30):
        self.url = url
        self.timeout = timeout

    def daily(self):
        try:
            return parse_f107(http_get_json(self.url, timeout=self.timeout))
        except Exception as e:
            log(f"SWPC F10.7 failed: {e!r}")
            return None
