"""Shared pieces for connectors: time parsing, HTTP, and the output schema."""

import json
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import pandas as pd

STATUS_OK = "ok"
STATUS_MISSING = "missing"  # null or fill value from the source
STATUS_FLAGGED = "flagged"  # source marks it bad, or value is outside the valid range

COLUMNS = ["Time", "status", "value"]

USER_AGENT = "benchlab-swp-connector/0.1"


def log(msg):
    print(f"[connectors] {msg}", file=sys.stderr)


def to_utc(ts):
    """Parse '20261111T060000Z', an ISO string, a datetime or a Timestamp into UTC."""
    if isinstance(ts, str) and len(ts) == 16 and ts[8] == "T" and ts.endswith("Z"):
        ts = pd.to_datetime(ts, format="%Y%m%dT%H%M%SZ", utc=True)
    out = pd.Timestamp(ts)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


class HTTPNotFound(Exception):
    """The server answered 404, which some services use for 'no data in range'."""


def http_get_json(url, headers=None, timeout=30, retries=3):
    """GET a URL and decode JSON. Retries on network and 5xx errors with backoff."""
    req_headers = {"Accept": "application/json"}
    req_headers.update(headers or {})
    body = http_get_bytes(url, headers=req_headers, timeout=timeout, retries=retries)
    return json.loads(body) if body.strip() else None


def http_get_bytes(url, headers=None, timeout=30, retries=3):
    """GET a URL and return the raw body. Retries on network and 5xx errors with backoff.

    Raises HTTPNotFound on 404 and re-raises other 4xx errors immediately.
    """
    req_headers = {"User-Agent": USER_AGENT}
    req_headers.update(headers or {})
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise HTTPNotFound(url) from e
            if e.code < 500:
                raise
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise last_err


def empty_frame(extra_columns=()):
    df = pd.DataFrame(
        {
            "Time": pd.Series(dtype="datetime64[ns, UTC]"),
            "status": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
    for col in extra_columns:
        df[col] = pd.Series(dtype="object")
    return df


class Connector(ABC):
    """Fetch a time series for [start, end] as a DataFrame with Time | status | value.

    Rows after `end` are always dropped, so passing end=t0 can't leak future data.
    Upstream failures are logged and give an empty frame instead of raising.
    """

    extra_columns = ()

    @property
    def columns(self):
        return COLUMNS + list(self.extra_columns)

    def fetch(self, start, end):
        start, end = to_utc(start), to_utc(end)
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        try:
            df = self._fetch(start, end)
        except Exception as e:  # one outage must not kill the run
            log(f"{type(self).__name__} failed for {start}..{end}: {e!r}")
            df = None
        if df is None or df.empty:
            return empty_frame(self.extra_columns)

        df = df[(df["Time"] >= start) & (df["Time"] <= end)]
        df = df.sort_values("Time", kind="stable").reset_index(drop=True)
        df["value"] = df["value"].astype("float64")
        return df[self.columns]

    @abstractmethod
    def _fetch(self, start, end):
        """Return a DataFrame with at least the schema columns, or None."""
