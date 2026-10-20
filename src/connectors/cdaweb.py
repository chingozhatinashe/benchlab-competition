"""NASA CDAWeb (CDAS REST web services), OMNI hourly solar wind speed by default."""

import numpy as np
import pandas as pd

from connectors.base import (
    STATUS_FLAGGED,
    STATUS_MISSING,
    STATUS_OK,
    Connector,
    HTTPNotFound,
    http_get_json,
)

BASE_URL = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets"
CHUNK = pd.Timedelta(days=365)


def _fmt(ts):
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _vattr(var, name):
    for attr in var.get("cdfVAttributes", {}).get("attribute", []):
        if attr["name"] == name and attr.get("entry"):
            return attr["entry"][0].get("value")
    return None


def _records(var):
    return [rec["value"][0] for rec in var.get("cdfVarData", {}).get("record", [])]


class CDAWebConnector(Connector):
    """One variable of a CDAWeb dataset, as Time | status | value.

    Defaults to OMNI2_H0_MRG1HR `V` (hourly flow speed, km/s). OMNI lags real
    time by several days, so it's mainly useful for history and training.

    `request_variable` is the name sent in the URL. For OMNI hourly, asking for
    `V` directly gives a 400, but asking for `V1800` returns `V` alongside it.
    """

    def __init__(
        self,
        dataset="OMNI2_H0_MRG1HR",
        variable="V",
        request_variable="V1800",
        epoch_variable="Epoch",
        timeout=60,
    ):
        self.dataset = dataset
        self.variable = variable
        self.request_variable = request_variable or variable
        self.epoch_variable = epoch_variable
        self.timeout = timeout

    def url(self, start, end):
        return (
            f"{BASE_URL}/{self.dataset}/data/{_fmt(start)},{_fmt(end)}/"
            f"{self.request_variable}?format=json"
        )

    def _fetch(self, start, end):
        frames = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + CHUNK, end)
            frame = self._fetch_chunk(chunk_start, chunk_end)
            if frame is not None and not frame.empty:
                frames.append(frame)
            chunk_start = chunk_end + pd.Timedelta(seconds=1)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates("Time", keep="first")

    def _fetch_chunk(self, start, end):
        try:
            payload = http_get_json(self.url(start, end), timeout=self.timeout)
        except HTTPNotFound:
            return None  # CDAS answers 404 when the range has no data
        return self.parse(payload)

    def parse(self, payload):
        if not payload or not payload.get("CDF"):
            return None
        variables = {v["name"]: v for v in payload["CDF"][0]["cdfVariables"]["variable"]}
        if self.variable not in variables or self.epoch_variable not in variables:
            raise KeyError(
                f"{self.dataset}: response lacks {self.variable!r} or {self.epoch_variable!r}; "
                f"got {sorted(variables)}"
            )
        var = variables[self.variable]
        times = pd.to_datetime(_records(variables[self.epoch_variable]), utc=True)
        values = pd.to_numeric(pd.Series(_records(var)), errors="coerce").to_numpy()
        if len(times) != len(values):
            raise ValueError(f"{self.dataset}: {len(times)} epochs but {len(values)} values")

        fill = _num(_vattr(var, "FILLVAL"))
        vmin, vmax = _num(_vattr(var, "VALIDMIN")), _num(_vattr(var, "VALIDMAX"))

        missing = np.isnan(values)
        if fill is not None:
            missing |= np.isclose(values, fill)
        flagged = np.zeros_like(missing)
        if vmin is not None:
            flagged |= values < vmin
        if vmax is not None:
            flagged |= values > vmax
        flagged &= ~missing

        values = np.where(missing, np.nan, values)
        status = np.select([missing, flagged], [STATUS_MISSING, STATUS_FLAGGED], STATUS_OK)
        return pd.DataFrame({"Time": times, "status": status, "value": values})


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
