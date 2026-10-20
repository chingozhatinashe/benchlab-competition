"""Connectors: schema, status mapping, time filtering and failure handling."""

import json
import os
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from connectors import (
    COLUMNS,
    STATUS_FLAGGED,
    STATUS_MISSING,
    STATUS_OK,
    CDAWebConnector,
    SWPCConnector,
    base,
    cdaweb,
    swpc,
)
from connectors.base import HTTPNotFound, to_utc

FIXTURES = Path(__file__).parent / "fixtures"
LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def check_schema(df, extra=()):
    assert list(df.columns) == COLUMNS + list(extra)
    assert str(df["Time"].dtype) == "datetime64[ns, UTC]"
    assert df["value"].dtype == "float64"
    assert set(df["status"]) <= {STATUS_OK, STATUS_MISSING, STATUS_FLAGGED}
    assert df["Time"].is_monotonic_increasing


@pytest.fixture
def swpc_payload(monkeypatch):
    payload = load("swpc_rtsw_wind_1m.json")
    monkeypatch.setattr(swpc, "http_get_json", lambda url, **kw: payload)
    return payload


@pytest.fixture
def cdaweb_calls(monkeypatch):
    payload = load("cdaweb_omni2_h0_mrg1hr.json")
    calls = []

    def fake(url, **kw):
        calls.append(url)
        return payload

    monkeypatch.setattr(cdaweb, "http_get_json", fake)
    return calls


def test_to_utc_formats():
    want = pd.Timestamp("2026-11-11T06:00:00", tz="UTC")
    assert to_utc("20261111T060000Z") == want
    assert to_utc("2026-11-11T06:00:00Z") == want
    assert to_utc("2026-11-11 06:00") == want
    assert to_utc(pd.Timestamp("2026-11-11T07:00:00", tz="Europe/Paris")) == want


def test_swpc_schema_and_status(swpc_payload):
    df = SWPCConnector().fetch("20261020T000000Z", "20261020T060000Z")
    check_schema(df, extra=("source",))
    assert len(df) == 6
    assert set(df["source"]) == {"SOLAR1", "ACE", "IMAP"}

    by_key = df.set_index([df["Time"].dt.strftime("%H:%M"), "source"])["status"]
    assert by_key[("03:00", "ACE")] == STATUS_OK
    assert by_key[("03:00", "SOLAR1")] == STATUS_FLAGGED  # 1850 km/s, out of range
    assert by_key[("03:01", "SOLAR1")] == STATUS_MISSING  # null speed
    assert by_key[("03:01", "IMAP")] == STATUS_FLAGGED  # overall_quality != 0


def test_swpc_end_is_inclusive_and_drops_later_rows(swpc_payload):
    df = SWPCConnector().fetch("20261020T030000Z", "20261020T030100Z")
    assert df["Time"].max() == pd.Timestamp("2026-10-20T03:01:00", tz="UTC")
    assert len(df) == 4


def test_cdaweb_schema_and_status(cdaweb_calls):
    df = CDAWebConnector().fetch("20240101T000000Z", "20240101T050000Z")
    check_schema(df)
    assert list(df["status"]) == [STATUS_OK, STATUS_MISSING, STATUS_FLAGGED, STATUS_OK]
    assert pd.isna(df["value"].iloc[1])  # fill value becomes NaN
    assert df["value"].iloc[2] == 1500.0  # flagged values are kept
    assert df["Time"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    assert cdaweb_calls == [
        "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/"
        "OMNI2_H0_MRG1HR/data/20240101T000000Z,20240101T050000Z/V1800?format=json"
    ]


def test_cdaweb_filters_to_range(cdaweb_calls):
    df = CDAWebConnector().fetch("20240101T010000Z", "20240101T020000Z")
    assert list(df["Time"].dt.hour) == [1, 2]


def test_cdaweb_chunks_long_ranges(cdaweb_calls):
    CDAWebConnector().fetch("20200101T000000Z", "20221231T000000Z")
    assert len(cdaweb_calls) == 3


def test_cdaweb_404_gives_empty_frame(monkeypatch):
    def not_found(url, **kw):
        raise HTTPNotFound(url)

    monkeypatch.setattr(cdaweb, "http_get_json", not_found)
    df = CDAWebConnector().fetch("20240101T000000Z", "20240102T000000Z")
    check_schema(df)
    assert df.empty


@pytest.mark.parametrize("connector", [SWPCConnector, CDAWebConnector])
def test_outage_gives_empty_frame(monkeypatch, connector):
    def down(url, **kw):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(swpc, "http_get_json", down)
    monkeypatch.setattr(cdaweb, "http_get_json", down)
    conn = connector()
    df = conn.fetch("20240101T000000Z", "20240102T000000Z")
    check_schema(df, extra=conn.extra_columns)
    assert df.empty


def test_end_before_start_raises():
    with pytest.raises(ValueError):
        SWPCConnector().fetch("20240102T000000Z", "20240101T000000Z")


def test_http_retries_then_raises(monkeypatch):
    attempts = []

    def boom(*a, **kw):
        attempts.append(1)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(base.urllib.request, "urlopen", boom)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    with pytest.raises(urllib.error.URLError):
        base.http_get_json("https://example.invalid", retries=3)
    assert len(attempts) == 3


@pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_TESTS=1 to hit the real services")
def test_live_swpc():
    end = pd.Timestamp.now(tz="UTC")
    df = SWPCConnector().fetch(end - pd.Timedelta(hours=6), end)
    check_schema(df, extra=("source",))
    assert not df.empty


@pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_TESTS=1 to hit the real services")
def test_live_cdaweb():
    df = CDAWebConnector().fetch("20240101T000000Z", "20240110T000000Z")
    check_schema(df)
    assert len(df) == 9 * 24 + 1
