"""Regressor connectors: parsing real (trimmed) responses, caching, availability cut-offs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from connectors import ace_archive, sdo, swpc
from connectors.ace_archive import ACEDailyArchive, parse_daily

FIX = Path(__file__).parent / "fixtures"


def fixture(name):
    return (FIX / name).read_text()


def test_mag_file_parses_every_component_and_blanks_bad_rows():
    df = parse_daily(fixture("ace_mag_1m_sample.txt"), "mag")
    assert list(df.columns) == ["bx", "by", "bz", "bt", "lat", "lon"]
    assert df.index[0] == pd.Timestamp("2025-07-01T00:00Z")
    assert df.iloc[0].tolist() == [3.1, 3.3, -2.8, 5.4, -31.5, 46.8]
    assert df.iloc[-1].isna().all()  # status 9, fill values


def test_epam_file_uses_each_detector_status():
    df = parse_daily(fixture("ace_epam_5m_sample.txt"), "epam")
    assert df.index[1] - df.index[0] == pd.Timedelta(minutes=5)
    assert df["p47"].iloc[0] == pytest.approx(5.21e3)
    last = df.iloc[-1]  # proton status 4: protons bad, electrons fine
    assert np.isnan(last["p47"]) and last["e38"] == pytest.approx(1.82e3)


def test_speed_parser_is_unchanged_by_the_generalisation():
    text = fixture("ace_swepam_1m_sample.txt")
    speeds = ace_archive.parse_minutes(text)
    assert speeds.name == "speed"
    assert speeds.equals(parse_daily(text, "swepam")["speed"].rename("speed"))


def test_daily_archive_caches_finished_days_only(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, timeout=None, retries=None):
        calls.append(url)
        return fixture("ace_mag_1m_sample.txt").encode()

    monkeypatch.setattr(ace_archive, "http_get_bytes", fake_get)
    arc = ACEDailyArchive("mag", cache_dir=tmp_path)
    old = pd.Timestamp("2025-07-01", tz="UTC")
    today = pd.Timestamp.now(tz="UTC").floor("D")
    arc.download(old)
    arc.download(old)  # second time from the cache
    arc.download(today)
    assert len(calls) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["20250701_ace_mag_1m.txt"]


def test_daily_archive_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ace_archive, "download_days", boom)
    now = pd.Timestamp.now(tz="UTC")
    out = ACEDailyArchive("epam").minutes(now - pd.Timedelta(days=1), now)
    assert out.empty and "p47" in out


def test_realtime_mag_keeps_good_ace_rows_in_the_window():
    records = json.loads(fixture("swpc_rtsw_mag_1m.json"))
    times = pd.to_datetime([r["time_tag"] for r in records], utc=True)
    rt = swpc.SWPCRealtime("mag")
    out = rt.parse(records, times.min(), times.max())
    ace = [r for r in records if r["source"] == "ACE" and r["overall_quality"] == 0]
    assert len(out) == len(ace)  # IMAP ignored while ACE is present; flagged row dropped
    oldest = min(ace, key=lambda r: r["time_tag"])  # the feed lists newest first
    assert out["bz"].iloc[0] == oldest["bz_gsm"]
    after = times.max() + pd.Timedelta(minutes=1)
    late = rt.parse(records, after, after + pd.Timedelta(hours=1))
    assert late.empty


def test_flare_list_and_f107_parse():
    flares = swpc.parse_flares(json.loads(fixture("swpc_xray_flares.json")))
    assert len(flares) == 6 and flares["peak_time"].is_monotonic_increasing
    assert flares["flare_class"].iloc[-1] == "C1.3"
    f107 = swpc.parse_f107(json.loads(fixture("swpc_f107.json")))
    assert f107[pd.Timestamp("2026-10-20", tz="UTC")] == 121.0  # the noon reading
    assert f107[pd.Timestamp("2026-10-21", tz="UTC")] == 115.0  # no noon yet: the day's mean


def test_sdo_listing_and_picking():
    times = sdo.parse_listing(fixture("sdo_listing_sample.html"))
    assert len(times) > 3 and times.is_monotonic_increasing
    cut = times[2] + pd.Timedelta(seconds=30)
    assert sdo.candidates(times, cut, pd.Timedelta(hours=24), 2) == [times[2], times[1]]
    assert sdo.candidates(times, times[0] - pd.Timedelta(seconds=1), pd.Timedelta(hours=1), 3) == []


def test_sdo_latest_skips_rejected_frames_and_never_looks_past_the_cutoff(monkeypatch):
    base = pd.Timestamp("2026-10-21T10:00Z")
    stamps = pd.DatetimeIndex([base + pd.Timedelta(minutes=10 * i) for i in range(12)])
    conn = sdo.SDOConnector()
    monkeypatch.setattr(conn, "listing", lambda day: stamps)
    monkeypatch.setattr(conn, "image", lambda t: f"{t:%H%M}".encode())
    cutoff = base + pd.Timedelta(minutes=55)  # newest allowed: 10:50
    got = conn.latest([cutoff], accept=lambda jpeg, t: jpeg != b"1050")
    taken, jpeg = got[cutoff]
    assert taken == base + pd.Timedelta(minutes=40) and jpeg == b"1040"
    nothing = conn.latest([base - pd.Timedelta(days=3)])
    assert list(nothing.values()) == [(None, None)]


def test_sdo_latest_with_no_listings_gives_nothing(monkeypatch):
    conn = sdo.SDOConnector()

    def down(day):
        raise RuntimeError("listing failed")

    monkeypatch.setattr(conn, "listing", down)
    cut = pd.Timestamp("2024-12-01T05:00Z")
    assert conn.latest([cut]) == {cut: (None, None)}
