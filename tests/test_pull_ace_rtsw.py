"""training.pull_ace_rtsw: parsing, hourly averaging without imputation, caching."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from connectors.base import HTTPNotFound
from training import pull_ace_rtsw

FIXTURE = (Path(__file__).parent / "fixtures" / "ace_swepam_1m_sample.txt").read_text()
DAY = pd.Timestamp("2021-10-20", tz="UTC")


def synthetic_day(day, speed=400.0, status=0):
    """A complete 1440-minute file in the SWPC layout."""
    lines = [":Data_list: synthetic", "# header"]
    for minute in range(1440):
        t = day + pd.Timedelta(minutes=minute)
        lines.append(
            f"{t:%Y %m %d  %H%M}   00000   {minute * 60:5d}    {status}"
            f"        1.0      {speed:.1f}     5.00e+04"
        )
    return "\n".join(lines) + "\n"


def test_parse_masks_bad_status_and_fill():
    minutes = pull_ace_rtsw.parse_minutes(FIXTURE)
    assert len(minutes) == 120
    assert minutes.index[0] == DAY
    assert minutes.index.tz is not None
    assert minutes.iloc[0] == 405.4
    assert minutes.notna().sum() == 17  # every S != 0 or -9999.9 row is NaN
    assert minutes["2021-10-20 01:00":].isna().all()


def test_parse_ignores_headers_and_empty_text():
    assert pull_ace_rtsw.parse_minutes("").empty
    assert pull_ace_rtsw.parse_minutes(":Data_list: x\n# comment\n").empty


def test_hourly_is_mean_of_valid_minutes_and_never_filled():
    minutes = pull_ace_rtsw.parse_minutes(FIXTURE)
    hourly = pull_ace_rtsw.to_hourly(minutes, DAY, DAY + pd.Timedelta(hours=3))

    assert list(hourly["Time"]) == list(pd.date_range(DAY, periods=3, freq="h"))
    assert hourly.loc[0, "value"] == pytest.approx(minutes.iloc[:60].mean())
    assert hourly.loc[0, "n_valid"] == 17
    assert hourly.loc[0, "status"] == "ok"
    # Hour 01 has no valid minutes; hour 02 has no rows at all. Both stay missing.
    for i in (1, 2):
        assert np.isnan(hourly.loc[i, "value"])
        assert hourly.loc[i, "n_valid"] == 0
        assert hourly.loc[i, "status"] == "missing"


@pytest.fixture
def archive(monkeypatch):
    """Fake download: day 1 is a full file, day 2 has no file, day 3 is all flagged."""
    files = {
        DAY: synthetic_day(DAY, 400.0),
        DAY + pd.Timedelta(days=2): synthetic_day(DAY + pd.Timedelta(days=2), 500.0, status=9),
    }
    calls = []

    def fake_download(day):
        calls.append(day)
        if day not in files:
            return None
        return files[day]

    monkeypatch.setattr(pull_ace_rtsw, "download_day", fake_download)
    return calls


def test_run_writes_complete_hourly_index_and_manifest(archive, tmp_path):
    manifest = pull_ace_rtsw.run(DAY, DAY + pd.Timedelta(days=3), tmp_path)

    df = pd.read_csv(tmp_path / "speed_hourly.csv")
    assert list(df.columns) == ["Time", "status", "value", "n_valid"]
    assert len(df) == 72
    assert df["value"].iloc[:24].eq(400.0).all()
    assert df["n_valid"].iloc[:24].eq(60).all()
    assert df["value"].iloc[24:].isna().all()  # missing file and all-flagged day
    assert df["status"].iloc[24:].eq("missing").all()

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved == manifest
    assert manifest["hours"] == 72
    assert manifest["missing_hours"] == 48
    assert manifest["missing_days"] == ["2021-10-21"]


def test_complete_cached_days_are_not_downloaded_again(archive, tmp_path):
    pull_ace_rtsw.run(DAY, DAY + pd.Timedelta(days=1), tmp_path)
    pull_ace_rtsw.run(DAY, DAY + pd.Timedelta(days=1), tmp_path)
    assert archive == [DAY]


def test_partial_cached_day_is_downloaded_again(archive, tmp_path):
    raw = tmp_path / "raw" / "20211020_ace_swepam_1m.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text(FIXTURE)  # only 120 minutes
    pull_ace_rtsw.run(DAY, DAY + pd.Timedelta(days=1), tmp_path)
    assert archive == [DAY]
    assert len(pull_ace_rtsw.parse_minutes(raw.read_text())) == 1440


def test_download_day_maps_404_to_none(monkeypatch):
    def not_found(url, **kw):
        raise HTTPNotFound(url)

    monkeypatch.setattr(pull_ace_rtsw, "http_get_bytes", not_found)
    assert pull_ace_rtsw.download_day(DAY) is None
