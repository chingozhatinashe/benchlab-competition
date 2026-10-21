"""training.pull_lasco_cme: catalogue parsing and the arrival-time model, without the network."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import pull_lasco_cme as lasco

FIXTURE = (Path(__file__).parent / "fixtures" / "lasco_univ_sample.txt").read_text()


def test_parse_reads_every_row_and_marks_halos_and_quality():
    df = lasco.parse_month(FIXTURE)
    assert len(df) == 6
    assert df["time"].iloc[0] == pd.Timestamp("2025-10-01T02:24:05Z")
    halo = df.iloc[-1]
    assert halo["halo"] and np.isnan(halo["central_pa"]) and halo["width"] == 360
    assert halo["linear_speed"] == 1192 and halo["accel"] == -24.9
    assert halo["mass"] == pytest.approx(2.4e16)  # the '*' uncertainty mark is dropped
    partial = df.iloc[4]
    assert partial["partial_halo"] and not partial["halo"] and partial["quality"] == "poor"
    assert np.isnan(df.iloc[2]["mass"])  # '-------' means missing
    assert df.iloc[1]["only_c2"] and df.iloc[0]["quality"] == "very poor"


def test_parse_ignores_headers_and_empty_text():
    assert lasco.parse_month("").empty
    assert lasco.parse_month(" ===== header\n Date Time\n").empty


def test_arrival_model_matches_gopalswamy_2001():
    # Worked by hand for 1000 km/s: a = 2.193 - 5.4 = -3.207 m/s^2 over 0.76 AU gives
    # 520 km/s after 41.6 h, then 0.24 AU at 520 km/s takes 19.2 h: 60.8 h in all.
    hours = lasco.arrival_hours([400.0, 1000.0, 2000.0])
    assert hours[1] == pytest.approx(60.8, abs=0.2)
    assert hours[0] == pytest.approx(102.4, abs=0.5)  # ~4.3 days
    assert hours[2] == pytest.approx(25.4, abs=0.5)  # ~1.1 days
    assert np.all(np.diff(hours) < 0)  # faster CMEs arrive sooner
    assert np.isnan(lasco.arrival_hours([np.nan]))[0]


def test_arrival_only_for_halo_candidates():
    df = lasco.add_arrival(lasco.parse_month(FIXTURE))
    assert df["est_arrival"].notna().tolist() == [False, False, False, False, True, True]
    halo = df.iloc[-1]
    assert halo["est_arrival"] - halo["time"] == pd.to_timedelta(halo["transit_hours"], unit="h")


def test_run_writes_csv_and_manifest_and_skips_unpublished_months(monkeypatch, tmp_path):
    def fake_fetch(month, raw_dir, refresh=False):
        return FIXTURE if month.month == 10 else None

    monkeypatch.setattr(lasco, "fetch_month", fake_fetch)
    cmes = lasco.run("2025-10", "2025-11", tmp_path)
    assert len(cmes) == 6
    loaded = lasco.load(tmp_path / "cmes.csv")
    assert loaded["time"].iloc[-1] == pd.Timestamp("2025-10-21T20:24:05Z")
    manifest = (tmp_path / "manifest.json").read_text()
    assert '"2025-11"' in manifest and '"halo": 1' in manifest
