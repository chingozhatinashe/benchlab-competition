"""training.pull_omni: CSV + manifest output, default window, failure handling."""

import json

import pandas as pd
import pytest

from connectors.base import empty_frame
from training import pull_omni

END = pd.Timestamp("2026-10-16T14:00:00", tz="UTC")


def fake_frame(start, end):
    times = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"Time": times, "status": "ok", "value": range(len(times))}).astype(
        {"value": "float64"}
    )


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_pull(var, start, end):
        seen.append((var, start, end))
        return fake_frame(start, end)

    monkeypatch.setattr(pull_omni, "pull_variable", fake_pull)
    monkeypatch.setattr(
        pull_omni,
        "http_get_json",
        lambda url, **kw: {
            "InventoryDescription": [
                {"TimeInterval": [{"Start": "1963-01-01T00:00:00Z", "End": "2026-10-16T14:20:00Z"}]}
            ]
        },
    )
    return seen


def test_writes_csv_per_variable_and_manifest(calls, tmp_path):
    failed = pull_omni.run(start="20261016T100000Z", end="20261016T140000Z", out=tmp_path)
    assert failed == []
    assert [c[0] for c in calls] == pull_omni.VARIABLES

    for var in pull_omni.VARIABLES:
        lines = (tmp_path / f"{var}.csv").read_text().splitlines()
        assert lines[0] == "Time,status,value"
        assert lines[1] == "2026-10-16T10:00:00Z,ok,0.0"
        assert len(lines) == 6
        df = pd.read_csv(tmp_path / f"{var}.csv", parse_dates=["Time"])
        assert str(df["Time"].dt.tz) == "UTC"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset"] == "OMNI2_H0_MRG1HR"
    assert manifest["start"] == "2026-10-16T10:00:00Z"
    assert manifest["end"] == "2026-10-16T14:00:00Z"
    assert manifest["failed"] == []
    assert manifest["variables"]["V"] == {
        "rows": 5,
        "status": {"ok": 5},
        "first": "2026-10-16T10:00:00Z",
        "last": "2026-10-16T14:00:00Z",
    }


def test_default_window_is_two_years_to_inventory_end(calls, tmp_path):
    pull_omni.run(variables=["V"], out=tmp_path)
    _, start, end = calls[0]
    assert end == END  # inventory end floored to the hour
    assert start == END - pd.Timedelta(days=730)


def test_empty_fetch_keeps_old_file_and_fails(monkeypatch, calls, tmp_path):
    (tmp_path / "N.csv").write_text("old")

    def pull(var, start, end):
        return empty_frame() if var == "N" else fake_frame(start, end)

    monkeypatch.setattr(pull_omni, "pull_variable", pull)
    code = pull_omni.main(
        [
            "--vars",
            "V",
            "N",
            "--start",
            "20261016T100000Z",
            "--end",
            "20261016T120000Z",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert (tmp_path / "N.csv").read_text() == "old"
    assert (tmp_path / "V.csv").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["failed"] == ["N"]


def test_pull_variable_requests_1800_name(monkeypatch):
    seen = {}

    def fake_fetch(self, start, end):
        seen.update(variable=self.variable, request=self.request_variable, dataset=self.dataset)
        return empty_frame()

    monkeypatch.setattr(pull_omni.CDAWebConnector, "fetch", fake_fetch)
    pull_omni.pull_variable("BZ_GSM", END, END)
    assert seen == {"variable": "BZ_GSM", "request": "BZ_GSM1800", "dataset": "OMNI2_H0_MRG1HR"}
