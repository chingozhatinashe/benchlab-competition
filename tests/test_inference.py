"""inference: live history assembly and the fallback chain, without the network."""

import numpy as np
import pandas as pd
import pytest

from connectors import ace_archive
from connectors.ace_archive import ACEArchiveConnector
from connectors.base import STATUS_MISSING, STATUS_OK, empty_frame
from inference import pipeline
from inference.pipeline import FORECAST_HOURS, fetch_history, run_forecast
from models.final_baseline import FinalBaseline

T0 = pd.Timestamp("2026-10-21T12:00Z")


class FakeConnector:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def fetch(self, start, end):
        self.calls.append((start, end))
        df = self.frame
        return df[(df["Time"] >= start) & (df["Time"] <= end)].reset_index(drop=True)


def minutes_frame(start, end, value, source=None, status=STATUS_OK):
    times = pd.date_range(start, end, freq="min", inclusive="left")
    df = pd.DataFrame({"Time": times, "status": status, "value": float(value)})
    if source is not None:
        df["source"] = source
    return df


def test_history_uses_only_minutes_before_t0_and_tops_up_from_realtime():
    hours = pd.Timedelta(hours=1)
    archive = FakeConnector(minutes_frame(T0 - 10 * hours, T0 - 2 * hours, 400))
    realtime = FakeConnector(pd.concat([
        minutes_frame(T0 - pd.Timedelta(hours=3), T0 + pd.Timedelta(hours=1), 500, "ACE"),
        minutes_frame(T0 - pd.Timedelta(hours=3), T0 + pd.Timedelta(hours=1), 900, "IMAP"),
    ], ignore_index=True))
    history = fetch_history(T0, 12, archive, realtime)

    assert len(history) == 12
    assert history.index[0] == T0 - pd.Timedelta(hours=12)
    assert history.index[-1] == T0 - pd.Timedelta(hours=1)
    assert history.iloc[:2].isna().all()  # before the archive starts
    assert (history.iloc[2:9] == 400).all()  # the archive wins where it has data
    assert (history.iloc[-2:] == 500).all()  # ACE real-time minutes fill the rest
    # nothing at or after t0 was requested
    assert all(end < T0 for _, end in archive.calls + realtime.calls)


def test_realtime_falls_back_to_other_spacecraft_without_ace():
    archive = FakeConnector(empty_frame())
    realtime = FakeConnector(minutes_frame(T0 - pd.Timedelta(hours=2), T0, 600, "IMAP"))
    history = fetch_history(T0, 4, archive, realtime)
    assert history.iloc[-2:].eq(600).all() and history.iloc[:2].isna().all()


def test_missing_minutes_are_not_averaged():
    frame = pd.concat([
        minutes_frame(T0 - pd.Timedelta(hours=1), T0 - pd.Timedelta(minutes=30), 400),
        minutes_frame(T0 - pd.Timedelta(minutes=30), T0, np.nan, status=STATUS_MISSING),
    ], ignore_index=True)
    history = fetch_history(T0, 1, FakeConnector(frame), FakeConnector(empty_frame(["source"])))
    assert history.iloc[-1] == 400


@pytest.fixture(scope="module")
def shipped():
    return FinalBaseline.load()


def synthetic_history(hours, value=450.0):
    index = pd.date_range(T0 - pd.Timedelta(hours=hours), T0, freq="h", inclusive="left")
    rng = np.random.default_rng(0)
    return pd.Series(value + rng.normal(0, 20, hours), index=index)


def test_forecast_with_the_shipped_model(shipped):
    out = run_forecast(T0, shipped, synthetic_history(shipped.config.history_hours))
    assert len(out) == FORECAST_HOURS and all(np.isfinite(out))
    assert all(200 <= v <= 1500 for v in out)


def test_no_data_gives_the_training_mean(shipped):
    out = run_forecast(T0, shipped, pipeline.empty_history(T0, 100))
    assert np.isfinite(out).all()


def test_broken_model_falls_back_to_persistence():
    class Broken:
        class config:
            history_hours = 10

        mean = 430.0

        def predict(self, history, t0):
            raise RuntimeError("boom")

    history = synthetic_history(10)
    out = run_forecast(T0, Broken(), history)
    assert out == [float(history.iloc[-1])] * FORECAST_HOURS


def test_missing_asset_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "ASSET", tmp_path / "nope.json")
    monkeypatch.setattr(pipeline, "load_model", lambda path=None: None)
    out = run_forecast(T0, None, pipeline.empty_history(T0, 5))
    assert out == [pipeline.DEFAULT_SPEED] * FORECAST_HOURS


def test_offline_mode_skips_the_network(monkeypatch, shipped):
    monkeypatch.setenv(pipeline.OFFLINE_ENV, "1")
    monkeypatch.setattr(pipeline, "fetch_history", lambda *a, **k: pytest.fail("network used"))
    assert len(run_forecast(T0, shipped)) == FORECAST_HOURS


def test_archive_connector_skips_future_days_and_tolerates_gaps(monkeypatch):
    requested = []

    def fake_download(self, day):
        requested.append(day)
        if day.day % 2:
            return None  # no file that day
        lines = [f"{t:%Y %m %d  %H%M}   00000   00000    0        1.0      480.0     5.00e+04"
                 for t in pd.date_range(day, periods=1440, freq="min")]
        return "\n".join(lines)

    monkeypatch.setattr(ACEArchiveConnector, "download", fake_download)
    now = pd.Timestamp.now(tz="UTC")
    df = ACEArchiveConnector().fetch(now - pd.Timedelta(days=3), now + pd.Timedelta(days=5))
    assert max(requested) == now.floor("D")  # nothing after today
    assert len(requested) == 4
    assert (df["value"] == 480.0).all() and df["Time"].max() <= now + pd.Timedelta(days=5)
    assert ace_archive.parse_minutes("").empty
