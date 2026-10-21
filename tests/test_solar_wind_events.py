"""training.solar_wind_events: in-situ CME, shock and stream detection on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from training import solar_wind_events as ev

START = pd.Timestamp("2026-01-01T00:00Z")


def test_expected_temperature_matches_lopez():
    assert ev.expected_temperature(400.0) == pytest.approx((0.031 * 400 - 5.1) ** 2 * 1e3)
    assert ev.expected_temperature(600.0) == pytest.approx((0.51 * 600 - 142) * 1e3)


def minutes(hours, speed=400.0, density=5.0, field=5.0):
    index = pd.date_range(START, periods=hours * 60, freq="min")
    temp = ev.expected_temperature(speed)
    return pd.DataFrame(
        {"speed": speed, "density": density, "temperature": temp, "bt": field, "bz": 0.0},
        index=index,
    )


def test_shock_needs_all_four_jumps_or_a_field_jump():
    m = minutes(10)
    after = m.index >= START + pd.Timedelta(hours=5)
    shocked = m.copy()
    shocked.loc[after, ["speed"]] += 100
    shocked.loc[after, ["density", "temperature", "bt"]] *= 2
    assert len(ev.detect_shocks(shocked)) == 1

    interface = m.copy()  # speed and temperature up, density down: not a shock
    interface.loc[after, "speed"] += 100
    interface.loc[after, "temperature"] *= 2
    interface.loc[after, "density"] /= 2
    assert len(ev.detect_shocks(interface)) == 0

    stuck = m.copy()  # plasma frozen, field jumps (January 2026)
    stuck.loc[after, "bt"] = 25.0
    assert len(ev.detect_shocks(stuck)) == 1


def hourly(n=96, speed=450.0, field=5.0):
    index = pd.date_range(START, periods=n, freq="h")
    return pd.DataFrame(
        {"speed": speed, "temperature": ev.expected_temperature(speed), "bt": field},
        index=index,
    )


def test_cold_strong_field_interval_is_an_icme_with_its_sheath():
    h = hourly()
    h.iloc[40:60, h.columns.get_loc("temperature")] *= 0.3
    h.iloc[40:60, h.columns.get_loc("bt")] = 15.0
    h.iloc[50, h.columns.get_loc("temperature")] = np.nan  # a gap inside is bridged
    shocks = pd.DatetimeIndex([START + pd.Timedelta(hours=32, minutes=20)])
    events = ev.detect_icmes(h, shocks)
    icme = events[events["kind"] == "ICME"].iloc[0]
    assert icme["ejecta_start"] == h.index[40]
    assert icme["end"] == h.index[60]
    assert icme["start"] == START + pd.Timedelta(hours=32)  # sheath from the shock
    assert (events["kind"] == "shock only").sum() == 0
    mask = ev.cme_mask(h.index, events)
    assert mask.iloc[32:60].all() and not mask.iloc[:32].any() and not mask.iloc[60:].any()


def test_weak_field_or_short_cold_intervals():
    h = hourly()
    h.iloc[10:30, h.columns.get_loc("temperature")] *= 0.3  # weak field: possible ICME
    h.iloc[60:65, h.columns.get_loc("temperature")] *= 0.3  # too short
    events = ev.detect_icmes(h, pd.DatetimeIndex([]))
    assert list(events["kind"]) == ["possible ICME"]
    assert not ev.cme_mask(h.index, events).any()
    assert ev.cme_mask(h.index, events, ("possible ICME",)).sum() == 20


def test_shock_without_ejecta_marks_a_day():
    h = hourly()
    events = ev.detect_icmes(h, pd.DatetimeIndex([START + pd.Timedelta(hours=10, minutes=30)]))
    assert list(events["kind"]) == ["shock only"]
    assert ev.cme_mask(h.index, events).sum() == 24


def test_stream_arrival_found_outside_cmes_only():
    n = 24 * 12
    speed = np.full(n, 380.0)
    speed[100:130] = np.linspace(380, 650, 30)
    speed[130:] = 650.0
    s = pd.Series(speed, index=pd.date_range(START, periods=n, freq="h"))
    none = pd.Series(False, index=s.index)
    arrivals = ev.detect_stream_arrivals(s, none)
    assert len(arrivals) == 1
    assert abs((arrivals[0] - s.index[115]) / pd.Timedelta(hours=1)) <= 2
    assert len(ev.detect_stream_arrivals(s, pd.Series(True, index=s.index))) == 0


def test_classify_runs():
    index = pd.date_range(START, periods=24 * 20, freq="h")
    cme = pd.Series(False, index=index)
    cme.iloc[100:110] = True
    possible = pd.Series(False, index=index)
    possible.iloc[300:305] = True
    arrivals = pd.DatetimeIndex([index[200]])
    t0s = pd.DatetimeIndex([index[50], index[150], index[250], index[350], index[400]])
    actual = np.array([np.full(72, v) for v in (500.0, 500.0, 500.0, 600.0, 400.0)])
    labels = ev.classify_runs(t0s, actual, cme, arrivals, possible)
    assert list(labels) == ["CME", "stream arrival", "possible CME", "fast stream", "slow wind"]
