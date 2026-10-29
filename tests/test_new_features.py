"""The new regressor statistics (feature_engineering.windows): protons, F10.7 trend, lags."""

import numpy as np
import pandas as pd

from feature_engineering import build, windows
from training.pull_protons import integral_10mev

T0S = pd.DatetimeIndex(["2024-05-12T00:00Z", "2024-05-12T06:00Z"])


def _flux(seed=0):
    hours = pd.date_range("2024-04-01", "2024-05-20", freq="1h", tz="UTC")
    return pd.Series(np.random.default_rng(seed).lognormal(0, 2, len(hours)), index=hours)


def test_protons_ignore_hours_not_finished_by_t0_minus_latency():
    flux = _flux()
    specs = windows.proton_candidates()
    real = windows.window_table(build.Inputs(protons=flux), T0S[:1], specs)
    # the hour that starts at t0 - 1 h ends at t0, after t0 - 10 min: not usable
    scrambled = flux.where(flux.index < T0S[0] - pd.Timedelta(hours=1), 1e6)
    late = windows.window_table(build.Inputs(protons=scrambled), T0S[:1], specs)
    pd.testing.assert_frame_equal(late, real)


def test_protons_values():
    flux = pd.Series(0.01, index=_flux().index)
    flux[pd.Timestamp("2024-05-11T20:00Z")] = 1000.0
    out = windows.window_table(build.Inputs(protons=flux), T0S[:1], windows.proton_candidates())
    assert out["log_p10_max_24h"].iloc[0] == 3.0
    assert np.isclose(out["log_p10_mean_6h"].iloc[0], 3 / 6)  # floored at 1 pfu


def test_f107_trend_is_change_of_mean_per_day():
    days = pd.date_range("2023-01-01", "2024-06-01", freq="D", tz="UTC")
    f107 = pd.Series(100 + 0.5 * np.arange(len(days)), index=days)
    out = windows.window_table(build.Inputs(f107=f107), T0S, windows.cycle_candidates())
    assert np.allclose(out["f107_trend_1944h"], 0.5)
    assert np.allclose(out["f107_trend_648h"], 0.5)


def test_lags_shift_by_one_day():
    hours = pd.date_range("2024-03-01", "2024-05-12", freq="1h", tz="UTC")
    hourly = pd.DataFrame({"bt": np.arange(len(hours), dtype=float)}, index=hours)
    out = windows.window_table(build.Inputs(hourly=hourly), T0S[:1],
                               windows.recurrence_candidates())
    assert out["bt_lag26_72h"].iloc[0] - out["bt_lag27_72h"].iloc[0] == 24
    assert out["bt_lag27_72h"].iloc[0] - out["bt_lag28_72h"].iloc[0] == 24


def test_integral_counts_only_the_part_above_10_mev():
    lower = np.array([[5_000.0, 20_000.0]])
    upper = np.array([[15_000.0, 30_000.0]])
    diff = np.array([[[1.0, 2.0]]])  # one time, one sensor
    assert integral_10mev(diff, lower, upper)[0] == 1.0 * 5_000 + 2.0 * 10_000
