"""Regressors: physics, the no-future-data guarantee, imputation and failure handling."""

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from feature_engineering import build, coronal_holes, impute, insitu, registry, solar_activity
from inference import fetch_inputs as fi

T0 = pd.Timestamp("2026-03-15T12:00Z")
H = pd.Timedelta(hours=1)


# ---------------------------------------------------------------------- helpers
def synthetic_minutes(start, end, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="min", inclusive="left", name="Time")
    n = len(idx)
    plasma = pd.DataFrame({"density": 5 + rng.random(n), "speed": 400 + 50 * rng.random(n),
                           "temperature": 8e4 + 1e4 * rng.random(n)}, index=idx)
    mag = pd.DataFrame({"bx": -3 + rng.random(n), "by": 2 + rng.random(n),
                        "bz": rng.normal(0, 2, n), "bt": 6 + rng.random(n)}, index=idx)
    epam = pd.DataFrame({"p47": 1e3 * (1 + rng.random(n)), "p310": 1 + rng.random(n)},
                        index=idx)[::5]
    return plasma, mag, epam


def jpeg(img):
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def synthetic_sun(taken, hole_lon=None, hole_halfwidth=8.0):
    """A uniform disk (grey 100) with a square dark patch (grey 20) centred on hole_lon."""
    r = coronal_holes.disk_radius(taken)
    lat, lon, mu = coronal_holes.heliographic(r)
    img = np.where(np.isfinite(mu), 100.0, 0.0)
    if hole_lon is not None:
        patch = (np.abs(np.nan_to_num(lon, nan=999) - hole_lon) <= hole_halfwidth) & (
            np.abs(np.nan_to_num(lat, nan=999)) <= 20)
        img[patch] = 20.0
    return img


# ---------------------------------------------------------------------- in-situ physics
def test_derived_quantities():
    idx = pd.date_range(T0, periods=3, freq="min")
    plasma = pd.DataFrame({"density": [5.0, 10.0, 5.0], "speed": [400.0, 400.0, 400.0],
                           "temperature": [8e4, 1e4, 9e9]}, index=idx)
    mag = pd.DataFrame({"bx": [-4.0, 3.0, 0.0], "by": [0, 0, 0.0], "bz": [3.0, 0, 0],
                        "bt": [5.0, 10.0, 1.0]}, index=idx)
    m = insitu.minute_frame(plasma, mag)
    assert m["pressure"].iloc[0] == pytest.approx(1.6726e-6 * 5 * 400**2)  # 1.34 nPa
    assert m["polarity"].tolist()[:2] == [0.8, -0.3]  # Bx < 0 points away from the Sun
    assert m["icme"].tolist()[:2] == [0.0, 1.0]  # cold (T/Texp ~ 0.15) and Bt >= 8 nT
    assert np.isnan(m["log_temperature"].iloc[2])  # out of range -> NaN, not used


def test_hourly_bins_are_shifted_by_the_latency():
    t0 = pd.Timestamp("2026-03-15T12:00Z")
    idx = pd.DatetimeIndex([t0 - pd.Timedelta(minutes=15), t0 - pd.Timedelta(minutes=5)])
    plasma = pd.DataFrame({"density": [1.0, 99.0], "speed": [400.0, 400.0],
                           "temperature": [1e5, 1e5]}, index=idx)
    hourly = insitu.hourly_frame(insitu.minute_frame(plasma, None), t0 - 2 * H, t0 + H)
    # 11:45 belongs to the 11:00 row (available by 12:00); 11:55 is only published after
    # t0, so it falls in the 12:00 row
    assert hourly.loc[t0 - H, "density"] == 1.0
    assert hourly.loc[t0, "density"] == 99.0


def test_shock_detection_and_its_confirmation_time():
    idx = pd.date_range(T0 - 2 * H, T0 + 2 * H, freq="min", inclusive="left")
    jump = idx >= T0
    minutes = pd.DataFrame({"speed": np.where(jump, 500.0, 400.0),
                            "density": np.where(jump, 10.0, 5.0),
                            "temperature": np.where(jump, 2e5, 1e5),
                            "bt": np.where(jump, 12.0, 5.0)}, index=idx)
    shocks = insitu.detect_shocks(minutes)
    assert len(shocks) == 1 and abs(shocks[0] - T0) <= pd.Timedelta(minutes=10)
    assert len(insitu.confirmed_by(shocks, shocks[0] + pd.Timedelta(minutes=19))) == 0
    assert len(insitu.confirmed_by(shocks, shocks[0] + pd.Timedelta(minutes=20))) == 1
    calm = insitu.detect_shocks(minutes[minutes.index < T0])
    assert len(calm) == 0 and str(calm.tz) == "UTC"  # comparable with tz-aware times


# ---------------------------------------------------------------------- coronal holes
def test_coronal_hole_at_the_meridian_is_found_in_the_right_strip():
    taken = pd.Timestamp("2024-01-15T12:00Z")
    centre = coronal_holes.metrics(jpeg(synthetic_sun(taken, 0.0)), taken)
    east = coronal_holes.metrics(jpeg(synthetic_sun(taken, -30.0)), taken)
    quiet = coronal_holes.metrics(jpeg(synthetic_sun(taken)), taken)
    # the patch (16 deg wide, |lat| <= 20) fills most of the 20 deg strips' middle third
    assert centre["ch_cm"] > 0.25 and centre["ch_east"] < 0.01
    assert east["ch_east"] > 0.25 and east["ch_cm"] < 0.01
    assert max(quiet.values()) < 0.01
    assert 0 < centre["ch_disk"] < centre["ch_cm"]


def test_blank_and_truncated_frames_are_rejected():
    taken = pd.Timestamp("2024-01-15T12:00Z")
    sun = synthetic_sun(taken)
    cut = sun.copy()
    cut[:256] = 0.0
    assert coronal_holes.usable(coronal_holes.grayscale(jpeg(sun)), taken)
    assert not coronal_holes.usable(coronal_holes.grayscale(jpeg(sun * 0)), taken)
    assert not coronal_holes.usable(coronal_holes.grayscale(jpeg(cut)), taken)
    assert coronal_holes.grayscale(b"not a jpeg") is None
    assert np.isnan(coronal_holes.metrics(b"not a jpeg", taken)["ch_cm"])


def test_disk_radius_follows_the_earth_sun_distance():
    jan = coronal_holes.disk_radius(pd.Timestamp("2024-01-03T00:00Z"))
    jul = coronal_holes.disk_radius(pd.Timestamp("2024-07-04T00:00Z"))
    assert 200 < jan < 204 and 194 < jul < 198


# ---------------------------------------------------------------------- no future data
def full_inputs(t0):
    plasma, mag, epam = synthetic_minutes(t0 - pd.Timedelta(days=31), t0 + pd.Timedelta(days=3))
    minutes = insitu.minute_frame(plasma, mag, epam)
    hourly = insitu.hourly_frame(minutes, t0 - pd.Timedelta(days=31), t0 + pd.Timedelta(days=3))
    shocks = pd.DatetimeIndex([t0 - pd.Timedelta(hours=30), t0 - pd.Timedelta(minutes=15)])
    cut = pd.date_range(t0 - pd.Timedelta(days=7), t0 + pd.Timedelta(days=2), freq="6h") - H
    ch = pd.DataFrame({n: np.linspace(0, 0.2, len(cut)) for n in coronal_holes.NAMES}, index=cut)
    flares = pd.DataFrame({"peak_time": [t0 - pd.Timedelta(hours=50), t0 - pd.Timedelta(minutes=5),
                                         t0 + pd.Timedelta(hours=2)],
                           "peak_flux": [2e-5, 5e-4, 1e-3], "flare_class": ["M2", "X5", "X10"]})
    today = t0.floor("D")
    days = pd.date_range(today - pd.Timedelta(days=40), today + pd.Timedelta(days=2))
    f107 = pd.Series(np.arange(len(days), dtype=float) + 100, index=days)
    return build.Inputs(hourly, shocks, ch, flares, f107)


def scrambled_after(inputs, t0):
    """The same inputs with everything not available by t0 replaced by nonsense."""
    hourly = inputs.hourly.copy()
    hourly.loc[hourly.index >= t0] = 1e6
    ch = inputs.coronal_holes.copy()
    ch.loc[ch.index > t0 - build.CH_LATENCY] = 0.99
    flares = inputs.flares.copy()
    flares.loc[flares["peak_time"] > t0 - solar_activity.FLARE_LATENCY, "peak_flux"] = 1.0
    f107 = inputs.f107.copy()
    f107[f107.index >= t0.floor("D")] = 999.0
    return build.Inputs(hourly, inputs.shocks, ch, flares, f107)


def test_features_do_not_change_when_the_future_does():
    inputs = full_inputs(T0)
    a = build.build_table(inputs, [T0])
    b = build.build_table(scrambled_after(inputs, T0), [T0])
    pd.testing.assert_frame_equal(a, b)
    assert a.notna().all(axis=None), a.columns[a.isna().iloc[0]].tolist()


def test_feature_values_respect_the_cutoffs():
    row = build.build_table(full_inputs(T0), [T0]).iloc[0]
    assert row["hours_since_shock"] == pytest.approx(30.0)  # the 11:45 shock isn't confirmed
    assert row["m_flares_72h"] == 1.0  # the X5 at 11:55 peaked after t0 - 10 min
    assert row["xray_peak_72h"] == pytest.approx(np.log10(2e-5))
    yesterday = T0.floor("D") - pd.Timedelta(days=1)
    assert row["f107_prev_day"] == full_inputs(T0).f107[yesterday]
    assert list(row.index) == registry.NAMES


def test_lagged_window_is_the_forecast_hours_one_rotation_earlier():
    idx = pd.date_range(T0 - pd.Timedelta(days=30), T0, freq="h", inclusive="left")
    hourly = pd.DataFrame(np.nan, index=idx, columns=insitu.HOURLY_COLUMNS)
    lag_start = T0 - pd.Timedelta(hours=648)
    window = (idx >= lag_start) & (idx < lag_start + 72 * H)
    hourly.loc[window, "density"] = 7.0
    hourly.loc[~window, "density"] = 1.0
    row = build.insitu_features(hourly, None, [T0]).iloc[0]
    assert row["density_lag27d"] == 7.0


# ---------------------------------------------------------------------- imputation
def test_imputation_fills_with_medians_and_flags():
    clim = {n: float(i) for i, n in enumerate(registry.NAMES)}
    raw = pd.DataFrame([[np.nan] * len(registry.NAMES)], columns=registry.NAMES)
    raw.loc[0, "m_flares_72h"] = 0.0  # an observed zero stays a zero
    out = impute.impute(raw, clim).iloc[0]
    assert out["density_1h"] == clim["density_1h"] and out["density_1h_missing"] == 1.0
    assert out["m_flares_72h"] == 0.0 and out["m_flares_72h_missing"] == 0.0
    assert len(out) == 2 * len(registry.NAMES)


def test_climatology_falls_back_to_registry_defaults(tmp_path):
    assert impute.load_climatology(tmp_path / "missing.json")["f107_prev_day"] == 140.0
    table = pd.DataFrame({n: [1.0, 3.0, np.nan] for n in registry.NAMES})
    medians = impute.fit_climatology(table)
    impute.save_climatology(medians, {"fitted_on": ["a", "b"]}, tmp_path / "c.json")
    assert impute.load_climatology(tmp_path / "c.json")["bt_6h"] == 2.0


def test_shipped_climatology_covers_every_feature():
    clim = impute.load_climatology()
    assert set(clim) == set(registry.NAMES) and all(np.isfinite(list(clim.values())))


# ---------------------------------------------------------------------- live failure modes
class Failing:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise RuntimeError("source down")
        return fail


def test_every_source_down_gives_an_all_imputed_row():
    sources = {k: Failing() for k in fi.default_sources()}
    inputs, report = fi.fetch_inputs(T0, sources=sources, budget=5)
    assert all(s.startswith("failed") for s in report["sources"].values())
    row = build.build_table(inputs, [T0])
    assert row.isna().all(axis=None)
    filled = impute.impute(row)
    assert filled[[f"{n}_missing" for n in registry.NAMES]].eq(1.0).all(axis=None)


class Static:
    def __init__(self, frame):
        self.frame = frame

    def minutes(self, start, end):
        return self.frame


def test_live_fetch_drops_minutes_published_after_t0():
    plasma, mag, epam = synthetic_minutes(T0 - pd.Timedelta(days=30), T0 + 2 * H)
    plasma.loc[plasma.index >= T0 - insitu.LATENCY, "density"] = 140.0
    sources = {k: Failing() for k in fi.default_sources()}
    sources.update(swepam=Static(plasma), mag=Static(mag), epam=Static(epam))
    inputs, _ = fi.fetch_inputs(T0, sources=sources, budget=10)
    row = build.build_table(inputs, [T0]).iloc[0]
    assert row["density_1h"] < 7.0 and row["pressure_max_24h"] < 5.0
