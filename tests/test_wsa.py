import numpy as np
import pytest

from models import wsa


def test_uniform_wind_is_unchanged():
    v = np.full((2, 360), 450.0)
    assert wsa.hux(v) == pytest.approx(v)


def test_a_small_stream_drifts_to_lower_longitude_by_the_travel_time():
    lon = np.arange(360.0)
    v = 400.0 + 5.0 * np.exp(-0.5 * ((lon - 180.0) / 8.0) ** 2)
    out = wsa.hux(v[None])[0]
    travel = (wsa.AU_RSUN - wsa.R_INNER) * wsa.R_SUN_KM / 400.0  # s
    expected = 180.0 - np.degrees(wsa.OMEGA * travel)  # about 125 deg
    assert np.argmax(out) == pytest.approx(expected, abs=2)


def test_fast_wind_steepens_into_slow_wind_ahead_of_it():
    lon = np.arange(360.0)
    v = np.where((lon > 150) & (lon < 210), 700.0, 350.0)
    out = wsa.hux(v[None])[0]
    assert out.max() <= 700.0 + 1e-6 and out.min() >= 350.0 - 1e-6
    # Earth meets higher longitudes first: the stream's front (fast wind running into the slow
    # wind ahead, at its high-longitude side) steepens; the trailing rarefaction spreads out
    grad = np.diff(np.concatenate([out, out[:1]]))
    assert -grad.min() > 3 * grad.max()


def test_wsa_relation_is_slow_for_large_expansion_and_near_boundaries():
    p = wsa.Params()
    fast = wsa.source_speed(np.array([[0.5]]), np.array([[10.0]]), p)
    slow_fs = wsa.source_speed(np.array([[3.0]]), np.array([[10.0]]), p)
    slow_edge = wsa.source_speed(np.array([[0.5]]), np.array([[0.1]]), p)
    assert fast > slow_fs and fast > slow_edge
    assert fast == pytest.approx(p.v0 + p.dv / (1 + 10**0.5) ** p.alpha
                                 * (1 - 0.8 * np.exp(-(10 / p.w) ** p.delta)) ** 3)


def test_missing_longitudes_are_interpolated_cyclically():
    v = np.arange(360.0)
    v[[0, 359]] = np.nan
    out = wsa.fill_cyclic(v)
    assert np.isfinite(out).all()
    # between 358 (at 358) and 1 (at 361): 1/3 and 2/3 of the way
    assert out[359] == pytest.approx(358 - 357 / 3)
    assert out[0] == pytest.approx(358 - 2 * 357 / 3)


def test_rolling_fit_recovers_scale_and_offset():
    rng = np.random.default_rng(0)
    h = rng.normal(450, 80, 600)
    o = 50 + 0.8 * h + rng.normal(0, 5, 600)
    offset, scale = wsa.rolling_fit(h, o, ridge=0.0)
    assert scale == pytest.approx(0.8, abs=0.02) and offset == pytest.approx(50, abs=10)
    # a strong ridge keeps the prior; too few hours keep it exactly
    assert wsa.rolling_fit(h, o, ridge=1e6)[1] == pytest.approx(1.0, abs=0.01)
    assert wsa.rolling_fit(h[:10], o[:10]) == (0.0, 1.0)


def test_at_longitude_wraps():
    ring = np.arange(360.0)[None]
    assert wsa.at_longitude(ring, np.array([[359.5, 360.25, -0.5]]))[0] == pytest.approx(
        [179.5, 0.25, 179.5])


def _rings(t0s, log_fs, theta_b):
    import pandas as pd

    from feature_engineering import pfss

    ring = np.full((len(t0s), *pfss.ring_shape()), np.nan, np.float32)
    ring[:, pfss.RING_VARS.index("log_fs")] = log_fs
    ring[:, pfss.RING_VARS.index("theta_b")] = theta_b
    t0s = pd.DatetimeIndex(t0s)
    return pfss.RingTable(t0=t0s, taken=t0s - pd.Timedelta(hours=2), ring=ring,
                          scalars=np.full((len(t0s), len(pfss.SCALARS)), np.nan))


def test_wsa_table_constant_ring_gives_one_speed_at_every_lag():
    import pandas as pd

    from training import hybrid_pfss as hp

    t0s = pd.date_range("2024-03-01", periods=4, freq="6h", tz="UTC")
    out = hp.wsa_table(_rings(t0s, 1.0, 5.0), t0s, wsa.Params())
    expected = wsa.source_speed(np.array([[1.0]]), np.array([[5.0]]))[0, 0]
    assert np.allclose(out.to_numpy(), expected, atol=0.5)


def test_wsa_table_uses_only_rings_by_t0():
    import pandas as pd

    from training import hybrid_pfss as hp

    t0s = pd.date_range("2024-03-01", periods=8, freq="6h", tz="UTC")
    lon = np.arange(360.0)
    log_fs = np.stack([1.0 + 0.5 * np.sin(np.radians(lon + 20 * k)) for k in range(8)])
    rings = _rings(t0s, log_fs, 5.0)
    early = hp.wsa_table(rings, t0s[:4], wsa.Params())
    scrambled = rings.ring.copy()
    scrambled[4:] = np.random.default_rng(0).permutation(scrambled[4:].ravel()).reshape(
        scrambled[4:].shape)
    rings2 = _rings(t0s, scrambled[:, 0], 5.0)
    rings2.ring[:] = scrambled
    assert np.allclose(hp.wsa_table(rings2, t0s[:4], wsa.Params()).to_numpy(),
                       early.to_numpy(), equal_nan=True)
    assert hp.wsa_table(rings, t0s[:1] - pd.Timedelta(hours=1), wsa.Params()).isna().all().all()


def _runtime_table(rings, t0s, lags):
    from feature_engineering import build, windows

    specs = [windows.WindowSpec(windows.WSA_SIGNAL, "at", lag) for lag in lags]
    return windows.window_table(build.Inputs(pfss=rings), t0s, specs)


def test_runtime_wsa_column_matches_the_study(monkeypatch):
    # feature_engineering.windows (live) against training.hybrid_pfss.wsa_table (studies)
    import pandas as pd

    from feature_engineering import windows
    from training import hybrid_pfss as hp

    params = wsa.Params(v0=330.0, dv=600.0, alpha=0.3, w=1.5, delta=2.0, shift=4.0)
    monkeypatch.setattr(windows, "wsa_params", lambda: params)
    t0s = pd.date_range("2024-03-01", periods=8, freq="6h", tz="UTC")
    lon = np.arange(360.0)
    log_fs = np.stack([1.0 + 0.8 * np.sin(np.radians(2 * lon + 30 * k)) for k in range(8)])
    rings = _rings(t0s, log_fs, 3.0 + 2.0 * np.cos(np.radians(lon)))
    lags = (-72, -36, 0, 12, 24)
    ours = _runtime_table(rings, t0s, lags)
    study = hp.wsa_table(rings, t0s, params, lags=lags)
    np.testing.assert_allclose(ours.to_numpy(), study.to_numpy(), rtol=1e-5)
    assert np.nanstd(ours.to_numpy()) > 10  # the ring's structure reaches the column


def test_runtime_wsa_uses_only_rings_by_t0_and_needs_the_asset(monkeypatch):
    import pandas as pd

    from feature_engineering import windows

    monkeypatch.setattr(windows, "wsa_params", lambda: wsa.Params())
    t0s = pd.date_range("2024-03-01", periods=8, freq="6h", tz="UTC")
    lon = np.arange(360.0)
    log_fs = np.stack([1.0 + 0.5 * np.sin(np.radians(lon + 20 * k)) for k in range(8)])
    rings = _rings(t0s, log_fs, 5.0)
    before = _runtime_table(rings, t0s[3:4], (-24, 0, 24))
    rings.ring[4:] = np.random.default_rng(0).normal(size=rings.ring[4:].shape)
    after = _runtime_table(rings, t0s[3:4], (-24, 0, 24))
    pd.testing.assert_frame_equal(before, after)
    assert before.notna().all().all()
    half = _rings(t0s, np.where(lon < 200, np.nan, 1.0)[None].repeat(8, 0), 5.0)
    assert _runtime_table(half, t0s, (0,)).isna().all().all()  # under half a ring
    monkeypatch.setattr(windows, "wsa_params", lambda: None)  # no calibration asset
    assert _runtime_table(rings, t0s, (0,)).isna().all().all()
    assert windows.uses_pfss([windows.WindowSpec.parse("wsa_speed_at_0h")])


def test_wsa_params_load_from_the_asset(tmp_path):
    import json

    from feature_engineering import windows

    assert windows.wsa_params(tmp_path / "missing.json") is None
    path = tmp_path / "wsa.json"
    path.write_text(json.dumps({"params": wsa.Params(v0=300.0, shift=5.0).to_dict()}))
    assert windows.wsa_params(path) == wsa.Params(v0=300.0, shift=5.0)
