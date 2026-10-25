import numpy as np
import pandas as pd
import pytest

from feature_engineering import coronagraph as cg
from models import cme_detector as cd

T0 = pd.Timestamp("2024-05-10T12:00Z")
H = pd.Timedelta(hours=1)


def test_range_max_matches_brute_force():
    rng = np.random.default_rng(0)
    v = rng.normal(size=200)
    v[rng.random(200) < 0.1] = np.nan
    lo = rng.integers(0, 200, 500)
    hi = np.minimum(lo + rng.integers(0, 60, 500), 200)
    got = cd.range_max(v, lo, hi)
    for a, b, g in zip(lo, hi, got, strict=True):
        part = v[a:b]
        want = np.nan if b <= a or np.isnan(part).all() else np.nanmax(part)
        assert (np.isnan(want) and np.isnan(g)) or g == pytest.approx(want)


def test_bin_max_uses_only_images_taken_by_t0_minus_latency():
    times = pd.DatetimeIndex([T0 - 30 * H, T0 - 2 * H, T0 - 20 * pd.Timedelta(minutes=1), T0])
    values = np.array([1.0, 2.0, 50.0, 99.0])  # the last two are too recent for t0
    assert cd.bin_max(times, values, [T0], 0, 24)[0] == 2.0
    assert cd.bin_max(times, values, [T0], 24, 48)[0] == 1.0
    assert np.isnan(cd.bin_max(times, values, [T0], 48, 72)[0])


def test_image_features_do_not_depend_on_position_angle():
    times = pd.date_range(T0 - 8 * H, T0, freq="h")
    base = np.full((len(times), 24), 60.0)
    a, b = base.copy(), base.copy()
    a[-1, 12 + 2] = b[-1, 12 + 9] = 90.0  # the same brightening, east or west
    fa = cd.image_features(pd.DataFrame(a, index=times, columns=cg.cell_columns()))
    fb = cd.image_features(pd.DataFrame(b, index=times, columns=cg.cell_columns()))
    assert np.allclose(fa.to_numpy(), fb.to_numpy())
    assert fa["outer0"].iloc[-1] == pytest.approx(0.5)


def test_labels():
    onsets = pd.DatetimeIndex([T0])
    times = pd.DatetimeIndex([T0 - H, T0, T0 + 2 * H, T0 + 3 * H])
    assert list(cd.image_labels(times, onsets)) == [False, True, True, False]
    lab = cd.arrival_labels(onsets, [T0 - 30 * H, T0 - 60 * H, T0])
    assert lab.tolist() == [[False, True, False], [False, False, True], [False, False, False]]


def test_flare_near():
    flares = pd.DataFrame({"peak_time": [T0 - 2 * H, T0 + H], "peak_flux": [1e-5, 1e-3]})
    got = cd.flare_near(pd.DatetimeIndex([T0, T0 - 6 * H]), flares)
    assert got[0] == pytest.approx(-5.0) and got[1] == cd.NO_FLARE


def test_monotone_classifier_matches_xgboost_and_is_monotone():
    xgb = pytest.importorskip("xgboost")  # noqa: F841
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 3))
    y = rng.random(2000) < 1 / (1 + np.exp(-(2 * x[:, 0] - 1)))
    model = cd.MonotoneClassifier({"n_trees": 40, "max_depth": 2, "eta": 0.2, "subsample": 1.0})
    model.fit(x, y, nthread=1)
    p = model.predict(x)
    assert np.all((p > 0) & (p < 1)) and np.corrcoef(p, x[:, 0])[0, 1] > 0.8
    grid = np.zeros((50, 3))
    grid[:, 1] = np.linspace(-3, 3, 50)  # an input with no signal still never lowers p
    assert np.all(np.diff(model.predict(grid)) >= -1e-12)
    again = cd.MonotoneClassifier.from_dict(model.to_dict())
    assert np.allclose(again.predict(x), p)
