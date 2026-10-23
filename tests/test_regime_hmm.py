"""models.regime_hmm: filtering, causality, EM and the asset round trip."""

import numpy as np
import pandas as pd
import pytest

from models.regime_hmm import Config, RegimeHMM, observations, previous_speed


def _hourly(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC", name="Time")
    level = np.where((np.arange(n) // 150) % 2 == 0, 350.0, 600.0)
    speed = level + rng.normal(0, 20, n)
    speed[rng.random(n) < 0.05] = np.nan
    ratio = rng.normal(0.0, 0.3, n)
    bt = np.exp(rng.normal(1.6, 0.3, n))
    return pd.DataFrame({"speed": speed, "log_t_ratio": ratio, "bt": bt}, index=index)


@pytest.fixture(scope="module", params=[False, True], ids=["level", "ar"])
def fitted(request):
    hourly = _hourly()
    return hourly, RegimeHMM(Config(autoregressive=request.param, max_iter=15)).fit(
        observations(hourly))


def test_filter_is_a_distribution(fitted):
    hourly, model = fitted
    p = model.filter(observations(hourly))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_em_does_not_decrease_likelihood(fitted):
    _, model = fitted
    assert np.all(np.diff(model.history) > -1e-6)


def test_features_use_only_rows_before_t0(fitted):
    hourly, model = fitted
    t0s = hourly.index[[400, 800]]
    base = model.features(hourly, t0s)
    for t0 in t0s:
        scrambled = hourly.copy()
        later = scrambled.index >= t0
        scrambled.loc[later] = scrambled.loc[later].to_numpy()[::-1] * 1.7
        np.testing.assert_allclose(model.features(scrambled, [t0]).to_numpy(),
                                   base.loc[[t0]].to_numpy())


def test_round_trip(fitted):
    hourly, model = fitted
    again = RegimeHMM.from_dict(model.to_dict())
    np.testing.assert_allclose(again.filter(observations(hourly)),
                               model.filter(observations(hourly)))


def test_previous_speed_respects_the_gap():
    v = np.array([400.0, np.nan, np.nan, 500.0])
    np.testing.assert_allclose(previous_speed(v, max_gap=2), [np.nan, 400, 400, np.nan])
