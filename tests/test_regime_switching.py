"""models.regime_switching: gating, no future data, round trip; the HMM's own forecast."""

import numpy as np
import pandas as pd
import pytest

from models.regime_hmm import Config as HMMConfig
from models.regime_hmm import RegimeHMM, observations
from models.regime_switching import Config, RegimeSwitchingModel


def _data(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC", name="Time")
    level = np.where((np.arange(n) // 150) % 2 == 0, 350.0, 600.0)
    speed = level + rng.normal(0, 20, n)
    speed[rng.random(n) < 0.03] = np.nan
    insitu = pd.DataFrame({"speed": speed, "log_t_ratio": rng.normal(0, 0.3, n),
                           "bt": np.exp(rng.normal(1.6, 0.3, n))}, index=index)
    return index, speed, insitu


@pytest.fixture(scope="module", params=[1, 3], ids=["K1", "K3"])
def fitted(request):
    index, speed, insitu = _data()
    model = RegimeSwitchingModel(Config(n_states=request.param, lags_days=(27.0, 13.5)))
    model.fit_hmm(insitu)
    pos = np.arange(700, 2200, 6)
    model.fit(speed, pos, model.gate(insitu, index[pos]))
    return index, speed, insitu, model


def test_gate_is_a_distribution(fitted):
    index, _, insitu, model = fitted
    g = model.gate(insitu, index[[0, 500, 2000]])  # the first has no rows before it
    np.testing.assert_allclose(g.sum(axis=1), 1.0)


def test_soft_and_hard_agree_with_one_state():
    index, speed, insitu = _data()
    model = RegimeSwitchingModel(Config(n_states=1)).fit_hmm(insitu)
    pos = np.arange(700, 2200, 6)
    g = model.gate(insitu, index[pos])
    model.fit(speed, pos, g)
    np.testing.assert_allclose(model.forecast(speed, pos, g),
                               model.forecast(speed, pos, g, hard=True))


def test_no_future_data(fitted):
    index, speed, insitu, model = fitted
    p = 1800
    t0 = index[p]
    base = model.forecast(speed, np.array([p]), model.gate(insitu, [t0]))
    s2, i2 = speed.copy(), insitu.copy()
    s2[p:] = s2[p:][::-1] * 1.5
    i2.loc[i2.index >= t0] = i2.loc[i2.index >= t0].to_numpy()[::-1] * 1.3
    again = model.forecast(s2, np.array([p]), model.gate(i2, [t0]))
    np.testing.assert_allclose(again, base)


def test_round_trip(fitted):
    index, speed, insitu, model = fitted
    pos = np.arange(1900, 2300, 6)
    g = model.gate(insitu, index[pos])
    again = RegimeSwitchingModel.from_dict(model.to_dict())
    np.testing.assert_allclose(again.forecast(speed, pos, again.gate(insitu, index[pos])),
                               model.forecast(speed, pos, g))


@pytest.mark.parametrize("ar", [False, True])
def test_hmm_forecast_tends_to_the_stationary_mean(ar):
    _, _, insitu = _data()
    hmm = RegimeHMM(HMMConfig(autoregressive=ar, max_iter=10)).fit(observations(insitu))
    p = hmm.filter(observations(insitu))[[100, 1000]]
    f = hmm.mean_forecast(p, np.array([300.0, 700.0]), horizon=5000)
    level = hmm.c / (1 - hmm.phi) if ar else hmm.m
    np.testing.assert_allclose(f[:, -1], hmm.start @ level, rtol=0.02)
