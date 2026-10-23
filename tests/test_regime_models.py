"""The regime study's labelling and mixing (training.regime_models)."""

import numpy as np
import pandas as pd

from training import regime_models as rm
from training import solar_wind_events as ev


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    speed = 420 + 150 * np.sin(np.arange(n) / 40) + rng.normal(0, 20, n)
    speed[rng.random(n) < 0.05] = np.nan
    cme = pd.Series(False, index=index)
    cme.iloc[100:130] = True
    possible = pd.Series(False, index=index)
    possible.iloc[300:320] = True
    arrivals = index[[200, 450]]
    return index, speed, cme, possible, arrivals


def test_label_runs_matches_classify_runs():
    index, speed, cme, possible, arrivals = _synthetic()
    pos = np.arange(0, len(index) - 10)
    actual = np.full((len(pos), 72), np.nan)
    for i, p in enumerate(pos):
        window = speed[p + 1 : p + 73]
        actual[i, : len(window)] = window
    expected = ev.classify_runs(index[pos], actual, cme, arrivals, possible)
    got = ev.label_runs(pos, speed, cme, possible, arrivals)
    assert list(got) == list(expected)
    assert set(got) >= {"CME", "possible CME", "stream arrival"}


def test_routes_cover_every_regime():
    assert set(ev.ROUTE_OF) == set(ev.REGIMES)
    assert set(ev.ROUTE_OF.values()) == set(ev.ROUTES)


def test_one_hot_mixing_picks_the_member():
    rng = np.random.default_rng(1)
    members = rng.normal(400, 50, (len(rm.ROUTES), 7, 72))
    route = rng.integers(0, len(rm.ROUTES), 7)
    mixed = rm.mix(rm.one_hot(route), members)
    np.testing.assert_allclose(mixed, members[route, np.arange(7)])


def test_soft_mixing_is_a_convex_combination():
    rng = np.random.default_rng(2)
    members = rng.normal(400, 50, (len(rm.ROUTES), 5, 72))
    p = rng.random((5, len(rm.ROUTES)))
    p /= p.sum(axis=1, keepdims=True)
    mixed = rm.mix(p, members)
    assert (mixed <= members.max(axis=0) + 1e-9).all()
    assert (mixed >= members.min(axis=0) - 1e-9).all()
