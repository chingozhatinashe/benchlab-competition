"""training.lead_aligned_ch: the lead-to-lag map and the stitched linear model."""

import numpy as np

from models.linear_model import Config, LinearModel
from training.lead_aligned_ch import Stitched, lead_lags


def test_lead_lags_follow_the_delay_on_the_image_grid():
    lags = lead_lags(78)
    assert lags[0] == 78 and lags[-1] == 6  # lead 1: 77 h -> 78; lead 72: 6 h
    assert np.all(lags % 6 == 0) and np.all(np.diff(lags) <= 0)
    assert np.all(78 - np.arange(1, 73) <= lags)  # never newer than the delay asks
    assert lead_lags(60)[-1] == 0  # far leads take the latest image


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n = 24 * 120
    speed = 420 + 80 * np.sin(2 * np.pi * np.arange(n) / 648) + rng.normal(0, 20, n)
    positions = np.arange(700, n - 80, 6)
    names = ("a", "ch_cm_at_0h", "ch_cm_at_6h")
    x = rng.normal(0, 1, (len(positions), 3))
    return speed, positions, names, x


def test_stitched_with_one_lag_equals_a_plain_linear_model():
    speed, positions, names, x = _data()
    cfg = Config(features=names[:2], lags_days=(27.0,), ridge=0.1)
    plain = LinearModel(cfg).fit(speed, positions, x[:, :2])
    st = Stitched(cfg, ("a",), np.zeros(72, dtype=int)).fit(speed, positions, x[:, :2])
    assert st.config.features == ("a", "ch_cm_at_0h")
    assert np.allclose(st.forecast(speed, positions[:30], x[:30, :2]),
                       plain.forecast(speed, positions[:30], x[:30, :2]))


def test_stitched_takes_each_lead_from_its_lag_model():
    speed, positions, names, x = _data(1)
    cfg = Config(features=(), lags_days=(27.0,), ridge=0.1)
    lags = np.where(np.arange(72) < 36, 6, 0)
    st = Stitched(cfg, ("a",), lags).fit(speed, positions, x)
    assert st.config.features == ("a", "ch_cm_at_0h", "ch_cm_at_6h")
    f = st.forecast(speed, positions[:30], x[:30])
    for lag, cols, leads in ((6, [0, 2], slice(0, 36)), (0, [0, 1], slice(36, 72))):
        m = LinearModel(Config(features=("a", f"ch_cm_at_{lag}h"), lags_days=(27.0,),
                               ridge=0.1, optional=(f"ch_cm_at_{lag}h",)))
        m.fit(speed, positions, x[:, cols])
        assert np.allclose(f[:, leads], m.forecast(speed, positions[:30], x[:30, cols])[:, leads])
