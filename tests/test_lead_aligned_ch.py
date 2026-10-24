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


# ---------------------------------------------------------------------- LinearModel support
def _lagged(seed=2, nan_share=0.0):
    """Speed plus one regressor "a" and ch_cm at every lag the 72 h delay uses."""
    from models.linear_model import lead_lags

    rng = np.random.default_rng(seed)
    n = 24 * 120
    speed = 420 + 80 * np.sin(2 * np.pi * np.arange(n) / 648) + rng.normal(0, 20, n)
    positions = np.arange(700, n - 80, 6)
    lags = np.unique(lead_lags(72))
    names = ("a", *(f"ch_cm_at_{lag}h" for lag in lags))
    x = rng.normal(0, 1, (len(positions), len(names)))
    if nan_share:
        x[rng.random(x.shape) < nan_share] = np.nan
        x[:, 0] = rng.normal(0, 1, len(positions))
    return speed, positions, names, x


def test_lead_feature_expands_to_its_lag_columns():
    from models.linear_model import lead_feature

    cfg = Config(features=("a", "ch_cm_lead_72h"), lags_days=(27.0,))
    assert lead_feature("ch_cm_lead_72h") == ("ch_cm", 72) and lead_feature("ch_cm_at_24h") is None
    assert cfg.input_features == ("a", *(f"ch_cm_at_{lag}h" for lag in range(0, 73, 6)))
    cols = cfg.lead_columns("ch_cm_lead_72h")
    assert cfg.input_features[cols[0]] == "ch_cm_at_72h"  # lead 1: 71 h -> 72
    assert cfg.input_features[cols[-1]] == "ch_cm_at_0h"
    assert cfg.n_columns == 1 + 2 + 2


def test_lead_aligned_model_equals_the_stitched_one():
    from models.linear_model import lead_lags

    speed, positions, names, x = _lagged()
    cfg = Config(features=("a", "ch_cm_lead_72h"), lags_days=(27.0,), ridge=0.1,
                 optional=("ch_cm_lead_72h",))
    model = LinearModel(cfg).fit(speed, positions, x)
    st = Stitched(Config(features=("a",), lags_days=(27.0,), ridge=0.1), ("a",), lead_lags(72))
    xs = x[:, [names.index(c) for c in st.config.features]]
    st.fit(speed, positions, xs)
    assert np.allclose(model.forecast(speed, positions[:40], x[:40]),
                       st.forecast(speed, positions[:40], xs[:40]))


def test_missing_lags_are_filled_from_the_nearest_image():
    speed, positions, names, x = _lagged()
    cfg = Config(features=("a", "ch_cm_lead_72h"), lags_days=(27.0,),
                 optional=("ch_cm_lead_72h",))
    model = LinearModel(cfg)
    row = x[:1].copy()
    i30, i24, i36 = (names.index(f"ch_cm_at_{h}h") for h in (30, 24, 36))
    row[0, i30] = np.nan
    filled = model.fill_leads(row)
    assert filled[0, i30] == row[0, i36]  # tie between 24 h and 36 h: the older image
    gap = x[:1].copy()
    gap[0, 1:] = np.nan
    gap[0, names.index("ch_cm_at_0h")] = 0.5
    filled = model.fill_leads(gap)
    assert filled[0, names.index("ch_cm_at_24h")] == 0.5  # within 24 h
    assert np.isnan(filled[0, names.index("ch_cm_at_30h")])  # beyond 24 h


def test_lead_aligned_model_falls_back_round_trips_and_predicts(tmp_path):
    speed, positions, names, x = _lagged(nan_share=0.05)
    cfg = Config(features=("a", "ch_cm_lead_72h"), lags_days=(27.0,),
                 optional=("ch_cm_lead_72h",), cap_quantile=0.005, despike=True)
    model = LinearModel(cfg).fit(speed, positions, x)
    assert model.fallback.config.input_features == ("a",)
    no_ch = x[:5].copy()
    no_ch[:, 1:] = np.nan
    assert model.uses_fallback(no_ch).all()
    assert np.allclose(model.forecast(speed, positions[:5], no_ch),
                       model.fallback.forecast(speed, positions[:5], no_ch[:, :1]))
    model.save(tmp_path / "m.json")
    again = LinearModel.load(tmp_path / "m.json")
    assert np.allclose(again.forecast(speed, positions[:20], x[:20]),
                       model.forecast(speed, positions[:20], x[:20]), equal_nan=True)
    p = positions[10]
    row = dict(zip(names, x[10], strict=True))
    assert np.allclose(model.predict(speed[:p], row),
                       model.forecast(speed, positions[10:11], x[10:11])[0])
