import numpy as np

from training import model_diagnostics as md


def test_hour_regimes_cme_first_then_speed_bands():
    obs = np.array([[300.0, 449.9, 450.0, 549.9, 550.0, np.nan, 700.0]])
    cme = np.array([[False, False, False, False, False, False, True]])
    got = md.hour_regimes(obs, cme)
    assert got.tolist() == [["slow", "slow", "intermediate", "intermediate", "fast", "", "CME"]]


def test_best_shift_finds_a_known_shift():
    k = np.arange(72)
    forecast = 400 + 200 / (1 + np.exp(-(k - 30) / 2.0))  # a stream arriving at lead 30
    late = 400 + 200 / (1 + np.exp(-(k - 36) / 2.0))  # observed 6 h later
    early = 400 + 200 / (1 + np.exp(-(k - 22) / 2.0))  # observed 8 h earlier
    got = md.best_shift(np.vstack([forecast, forecast, forecast]),
                        np.vstack([late, early, forecast]))
    assert got.tolist() == [6.0, -8.0, 0.0]


def test_best_shift_needs_enough_valid_pairs():
    f = np.full((1, 72), 400.0)
    y = np.full((1, 72), np.nan)
    y[0, :10] = 400.0
    assert np.isnan(md.best_shift(f, y)[0])


def test_drop_configs_remove_one_input_each():
    from training import linear_model as tl

    cfg = tl.with_fixes(tl.load_config())
    drops = md.drop_configs(cfg)
    for f in cfg.features:
        assert f not in drops[f].features and len(drops[f].features) == len(cfg.features) - 1
        assert set(drops[f].optional) <= set(drops[f].features)
    assert drops["both recurrence terms"].lags_days == ()
    assert drops["both recurrence terms"].features == cfg.features
