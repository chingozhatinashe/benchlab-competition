"""models.final_baseline: the reference baseline that ships in the image."""

import json

import numpy as np
import pandas as pd
import pytest

from feature_engineering.recurrence import check_lags, design, targets
from models.final_baseline import (
    ASSET,
    HORIZON,
    Config,
    FinalBaseline,
    TailWeights,
    cycle_positions,
    run_timse_weights,
)
from training import final_baseline as training_fb
from training.evaluate import evaluate
from training.metrics import timse

ROTATION = 27 * 24
START = pd.Timestamp("2021-10-20T00:00Z")


def rotating(n, seed=0, missing=200, noise=10.0):
    """Hourly speeds that repeat every rotation, with a sub-rotation stream and gaps."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    phase = 2 * np.pi * t / ROTATION
    v = 450 + 120 * np.sin(phase) + 60 * np.sin(2 * phase) + rng.normal(0, noise, n)
    v[rng.choice(n, missing, replace=False)] = np.nan
    return pd.Series(v, index=pd.date_range(START, periods=n, freq="h"), dtype=float)


SMALL = Config(window_days=20, rolling_weight=0.5, tail_power=0.5, stack_days=60)


@pytest.fixture(scope="module")
def series():
    return rotating(24 * 300)


@pytest.fixture(scope="module")
def model(series):
    return FinalBaseline(SMALL).fit(series.iloc[: 24 * 240])


# ------------------------------------------------------------------ loss weights
@pytest.mark.parametrize("decimals", [None, -1])  # -1 rounds to tens, so there are ties
def test_run_timse_weights_reproduce_timse_exactly(decimals):
    rng = np.random.default_rng(1)
    y = rng.normal(450, 80, (50, HORIZON))
    if decimals is not None:
        y = y.round(decimals)
    y[rng.random(y.shape) < 0.1] = np.nan
    y[0, 1:] = np.nan  # a single observation scores 0
    y[1, :] = 500.0  # all tied
    f = np.where(np.isnan(y), 400.0, y + rng.normal(0, 60, y.shape))
    w = run_timse_weights(y)
    for row in range(len(y)):
        ok = np.isfinite(y[row])
        got = np.sum(w[row, ok] * (y[row, ok] - f[row, ok]) ** 2)
        want = timse(y[row], f[row]) if ok.sum() >= 2 else 0.0
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9)
    assert (w[np.isnan(y)] == 0).all()


def test_pooled_tail_weights_grow_with_speed_and_average_one():
    speeds = np.random.default_rng(2).gamma(20, 22, 20000)
    tail = TailWeights.fit(speeds)
    g = tail(np.sort(speeds), 1.0)
    assert np.all(np.diff(g) >= 0)
    assert g.mean() == pytest.approx(1.0, rel=1e-3)
    assert tail(np.array([np.nan, 500.0]), 0.5)[0] == 0
    np.testing.assert_array_equal(tail(np.array([300.0, np.nan]), 0.0), [1.0, 0.0])


# ------------------------------------------------------------------ causality
def test_features_never_reach_t0():
    with pytest.raises(ValueError):
        check_lags([HORIZON + 3], 3)  # window would include t0 at the longest lead
    values = rotating(3000).to_numpy()
    p = np.array([2500])
    base = design(values, p, 450.0, (648, 324, 216), 3)
    changed = values.copy()
    changed[2500:] = 9999.0
    np.testing.assert_array_equal(design(changed, p, 450.0, (648, 324, 216), 3), base)


def test_forecast_ignores_everything_from_t0_on(model, series):
    values = series.to_numpy()
    p = 24 * 270
    expected = model.predict(series.iloc[:p])
    changed = values.copy()
    changed[p:] = 2000.0
    np.testing.assert_array_equal(model.forecast(changed, [p])[0], expected)
    np.testing.assert_allclose(model.forecast(values, [p])[0], expected, atol=1e-9)


def test_history_hours_is_enough_history(model, series):
    p = 24 * 280
    full = model.predict(series.iloc[:p])
    trimmed = model.predict(series.iloc[p - model.config.history_hours : p])
    np.testing.assert_allclose(trimmed, full, atol=1e-9)


def test_walk_forward_matches_one_call_per_t0(model, series):
    values = series.to_numpy()
    ps = cycle_positions(series.index, series.index[24 * 250], series.index[24 * 252])
    batch = model.forecast(values, ps)
    for i, p in enumerate(ps):
        np.testing.assert_allclose(batch[i], model.predict(series.iloc[:p]), atol=1e-9)


def test_rolling_targets_stop_before_t0():
    y = targets(np.arange(100.0), np.array([10, 50]), end=60)
    assert np.isnan(y[1, 9:]).all() and y[1, 8] == 59.0
    assert y[0, 0] == 11.0


# ------------------------------------------------------------------ behaviour
def test_learns_the_rotation_and_beats_persistence(model, series):
    values = series.to_numpy()
    ps = cycle_positions(series.index, series.index[24 * 245], series.index[-1])
    forecasts = model.forecast(values, ps)
    y = targets(values, ps)
    last = pd.Series(values).ffill().to_numpy()[ps - 1]
    persistence = np.repeat(last[:, None], HORIZON, axis=1)
    assert timse(y, forecasts) < 0.3 * timse(y, persistence)
    ok = np.isfinite(y)
    assert np.sqrt(np.mean((forecasts[ok] - y[ok]) ** 2)) < 30


def test_rolling_weight_moves_between_frozen_and_rolling(series):
    values, p = series.to_numpy(), 24 * 270
    train = series.iloc[: 24 * 240]
    out = {}
    for lam in (0.0, 0.5, 1.0):
        cfg = Config(window_days=20, rolling_weight=lam, blend_persistence=False)
        out[lam] = FinalBaseline(cfg).fit(train).forecast(values, [p])[0]
    np.testing.assert_allclose(out[0.5], (out[0.0] + out[1.0]) / 2, atol=1e-6)
    assert not np.allclose(out[0.0], out[1.0])


def test_persistence_weights_are_fitted_and_bounded(model, series):
    assert ((model.persistence_weight >= 0) & (model.persistence_weight <= 1)).all()
    off = FinalBaseline(Config(blend_persistence=False)).fit(series.iloc[: 24 * 240])
    assert (off.persistence_weight == 0).all()


def test_empty_history_gives_a_finite_forecast(model):
    out = model.predict(pd.Series([], dtype=float))
    assert out.shape == (HORIZON,) and np.isfinite(out).all()
    out = model.predict(pd.Series(np.nan, index=range(500)))
    assert np.isfinite(out).all()


def test_config_validation():
    with pytest.raises(ValueError):
        Config(rolling_weight=1.5)
    with pytest.raises(ValueError):
        Config(loss="mae")
    assert Config(lags_days=[27, 13.5]).lag_hours == (648, 324)


def test_runs_through_the_eval_harness(tmp_path):
    from training import datasets

    times = pd.date_range(datasets.DATA_START, datasets.EVAL_END, freq="h", inclusive="left")
    values = rotating(len(times)).to_numpy()
    path = tmp_path / "speed_hourly.csv"
    pd.DataFrame({"Time": times, "status": "ok", "value": values, "n_valid": 60}).to_csv(
        path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    )
    result = evaluate(FinalBaseline(Config(window_days=14)), path)
    assert np.isfinite(result["timse"]) and result["rmse"] < 40


# ------------------------------------------------------------------ the shipped asset
def test_save_and_load_round_trip(model, series, tmp_path):
    path = tmp_path / "m.json"
    model.save(path)
    loaded = FinalBaseline.load(path)
    assert loaded.config == model.config
    history = series.iloc[: 24 * 270]
    np.testing.assert_allclose(loaded.predict(history), model.predict(history), atol=1e-6)


def test_shipped_asset_matches_the_tuned_config():
    shipped = FinalBaseline.load(ASSET)
    assert shipped.config == training_fb.load_config()
    assert shipped.coef.shape == (HORIZON, shipped.config.n_features)
    assert np.isfinite(shipped.coef).all()
    info = json.loads(ASSET.read_text())["fit_info"]
    assert info["end"] <= "2025-10-20T00:00Z"  # fitted on the training years only
