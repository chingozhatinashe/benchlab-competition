"""training.baselines: speed-only reference models."""

import numpy as np
import pandas as pd
import pytest

from training import baselines, datasets
from training.baselines import (
    ALL,
    Autoregressive,
    ClimatologyBlend,
    Persistence,
    Recurrence27,
    last_observed,
    window_means,
)
from training.evaluate import HORIZON, evaluate

TIMES = pd.date_range(datasets.DATA_START, datasets.EVAL_END, freq="h", inclusive="left")


def write_csv(path, value):
    pd.DataFrame({"Time": TIMES, "status": "ok", "value": value, "n_valid": 60}).to_csv(
        path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    )
    return path


def rotating_signal(seed=0, missing=300):
    """Speeds that repeat every solar rotation, with noise and missing hours."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(TIMES))
    phase = 2 * np.pi * t / baselines.ROTATION_HOURS
    value = 450 + 150 * np.sin(phase) + 80 * np.sin(3 * phase) + rng.normal(0, 5, len(t))
    value[rng.choice(len(t), missing, replace=False)] = np.nan
    return value


@pytest.fixture(scope="module")
def rotating_csv(tmp_path_factory):
    return write_csv(tmp_path_factory.mktemp("ace") / "speed_hourly.csv", rotating_signal())


def series(values):
    return pd.Series(values, index=TIMES[: len(values)], dtype=float)


def test_last_observed_skips_gaps_and_never_looks_ahead():
    values = np.array([np.nan, 1.0, np.nan, np.nan, 4.0])
    got = last_observed(values, np.array([0, 1, 2, 4, 5]))
    np.testing.assert_array_equal(got, [np.nan, np.nan, 1.0, 1.0, 4.0])


def test_window_means_ignore_nan_and_out_of_range():
    values = np.array([1.0, np.nan, 3.0, 5.0])
    got = window_means(values, np.array([[1, 3, -5]]), 1)
    np.testing.assert_allclose(got, [[2.0, 4.0, np.nan]])


@pytest.mark.parametrize("cls", [ClimatologyBlend, Recurrence27, Autoregressive])
def test_predict_features_match_fit_features(cls):
    values = rotating_signal()[:5000]
    model = cls().fit(series(values))
    p = np.array([4321])
    at_fit = model.features(values, p)
    at_predict = model.features(values[:4321], p)
    if not isinstance(at_fit, tuple):
        at_fit, at_predict = (at_fit,), (at_predict,)
    for a, b in zip(at_fit, at_predict, strict=True):
        np.testing.assert_allclose(a, b)


def test_climatology_weights_are_bounded_and_near_zero_for_noise():
    rng = np.random.default_rng(1)
    model = ClimatologyBlend().fit(series(450 + rng.normal(0, 50, 20000)))
    assert ((model.weights >= 0) & (model.weights <= 1)).all()
    assert model.weights.max() < 0.05


def test_climatology_weight_decays_with_lead_on_persistent_signal():
    model = ClimatologyBlend().fit(series(rotating_signal()[:20000]))
    assert model.weights[0] > model.weights[-1]
    assert model.weights[0] > 0.9


def test_autoregressive_handles_gappy_history():
    values = rotating_signal()[:20000]
    model = Autoregressive().fit(series(values))
    history = values.copy()
    history[-30:] = np.nan
    forecast = model.predict(series(history), TIMES[len(history)])
    assert forecast.shape == (HORIZON,)
    assert np.isfinite(forecast).all()


def test_recurrence_learns_a_rotating_signal(rotating_csv):
    recur = evaluate(Recurrence27(), rotating_csv)
    persist = evaluate(Persistence(), rotating_csv)
    assert recur["timse"] < 0.05 * persist["timse"]
    assert recur["rmse"] < 15


@pytest.mark.parametrize("name", list(ALL))
def test_every_baseline_runs_through_the_harness(rotating_csv, name):
    result = evaluate(ALL[name](), rotating_csv)
    assert np.isfinite(result["timse"]) and result["timse"] > 0
    assert np.isfinite(result["rmse"])
