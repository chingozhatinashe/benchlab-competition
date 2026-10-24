"""models.xgboost_model: numpy trees vs xgboost, pooled rows, the hybrid, the live path."""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import build, windows
from feature_engineering.recurrence import HORIZON, LEADS, last_observed, window_means
from models.linear_model import Config as LinearConfig
from models.linear_model import LinearModel
from models.xgboost_model import Config, TreeEnsemble, XGBoostModel

xgb = pytest.importorskip("xgboost")

T0 = pd.Timestamp("2026-03-15T12:00Z")
H = pd.Timedelta(hours=1)


def synthetic(n_hours=24 * 150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    driver = rng.normal(0, 1, n_hours + 48)
    speed = 420 + 80 * np.sin(2 * np.pi * t / 648) + 40 * driver[24 : n_hours + 24]
    speed += rng.normal(0, 15, n_hours)
    positions = np.arange(800, n_hours - 80)  # hourly runs
    x = np.column_stack([driver[positions + 24], rng.normal(0, 1, len(positions))])
    x[::9, 1] = np.nan
    return speed, positions, x


def hybrid(speed, positions, x, **kw):
    lin = LinearModel(LinearConfig(features=("a",), relative=True, lags_days=(27.0, 13.5),
                                   despike=True)).fit(speed, positions[::6], x[::6, :1])
    cfg = Config(features=("b",), base="linear", n_trees=30, max_depth=3, **kw)
    model = XGBoostModel(cfg, lin)
    assert model.input_features == ("b", "a")
    return model.fit(speed, positions, x[:, ::-1])  # columns in input_features order


# ---------------------------------------------------------------------- numpy trees
def test_numpy_trees_match_xgboost_with_missing_values():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 5)).astype(np.float32)
    x[rng.random(x.shape) < 0.15] = np.nan
    y = np.nan_to_num(x[:, 0]) * 3 + np.nan_to_num(x[:, 1]) ** 2 + rng.normal(size=3000)
    booster = xgb.train({"max_depth": 5, "eta": 0.1, "nthread": 2, "max_bin": 64},
                        xgb.QuantileDMatrix(x, y, max_bin=64), num_boost_round=80)
    trees = TreeEnsemble.from_booster(booster)
    ref = booster.predict(xgb.DMatrix(x))
    assert np.allclose(trees.predict(x), ref, atol=1e-4)
    first = TreeEnsemble.from_booster(booster, 20)
    assert np.allclose(first.predict(x), booster.predict(xgb.DMatrix(x), iteration_range=(0, 20)),
                       atol=1e-4)
    again = TreeEnsemble.from_dict(trees.to_dict())
    assert np.array_equal(again.predict(x), trees.predict(x))


# ---------------------------------------------------------------------- rows and targets
def test_rows_hold_lead_last_recurrence_and_regressors():
    speed, positions, x = synthetic()
    model = XGBoostModel(Config(features=("a", "b"), despike=False))
    p = positions[100:103]
    rows, last = model.rows(speed, p, x[100:103])
    rows = rows.reshape(3, HORIZON, -1)
    assert np.array_equal(rows[:, :, 0], np.broadcast_to(LEADS, (3, HORIZON)))
    assert np.allclose(last, last_observed(speed, p))
    assert np.allclose(rows[:, :, 1], last[:, None])
    rec = window_means(speed, p[:, None] + LEADS - 648, 3)
    assert np.allclose(rows[:, :, 2], rec, rtol=1e-6)
    assert np.allclose(rows[:, 5, 4:], x[100:103], equal_nan=True)


@pytest.mark.parametrize("base", ["persistence", "linear"])
def test_targets_are_corrections_to_the_anchor(base):
    speed, positions, x = synthetic()
    if base == "linear":
        model = hybrid(speed, positions, x)
        reg = x[:, ::-1]
    else:
        model = XGBoostModel(Config(features=("a", "b"), n_trees=5, despike=False))
        reg = x
    rows, y, w = model.training_set(speed, positions[:50], reg[:50])
    anchor = model.anchor(speed, positions[:50], reg[:50])
    lead = rows[:, 0].astype(int)
    run = np.repeat(np.arange(50), model.config.leads_per_run)
    keep = np.isfinite(speed[positions[:50][run] + lead])  # rows dropped only where missing
    assert len(y) == keep.sum() and (w > 0).all()
    expected = speed[positions[:50][run] + lead] - anchor[run, lead - 1]
    assert np.allclose(y, expected[keep], atol=1e-3)


# ---------------------------------------------------------------------- forecasting
@pytest.mark.parametrize("base", ["persistence", "linear"])
def test_forecast_uses_no_future_hours(base):
    speed, positions, x = synthetic()
    if base == "linear":
        model, reg = hybrid(speed, positions, x), x[:, ::-1]
    else:
        model = XGBoostModel(Config(features=("a", "b"), n_trees=20)).fit(speed, positions, x)
        reg = x
    model.booster = None
    i = len(positions) - 200
    p = positions[i]
    full = model.forecast(speed, [p], reg[i : i + 1])
    scrambled = speed.copy()
    scrambled[p:] = np.random.default_rng(5).uniform(250, 1200, len(speed) - p)
    assert np.allclose(model.forecast(scrambled, [p], reg[i : i + 1]), full)


def test_hybrid_booster_numpy_predict_and_json_agree(tmp_path):
    speed, positions, x = synthetic()
    model = hybrid(speed, positions, x)
    reg = x[:, ::-1]
    p = positions[-300:-290]
    with_booster = model.forecast(speed, p, reg[-300:-290])
    model.booster = None
    numpy = model.forecast(speed, p, reg[-300:-290])
    assert np.allclose(with_booster, numpy, atol=1e-3)
    model.save(tmp_path / "m.json")
    again = XGBoostModel.load(tmp_path / "m.json")
    assert again.base_model is not None
    assert np.allclose(again.forecast(speed, p, reg[-300:-290]), numpy)
    i = len(positions) - 300
    row = {"a": x[i, 0], "b": x[i, 1]}
    assert np.allclose(again.predict(speed[: positions[i]], row), numpy[0])


def test_missing_history_and_regressors_still_give_finite_forecasts():
    speed, positions, x = synthetic()
    model = hybrid(speed, positions, x)
    model.booster = None
    out = model.predict(np.full(800, np.nan), {})
    assert out.shape == (HORIZON,) and np.isfinite(out).all()


# ---------------------------------------------------------------------- coronal holes at hourly t0s
def test_coronal_hole_strip_at_hourly_t0_uses_the_latest_image():
    cut = T0 - build.CH_LATENCY
    index = pd.date_range(cut - pd.Timedelta(hours=48), cut, freq="6h")
    ch = pd.DataFrame({"ch_cm": np.arange(len(index), dtype=float), "ch_east": 1.0,
                       "ch_disk": 2.0}, index=index)
    inputs = build.Inputs(coronal_holes=ch)
    spec = windows.WindowSpec("ch_cm", "at", 12)
    on_grid = windows.window_table(inputs, [T0], [spec]).iloc[0, 0]
    later = windows.window_table(inputs, [T0 + 5 * H], [spec]).iloc[0, 0]
    assert later == on_grid  # 5 h after the grid: the same image
    east = windows.window_table(inputs, [T0], [windows.WindowSpec("ch_east", "at", 0)])
    assert east.iloc[0, 0] == 1.0
    assert windows.ch_lags([windows.WindowSpec("ch_east", "at", 24)]) == [24, 30, 36]


# ---------------------------------------------------------------------- live path
def test_shipped_asset_is_a_hybrid_over_real_regressors():
    model = XGBoostModel.load()
    assert model.config.base == "linear" and model.base_model is not None
    names = {s.name for s in windows.candidates()} | {
        "ch_cm_at_0h", "ch_cm_at_24h", "ch_cm_at_48h", "ch_cm_at_72h", "ch_east_at_0h",
        "ch_east_at_24h", "ch_disk_at_0h"}
    assert set(model.input_features) <= names
    hours = pd.date_range(T0 - pd.Timedelta(days=40), T0, freq="h", inclusive="left")
    out = model.predict(pd.Series(450.0, index=hours), {})
    assert out.shape == (HORIZON,) and np.isfinite(out).all()


def test_xgboost_live_path_offline(monkeypatch):
    from inference import pipeline

    monkeypatch.setenv(pipeline.OFFLINE_ENV, "1")
    out = pipeline.run_xgboost_forecast(T0)
    assert len(out) == HORIZON and np.isfinite(out).all()


def test_broken_xgboost_falls_back_to_the_linear_model(monkeypatch):
    from inference import pipeline

    class Broken:
        config = Config()
        base_model = None
        input_features = ()

        def predict(self, history, features):
            raise RuntimeError("boom")

    hours = pd.date_range(T0 - pd.Timedelta(days=100), T0, freq="h", inclusive="left")
    history = pd.Series(500.0, index=hours)
    calls = []
    monkeypatch.setattr(pipeline, "run_linear_forecast",
                        lambda t0, **kw: calls.append(kw) or [1.0] * HORIZON)
    out = pipeline.run_xgboost_forecast(T0, model=Broken(), history=history, features={})
    assert out == [1.0] * HORIZON and calls[0]["history"] is history


def test_submission_default_is_the_hybrid(monkeypatch):
    from inference import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "run_forecast", lambda t0: calls.append("baseline") or [])
    monkeypatch.setattr(pipeline, "run_linear_forecast", lambda t0: calls.append("linear") or [])
    monkeypatch.setattr(pipeline, "run_xgboost_forecast", lambda t0: calls.append("xgb") or [])
    for choice in (None, "xgboost", "linear", "baseline"):
        if choice is None:
            monkeypatch.delenv(pipeline.MODEL_ENV, raising=False)
        else:
            monkeypatch.setenv(pipeline.MODEL_ENV, choice)
        pipeline.run_selected_forecast(T0)
    assert calls == ["xgb", "xgb", "linear", "baseline"]


def test_unexpected_failure_in_the_hybrid_path_ends_in_the_baseline(monkeypatch):
    from inference import pipeline

    def boom(t0):
        raise RuntimeError("boom")

    monkeypatch.delenv(pipeline.MODEL_ENV, raising=False)
    monkeypatch.setattr(pipeline, "run_xgboost_forecast", boom)
    monkeypatch.setattr(pipeline, "run_forecast", lambda t0: [2.0] * HORIZON)
    assert pipeline.run_selected_forecast(T0) == [2.0] * HORIZON


def test_submitted_forecast_offline_runs_the_hybrid(monkeypatch, capsys):
    from inference import pipeline

    monkeypatch.setenv(pipeline.OFFLINE_ENV, "1")
    monkeypatch.delenv(pipeline.MODEL_ENV, raising=False)
    out = pipeline.run_selected_forecast(T0)
    assert len(out) == HORIZON and np.isfinite(out).all()


def test_regressor_never_missing_in_training_is_imputed_with_its_median():
    speed, positions, x = synthetic()  # "b" has gaps in training, "a" has none
    model = XGBoostModel(Config(features=("a", "b"), n_trees=30)).fit(speed, positions, x)
    assert set(model.impute_values) == {"a"}
    assert model.impute_values["a"] == pytest.approx(np.median(x[:, 0]), rel=1e-6)
    model.booster = None
    i = len(positions) - 100
    p = positions[i]
    nan_a = model.forecast(speed, [p], np.array([[np.nan, x[i, 1]]]))
    median_a = model.forecast(speed, [p], np.array([[model.impute_values["a"], x[i, 1]]]))
    assert np.allclose(nan_a, median_a)
    again = XGBoostModel.from_dict(model.to_dict())
    assert again.impute_values == model.impute_values


def test_shipped_asset_imputes_the_feeds_never_missing_in_training():
    model = XGBoostModel.load()
    # flare and F10.7 feeds are never missing in training, so their default branch is untrained
    activity = {f for f in model.config.features
                if f.startswith(("m_flares_", "xray_peak_", "f107_"))}
    assert activity and activity <= set(model.impute_values)
