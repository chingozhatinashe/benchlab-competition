"""models.linear_model and feature_engineering.windows: fitting, transforms, pipelines."""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import build, insitu, windows
from feature_engineering.recurrence import HORIZON
from models.linear_model import Config, LinearModel, boxcox, inv_boxcox
from training.linear_model import run_scores

T0 = pd.Timestamp("2026-03-15T12:00Z")
H = pd.Timedelta(hours=1)


def synthetic(n_hours=24 * 120, n_feat=2, seed=0):
    """Speed with a 27-day cycle plus a regressor that leads the speed by a day."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    driver = rng.normal(0, 1, n_hours + 48)
    speed = 420 + 80 * np.sin(2 * np.pi * t / 648) + 40 * driver[24 : n_hours + 24]
    speed += rng.normal(0, 15, n_hours)
    positions = np.arange(700, n_hours - 80, 6)
    x = np.column_stack([driver[positions + 24]] + [rng.normal(0, 1, len(positions))
                                                    for _ in range(n_feat - 1)])
    return speed, positions, x


# ---------------------------------------------------------------------- transforms
@pytest.mark.parametrize("lam", [1.0, 0.5, 0.0, -0.5])
def test_boxcox_round_trips(lam):
    y = np.array([250.0, 400.0, 800.0, 1200.0])
    assert np.allclose(inv_boxcox(boxcox(y, lam), lam), y)


def test_smearing_removes_back_transform_bias():
    rng = np.random.default_rng(1)
    speed, positions, x = synthetic()
    speed = 400 * np.exp(rng.normal(0, 0.35, len(speed)))  # skewed: log-normal around 400
    cfg = Config(features=("a", "b"), boxcox=0.0, weights="uniform", ridge=0.0, lags_days=(27.0,))
    model = LinearModel(cfg).fit(speed, positions, x)
    f = model.forecast(speed, positions, x)
    naive = np.exp(np.log(f).mean())  # roughly what dropping the smear would give
    true_mean = speed.mean()
    assert abs(f.mean() - true_mean) < 0.03 * true_mean
    assert naive < f.mean()


# ---------------------------------------------------------------------- parity with scikit-learn
def _lead_problem(family):
    speed, positions, x = synthetic(n_feat=2)
    cfg = Config(features=("a", "b"), family=family, weights="uniform", ridge=0.0,
                 lags_days=(27.0,), tol=1e-10, max_iter=100)
    model = LinearModel(cfg).fit(speed, positions, x)
    design, _ = model.design(speed, positions, x)
    y = speed[positions + 10]  # lead 10
    return model, design[:, 9, 1:], y


def test_gaussian_matches_sklearn_least_squares():
    from sklearn.linear_model import LinearRegression

    model, X, y = _lead_problem("gaussian")
    ref = LinearRegression().fit(X, boxcox(y, 1.0))  # lambda = 1 is y - 1
    assert np.allclose(model.coef[9, 1:], ref.coef_, rtol=1e-5, atol=1e-6)
    assert model.coef[9, 0] == pytest.approx(ref.intercept_, rel=1e-6)


@pytest.mark.parametrize("family,power", [("gamma", 2), ("inverse_gaussian", 3)])
def test_log_link_glms_match_sklearn(family, power):
    from sklearn.linear_model import TweedieRegressor

    model, X, y = _lead_problem(family)
    ref = TweedieRegressor(power=power, link="log", alpha=0.0, max_iter=1000, tol=1e-10).fit(X, y)
    assert np.allclose(model.coef[9, 1:], ref.coef_, rtol=1e-3, atol=1e-5)
    assert model.coef[9, 0] == pytest.approx(ref.intercept_, rel=1e-4)


def test_weights_act_as_sample_weights():
    from sklearn.linear_model import LinearRegression

    speed, positions, x = synthetic(n_feat=1)
    cfg = Config(features=("a",), ridge=0.0, lags_days=(27.0,))  # run-TIMSE weights
    model = LinearModel(cfg).fit(speed, positions, x)
    design, _ = model.design(speed, positions, x)
    from feature_engineering.recurrence import targets

    y = targets(speed, positions)
    q = model.loss_weights(y)
    ok = q[:, 30] > 0
    ref = LinearRegression().fit(design[ok, 30, 1:], boxcox(y[ok, 30], 1.0),
                                 sample_weight=q[ok, 30])
    assert np.allclose(model.coef[30, 1:], ref.coef_, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------- pipelines
def test_training_drops_incomplete_rows_and_live_imputes():
    speed, positions, x = synthetic()
    x = x.copy()
    x[::5, 1] = np.nan
    model = LinearModel(Config(features=("a", "b"), lags_days=(27.0,))).fit(speed, positions, x)
    assert model.fit_info["dropped_incomplete"] == int(np.isnan(x[:, 1]).sum())
    assert model.medians[1] == pytest.approx(np.nanmedian(x[:, 1]))
    out = model.forecast(speed, positions[:3], np.full((3, 2), np.nan))
    filled = model.forecast(speed, positions[:3], np.tile(model.medians, (3, 1)))
    assert np.allclose(out, filled) and np.isfinite(out).all()


def test_predict_matches_batch_and_survives_missing_everything():
    speed, positions, x = synthetic()
    model = LinearModel(Config(features=("a", "b"), relative=True, lags_days=(27.0,))).fit(
        speed, positions, x)
    p = positions[-1]
    row = {"a": x[-1, 0], "b": x[-1, 1]}
    assert np.allclose(model.predict(speed[:p], row), model.forecast(speed, [p], x[-1:])[0])
    empty = model.predict(np.full(800, np.nan), {})
    assert empty.shape == (HORIZON,) and np.isfinite(empty).all()


@pytest.mark.parametrize("cfg", [
    Config(features=("a", "b"), boxcox=0.5, lags_days=(27.0, 13.5)),
    Config(features=("a", "b"), family="gamma", relative=True, lags_days=(27.0, 13.5)),
])
def test_json_round_trip(tmp_path, cfg):
    speed, positions, x = synthetic()
    model = LinearModel(cfg).fit(speed, positions, x)
    model.save(tmp_path / "m.json")
    again = LinearModel.load(tmp_path / "m.json")
    assert np.allclose(again.forecast(speed, positions[:20], x[:20]),
                       model.forecast(speed, positions[:20], x[:20]), rtol=1e-7)


def test_fit_ignores_targets_after_end():
    speed, positions, x = synthetic()
    cfg = Config(features=("a", "b"), lags_days=(27.0,))
    end = positions[len(positions) // 2] + 1
    a = LinearModel(cfg).fit(speed, positions, x, end=end)
    changed = speed.copy()
    changed[end:] = 5000.0
    b = LinearModel(cfg).fit(changed, positions[positions < end], x[positions < end], end=end)
    keep = positions < end
    c = LinearModel(cfg).fit(speed, positions[keep], x[keep], end=end)
    assert np.allclose(b.coef, c.coef)
    assert a.coef.shape == c.coef.shape


def test_run_scores_equal_timse():
    from training.metrics import timse

    rng = np.random.default_rng(3)
    y = 400 + 100 * rng.random((4, HORIZON))
    f = y + rng.normal(0, 30, y.shape)
    from models.final_baseline import run_timse_weights

    got = run_scores(f, y, run_timse_weights(y))
    assert np.allclose(got, [timse(y[i], f[i]) for i in range(4)])


# ---------------------------------------------------------------------- windows
def test_window_names_round_trip_and_families():
    specs = windows.candidates()
    assert len({s.name for s in specs}) == len(specs) > 100
    for s in specs:
        assert windows.WindowSpec.parse(s.name) == s
    assert windows.WindowSpec.parse("log_t_ratio_mean_6h").family == "log_t_ratio_mean"


def test_window_means_coverage_and_no_future():
    idx = pd.date_range(T0 - pd.Timedelta(days=30), T0 + pd.Timedelta(days=2), freq="h",
                        inclusive="left")
    hourly = pd.DataFrame(np.nan, index=idx, columns=insitu.HOURLY_COLUMNS)
    hourly["bt"] = np.arange(len(idx), dtype=float)
    hourly.loc[T0 - 6 * H : T0 - 5 * H, "bt"] = np.nan  # 2 of the last 6 hours missing
    specs = [windows.WindowSpec("bt", "mean", 6), windows.WindowSpec("bt", "mean", 3),
             windows.WindowSpec("bt", "max", 24)]
    inputs = build.Inputs(hourly=hourly)
    row = windows.window_table(inputs, [T0], specs).iloc[0]
    p = idx.get_loc(T0)
    assert row["bt_mean_6h"] == pytest.approx(np.mean([p - 4, p - 3, p - 2, p - 1]))
    assert row["bt_mean_3h"] == pytest.approx(p - 2)
    later = hourly.copy()
    later.loc[later.index >= T0] = 1e6
    again = windows.window_table(build.Inputs(hourly=later), [T0], specs).iloc[0]
    assert row.equals(again)
    sparse = hourly.copy()
    sparse.loc[T0 - 6 * H : T0 - H, "bt"] = np.nan
    sparse.loc[T0 - 2 * H, "bt"] = 1.0
    assert np.isnan(windows.window_table(build.Inputs(hourly=sparse), [T0], specs[:1]).iloc[0, 0])


# ---------------------------------------------------------------------- live path
def test_shipped_asset_loads_and_names_real_regressors():
    model = LinearModel.load()
    assert model.coef.shape == (HORIZON, model.config.n_columns)
    names = {s.name for s in windows.candidates()}
    assert set(model.config.features) <= names


def test_linear_live_path_offline_imputes_everything(monkeypatch):
    from inference import pipeline

    monkeypatch.setenv(pipeline.OFFLINE_ENV, "1")
    out = pipeline.run_linear_forecast(T0)
    assert len(out) == HORIZON and np.isfinite(out).all()


def test_linear_live_path_uses_given_history_and_features():
    from inference import pipeline

    model = LinearModel.load()
    hours = pd.date_range(T0 - pd.Timedelta(days=60), T0, freq="h", inclusive="left")
    history = pd.Series(450.0, index=hours)
    features = dict(zip(model.config.features, model.medians, strict=True))
    out = pipeline.run_linear_forecast(T0, linear=model, history=history, features=features)
    assert np.allclose(out, model.predict(history, features))


def test_broken_linear_model_falls_back_to_the_baseline(monkeypatch):
    from inference import pipeline

    class Broken:
        config = Config()

        def predict(self, history, features):
            raise RuntimeError("boom")

    hours = pd.date_range(T0 - pd.Timedelta(days=100), T0, freq="h", inclusive="left")
    history = pd.Series(500.0, index=hours)
    baseline = pipeline.load_model()
    out = pipeline.run_linear_forecast(T0, linear=Broken(), baseline=baseline, history=history,
                                       features={})
    assert out == pipeline.run_forecast(T0, baseline, history)


def test_submission_default_stays_the_final_baseline(monkeypatch):
    from inference import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "run_forecast", lambda t0: calls.append("baseline") or [])
    monkeypatch.setattr(pipeline, "run_linear_forecast", lambda t0: calls.append("linear") or [])
    monkeypatch.delenv(pipeline.MODEL_ENV, raising=False)
    pipeline.run_selected_forecast(T0)
    monkeypatch.setenv(pipeline.MODEL_ENV, "linear")
    pipeline.run_selected_forecast(T0)
    assert calls == ["baseline", "linear"]
