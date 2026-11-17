"""models.bayes_model: the linear model with the latent Bayes stream's regressors, and its
live path (SWP_MODEL=bayes)."""

import numpy as np
import pandas as pd
import pytest
from test_bayes_features import draws

from models import bayes_model as bm
from models import bayes_stream as bs
from models import structural as st
from models.linear_model import Config as LinearConfig
from models.linear_model import LinearModel

T0 = pd.Timestamp("2026-03-15T12:00Z")
HORIZON = bs.HORIZON


def synthetic(n_hours=24 * 150, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    driver = rng.normal(0, 1, n_hours + 48)
    speed = 420 + 80 * np.sin(2 * np.pi * t / 648) + 40 * driver[24: n_hours + 24]
    speed += rng.normal(0, 15, n_hours)
    positions = np.arange(800, n_hours - 80, 6)
    x = driver[positions + 24][:, None]
    return speed, positions, x


def small_model(speed, positions, x):
    stream = bs.BayesStreamModel(st.Config(features=("a",)), bs.Config(rule="add", nmax=1,
                                                                       latent=True))
    stream.set_statistics(speed, positions, x)
    p = stream.config.n_inputs
    stream.draws = {k: v for k, v in draws(5).items()}
    stream.draws["w_N"] = stream.draws["w_N"][:, :p]
    stream.draws["w_tau"] = stream.draws["w_tau"][:, :p]
    stream.draws["w_E"] = stream.draws["w_E"][:, :p - 1]
    stream.draws["w_D"] = stream.draws["w_D"][:, :p - 1]
    cfg = LinearConfig(features=("a", *(f"{bm.signal_name(s)}_lead_0h" for s in bm.SIGNALS)),
                       relative=True, lags_days=(27.0, 13.5), despike=True)
    model = bm.BayesModel(stream, LinearModel(cfg))
    model.linear = LinearModel(cfg).fit(speed, positions, model.design(speed, positions, x))
    return model


def test_design_holds_the_base_regressors_and_the_lead_aligned_signals():
    speed, positions, x = synthetic()
    model = small_model(speed, positions, x)
    names = list(model.linear.config.input_features)
    design = model.design(speed, positions[:7], x[:7])
    assert design.shape == (7, len(names))
    np.testing.assert_array_equal(design[:, names.index("a")], x[:7, 0])
    inp = model.stream.prepare(speed, positions[:7], x[:7])
    f = bm.features(model.stream.draws, inp, model.stream.bayes, n_draws=5)
    for s in bm.SIGNALS:
        for col, v in bm.lead_columns(bm.signal_name(s), f[s]).items():
            np.testing.assert_allclose(design[:, names.index(col)], v)
    assert any(n.startswith("structural_bayes_parr_at_") for n in names)


def test_predict_matches_forecast_and_survives_the_json_round_trip(tmp_path):
    speed, positions, x = synthetic()
    model = small_model(speed, positions, x)
    pos = positions[-1]
    want = model.forecast(speed, np.array([pos]), x[-1:])[0]
    got = model.predict(speed[:pos], {"a": float(x[-1, 0])})
    np.testing.assert_allclose(got, want)
    model.save(tmp_path / "m.json")
    again = bm.BayesModel.load(tmp_path / "m.json")
    np.testing.assert_allclose(again.predict(speed[:pos], {"a": float(x[-1, 0])}), want)
    assert np.isfinite(model.predict(speed[:pos], {})).all()  # regressor imputed


def test_forecast_uses_no_future_hours():
    speed, positions, x = synthetic()
    model = small_model(speed, positions, x)
    pos = positions[-3]
    scrambled = speed.copy()
    scrambled[pos:] = np.random.default_rng(5).normal(700, 200, len(speed) - pos)
    np.testing.assert_allclose(model.forecast(scrambled, np.array([pos]), x[-3:-2]),
                               model.forecast(speed, np.array([pos]), x[-3:-2]))


def test_broken_bayes_model_falls_back_to_the_linear_model(monkeypatch):
    from inference import pipeline

    class Broken:
        regressor_names = ("a",)
        history_hours = 100

        def predict(self, history, features):
            raise RuntimeError("boom")

    hours = pd.date_range(T0 - pd.Timedelta(days=30), T0, freq="h", inclusive="left")
    history = pd.Series(500.0, index=hours)
    calls = []
    monkeypatch.setattr(pipeline, "run_linear_forecast",
                        lambda t0, **kw: calls.append(kw) or [1.0] * HORIZON)
    out = pipeline.run_bayes_forecast(T0, model=Broken(), history=history, features={})
    assert out == [1.0] * HORIZON and calls[0]["history"] is history


def test_swp_model_bayes_selects_the_bayes_model(monkeypatch):
    from inference import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "run_linear_forecast", lambda t0: calls.append("linear") or [])
    monkeypatch.setattr(pipeline, "run_bayes_forecast", lambda t0: calls.append("bayes") or [])
    for choice in ("bayes", "BAYES", "linear"):
        monkeypatch.setenv(pipeline.MODEL_ENV, choice)
        pipeline.run_selected_forecast(T0)
    assert calls == ["bayes", "bayes", "linear"]


shipped = pytest.mark.skipif(not bm.ASSET.exists(), reason="no bayes model asset")


@shipped
def test_shipped_asset_is_consistent():
    # built on the submitted config of 2026-11-16; it fetches its own regressors, so it
    # need not follow later changes of the submitted linear model
    model = bm.BayesModel.load()
    assert set(model.regressor_names) <= set(model.linear.config.input_features)
    assert model.signals == bm.SIGNALS and len(model.stream.draws["rho_up"]) == bm.N_DRAWS
    assert model.stream.bayes.latent


@shipped
def test_bayes_live_path_offline(monkeypatch, capsys):
    from inference import pipeline

    monkeypatch.setenv(pipeline.OFFLINE_ENV, "1")
    monkeypatch.setenv(pipeline.MODEL_ENV, "bayes")
    out = pipeline.run_selected_forecast(T0)
    assert len(out) == HORIZON and np.isfinite(out).all()
    assert "bayes model done" in capsys.readouterr().err
