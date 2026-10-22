"""training.datasets split and the training.evaluate harness, on a synthetic series."""

import numpy as np
import pandas as pd
import pytest

from training import datasets
from training.baselines import Persistence
from training.evaluate import HORIZON, evaluate, forecast_times, run_forecasts, run_timse, score
from training.metrics import timse


@pytest.fixture(scope="module")
def hourly_csv(tmp_path_factory):
    """A full-window hourly file: a smooth signal with some missing hours."""
    times = pd.date_range(datasets.DATA_START, datasets.EVAL_END, freq="h", inclusive="left")
    rng = np.random.default_rng(0)
    value = 450 + 100 * np.sin(np.arange(len(times)) / 50) + rng.normal(0, 10, len(times))
    value[rng.choice(len(times), 500, replace=False)] = np.nan
    path = tmp_path_factory.mktemp("ace") / "speed_hourly.csv"
    pd.DataFrame({"Time": times, "status": "ok", "value": value, "n_valid": 60}).to_csv(
        path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ"
    )
    return path


def test_split_is_contiguous_and_disjoint(hourly_csv):
    train = datasets.load_train(hourly_csv)
    held_out = datasets.load_eval(hourly_csv)
    assert train.index[0] == datasets.DATA_START
    assert train.index[-1] == datasets.TRAIN_END - pd.Timedelta(hours=1)
    assert held_out.index[0] == datasets.TRAIN_END
    assert held_out.index[-1] == datasets.EVAL_END - pd.Timedelta(hours=1)
    assert len(train) == 78888  # 9 years incl. 29 Feb 2020 and 2024
    assert len(held_out) == 8760


def test_missing_hours_stay_nan(hourly_csv):
    assert datasets.load_hourly(hourly_csv).isna().sum() == 500


def test_load_fails_clearly_when_data_is_absent_or_short(tmp_path, hourly_csv):
    with pytest.raises(FileNotFoundError, match="pull_ace_rtsw"):
        datasets.load_train(tmp_path / "nope.csv")
    short = tmp_path / "short.csv"
    short.write_text("\n".join(hourly_csv.read_text().splitlines()[:100]) + "\n")
    with pytest.raises(ValueError, match="does not cover"):
        datasets.load_eval(short)


def test_forecast_times_cover_the_eval_year_on_the_6h_cycle():
    t0s = forecast_times()
    assert t0s[0] == datasets.TRAIN_END
    assert (t0s.hour % 6 == 0).all()
    assert t0s[-1] + pd.Timedelta(hours=HORIZON) < datasets.EVAL_END
    assert t0s[-1] + pd.Timedelta(hours=HORIZON + 6) >= datasets.EVAL_END


class Spy(Persistence):
    """Persistence that records what the harness shows it."""

    def fit(self, train):
        self.train_last = train.index.max()
        return super().fit(train)

    def predict(self, history, t0):
        assert history.index.max() < t0, "predict saw data at or after t0"
        assert history.index.max() == t0 - pd.Timedelta(hours=1)
        return super().predict(history, t0)


def test_fit_sees_only_training_years_and_predict_never_sees_the_future(hourly_csv):
    spy = Spy()
    evaluate(spy, hourly_csv)
    assert spy.train_last < datasets.TRAIN_END


class Oracle:
    """Cheats by reading the answers, to check the target alignment is lead k -> t0+k."""

    def __init__(self, series):
        self.series = series

    def fit(self, train):
        return self

    def predict(self, history, t0):
        leads = pd.date_range(t0 + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
        return self.series.reindex(leads).fillna(0).to_numpy()


def test_perfect_forecast_scores_zero(hourly_csv):
    result = evaluate(Oracle(datasets.load_hourly(hourly_csv)), hourly_csv)
    assert result["timse"] == 0.0
    assert result["n_forecasts"] == len(forecast_times())
    assert result["n_scored"] + result["n_missing_actual"] == result["n_forecasts"] * HORIZON
    assert result["n_missing_actual"] > 0


def test_persistence_gets_a_positive_score(hourly_csv):
    assert evaluate(Persistence(), hourly_csv)["timse"] > 0


@pytest.mark.parametrize("bad", [np.zeros(71), np.full(HORIZON, np.nan)])
def test_rejects_malformed_forecasts(hourly_csv, bad):
    class Bad(Persistence):
        def predict(self, history, t0):
            return bad

    with pytest.raises(ValueError, match="72 finite values"):
        evaluate(Bad(), hourly_csv)


def test_run_timse_scores_each_run_on_its_own_hours():
    actual = np.array([[400.0, 500.0, 600.0], [400.0, np.nan, np.nan], [np.nan] * 3])
    forecast = np.array([[410.0, 480.0, 600.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scores = run_timse(forecast, actual)
    assert scores[0] == pytest.approx(timse(actual[0], forecast[0]))
    assert np.isnan(scores[1]) and np.isnan(scores[2])  # fewer than 2 observations


def test_score_summarises_the_per_run_distribution():
    rng = np.random.default_rng(0)
    actual = rng.normal(450, 100, (50, HORIZON))
    forecast = actual + rng.normal(0, 30, (50, HORIZON))
    actual[0] = np.nan  # an unscorable run
    result = score(forecast, actual)
    runs = run_timse(forecast, actual)[1:]
    assert result["n_forecasts"] == 50
    assert result["n_runs_scored"] == 49
    assert result["run_timse_mean"] == pytest.approx(runs.mean())
    assert result["run_timse_median"] == pytest.approx(np.median(runs))
    assert result["run_timse_p10"] <= result["run_timse_median"] <= result["run_timse_p90"]
    assert result["run_timse_max"] == pytest.approx(runs.max())
    assert result["timse"] == pytest.approx(timse(actual, forecast))


def test_run_forecasts_returns_one_row_per_t0(hourly_csv):
    t0s, forecasts, actuals = run_forecasts(Persistence(), hourly_csv)
    assert forecasts.shape == actuals.shape == (len(forecast_times()), HORIZON)
    assert (t0s == forecast_times()).all()
