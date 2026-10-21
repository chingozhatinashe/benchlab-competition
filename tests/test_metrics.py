"""training.metrics.timse against a literal reading of the briefing formula."""

import numpy as np
import pandas as pd
import pytest

from training.metrics import tail_mse_curve, timse


def reference_timse(actual, forecast):
    """Direct O(n^2) transcription of the slide formula, for checking."""
    pairs = [(v, f) for v, f in zip(actual, forecast, strict=True) if not np.isnan(v)]
    thresholds = sorted(v for v, _ in pairs)
    total = 0.0
    for j in range(len(thresholds) - 1):
        tail = [(v - f) ** 2 for v, f in pairs if v >= thresholds[j]]
        total += (thresholds[j + 1] - thresholds[j]) * sum(tail) / len(tail)
    return total


def test_hand_worked_example():
    # thresholds 300, 400, 600; squared errors 100, 400, 900
    # M_1 = (100+400+900)/3, M_2 = (400+900)/2
    actual = [400.0, 300.0, 600.0]
    forecast = [420.0, 310.0, 570.0]
    expected = 100 * (1400 / 3) + 200 * (1300 / 2)
    assert timse(actual, forecast) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference_on_random_data(seed):
    rng = np.random.default_rng(seed)
    actual = rng.normal(450, 100, 200).round(-1)  # rounding forces ties
    forecast = actual + rng.normal(0, 50, 200)
    actual[rng.choice(200, 15, replace=False)] = np.nan
    assert timse(actual, forecast) == pytest.approx(reference_timse(actual, forecast))


def test_perfect_forecast_scores_zero():
    actual = [350.0, 420.0, 700.0]
    assert timse(actual, actual) == 0.0


def test_missing_observations_are_ignored():
    actual = [400.0, np.nan, 300.0, 600.0]
    forecast = [420.0, 9999.0, 310.0, 570.0]
    assert timse(actual, forecast) == pytest.approx(timse([400, 300, 600], [420, 310, 570]))


def test_missing_forecast_for_missing_observation_is_fine():
    assert timse([400.0, np.nan, 500.0], [410.0, np.nan, 480.0]) == pytest.approx(
        100 * (100 + 400) / 2
    )


def test_high_speed_errors_weigh_more():
    actual = np.array([300.0, 350.0, 400.0, 450.0, 800.0])
    low_miss = actual.copy()
    low_miss[0] += 50
    high_miss = actual.copy()
    high_miss[-1] += 50
    assert timse(actual, high_miss) > timse(actual, low_miss)


def test_order_and_shape_do_not_matter():
    rng = np.random.default_rng(0)
    actual = rng.normal(450, 100, (4, 72))
    forecast = actual + rng.normal(0, 30, (4, 72))
    perm = rng.permutation(actual.size)
    assert timse(actual, forecast) == pytest.approx(
        timse(actual.ravel()[perm], forecast.ravel()[perm])
    )


def test_accepts_pandas_series():
    actual = pd.Series([400.0, 300.0, 600.0])
    forecast = pd.Series([420.0, 310.0, 570.0])
    assert timse(actual, forecast) == pytest.approx(timse(actual.values, forecast.values))


def test_degenerate_inputs():
    assert np.isnan(timse([], []))
    assert np.isnan(timse([np.nan, np.nan], [400.0, 400.0]))
    assert timse([500.0], [400.0]) == 0.0
    assert timse([500.0, 500.0], [400.0, 450.0]) == 0.0  # no spread, no gaps


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="values"):
        timse([400.0, 500.0], [400.0])
    with pytest.raises(ValueError, match="forecast"):
        timse([400.0, 500.0], [400.0, np.nan])
    with pytest.raises(ValueError, match="infinite"):
        timse([400.0, np.inf], [400.0, 500.0])


def test_timse_is_the_area_under_the_tail_mse_curve():
    rng = np.random.default_rng(3)
    actual = rng.normal(450, 100, 500).round(-1)  # ties
    forecast = actual + rng.normal(0, 40, 500)
    thresholds, tail = tail_mse_curve(actual, forecast)
    assert np.all(np.diff(thresholds) >= 0)
    assert np.dot(np.diff(thresholds), tail[:-1]) == pytest.approx(timse(actual, forecast))
    assert tail[0] == pytest.approx(np.mean((actual - forecast) ** 2))  # S_1 is every point
