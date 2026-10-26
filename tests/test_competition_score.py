"""The official score: the invariants of the organisers' spec, section 11."""

import numpy as np
import pandas as pd
import pytest

from training import competition_score as cs

RNG = np.random.default_rng(0)


def brute_timse(obs, fc):
    """Spec section 6 written out literally."""
    q = np.quantile(obs, np.arange(101) / 100, method=cs.QUANTILE_METHOD)
    total = 0.0
    for k in range(100):
        sel = obs >= q[k]
        total += (q[k + 1] - q[k]) * np.mean((obs[sel] - fc[sel]) ** 2)
    return total / (obs.max() - obs.min())


def sample(n=200, leads=cs.HORIZON):
    y = 300 + 400 * RNG.random((n, leads))
    f = y + RNG.normal(0, 60, (n, leads))
    return f, y, y[:, 0] + RNG.normal(0, 30, n)


def test_percentile_timse_matches_the_formula():
    obs = 300 + 500 * RNG.random(500)
    fc = obs + RNG.normal(0, 50, 500)
    assert cs.percentile_timse(obs, fc) == pytest.approx(brute_timse(obs, fc), rel=1e-12)


def test_exactly_100_intervals_with_velocity_width_weights():
    # 101 distinct equally spaced observations: every interval has width 1, and the
    # weights are widths, not 1/100 each
    obs = np.arange(101, dtype=float) * 10.0  # widths 10, span 1000
    fc = obs.copy()
    fc[-1] += 10.0  # one error, only in the top pair
    # M_k = 100 / (101 - k) for the set v >= q_k = 10 k; TIMSE = sum 10 M_k / 1000
    expected = sum(10.0 * 100.0 / (101 - k) for k in range(100)) / 1000.0
    assert cs.percentile_timse(obs, fc) == pytest.approx(expected)
    # with uneven spacing, velocity widths differ from equal 1/100 weights
    obs = 300.0 + np.arange(101, dtype=float) ** 2 / 10.0  # widths grow with speed
    fc = obs.copy()
    fc[-1] += 10.0
    widths = np.diff(obs) / (obs[-1] - obs[0])
    mse = np.array([100.0 / (101 - k) for k in range(100)])
    assert cs.percentile_timse(obs, fc) == pytest.approx(np.sum(widths * mse))
    assert cs.percentile_timse(obs, fc) != pytest.approx(mse.mean(), rel=1e-3)
    assert len(cs.quantiles(obs)) == cs.N_INTERVALS + 1 == 101


def test_zero_width_intervals_contribute_nothing():
    obs = np.array([400.0] * 50 + [500.0] * 50)  # most quantiles coincide
    fc = obs + 10.0
    # only the jump from 400 to 500 has width; above it every pair has error 100
    assert cs.percentile_timse(obs, fc) == pytest.approx(brute_timse(obs, fc))
    assert cs.percentile_timse(obs, fc) == pytest.approx(100.0)


def test_quantiles_use_observations_only():
    obs = 300 + 500 * RNG.random(300)
    fc = obs + RNG.normal(0, 40, 300)
    # a forecast change that keeps every squared error leaves TIMSE unchanged
    mirrored = 2 * obs - fc
    assert cs.percentile_timse(obs, fc) == pytest.approx(cs.percentile_timse(obs, mirrored))


def test_quantiles_are_per_lead():
    f, y, p = sample()
    y[:, 1] += 500.0  # lead 2 lives in another speed range
    f[:, 1] += 500.0
    per = cs.lead_scores(f, y, p)
    for j in (0, 1):
        assert per["timse"][j] == pytest.approx(brute_timse(y[:, j], f[:, j]))


def test_missing_truth_contributes_nothing():
    f, y, p = sample()
    base = cs.score(f, y, p)
    y2, f2 = y.copy(), f.copy()
    y2[:20, 5] = np.nan
    f2[:20, 5] = 1e6  # a wild forecast on those pairs changes nothing
    per = cs.lead_scores(f2, y2, p)
    assert per["n"][5] == len(y) - 20
    ref = cs.lead_scores(f[20:, 5:6], y[20:, 5:6], p[20:])
    assert per["s"][5] == pytest.approx(ref["s"][0])
    assert np.isfinite(base["s_final"])


def test_missing_prediction_becomes_persistence_and_stays_in():
    f, y, p = sample()
    f2 = f.copy()
    f2[3, :] = np.nan  # a failed run
    f2[7, 10] = np.inf
    per = cs.lead_scores(f2, y, p)
    assert per["n"].sum() == y.size and per["n_replaced"].sum() == 72 + 1
    f3 = f.copy()
    f3[3, :] = p[3]
    f3[7, 10] = p[7]
    assert np.allclose(per["s"], cs.lead_scores(f3, y, p)["s"])


def test_leads_are_weighted_equally_whatever_their_counts():
    f, y, p = sample()
    y2 = y.copy()
    y2[: len(y) // 2, 0] = np.nan  # lead 1 keeps half its pairs
    out = cs.score(f, y2, p)
    s = np.array(out["curves"]["s"])
    assert out["s_final"] == pytest.approx(s.mean())
    assert out["s_49_72"] == pytest.approx(s[48:72].mean())
    assert out["curves"]["n"][0] == len(y) - len(y) // 2


def test_s_is_the_geometric_mean_of_rmse_and_trmse():
    f, y, p = sample()
    per = cs.lead_scores(f, y, p)
    assert np.allclose(per["s"], np.sqrt(per["rmse"] * np.sqrt(per["timse"])))


def test_not_pooled_and_not_per_run():
    f, y, p = sample()
    out = cs.score(f, y, p)
    pooled_rmse = np.sqrt(np.mean((f - y) ** 2))
    pooled_s = np.sqrt(pooled_rmse * np.sqrt(brute_timse(y.ravel(), f.ravel())))
    assert out["s_final"] != pytest.approx(pooled_s, rel=1e-6)


def test_persistence_and_recurrence_use_nothing_after_t0():
    hours = pd.date_range("2026-01-01", periods=24 * 40, freq="h", tz="UTC")
    truth = pd.Series(np.arange(len(hours), dtype=float), index=hours)
    t0 = pd.Timestamp("2026-02-01T06:00Z")
    truth.loc[t0 - pd.Timedelta(hours=1)] = np.nan  # the latest hour is missing
    p = cs.persistence(truth, [t0])
    assert p[0] == truth.loc[t0 - pd.Timedelta(hours=2)]
    future = truth.copy()
    future[future.index >= t0] = -1.0  # scrambling t0 onwards changes nothing
    assert cs.persistence(future, [t0])[0] == p[0]
    rec = cs.recurrence(truth, [t0])
    assert rec[0, 0] == truth.loc[t0 + pd.Timedelta(hours=1) - cs.RECURRENCE]
    assert np.array_equal(cs.recurrence(future, [t0]), rec)  # t0 + 72 h - 27 d < t0
    obs = cs.verifying(truth, [t0])
    assert obs[0, 0] == truth.loc[t0 + pd.Timedelta(hours=1)] and obs.shape == (1, 72)


MONTHLY = """\
:Product: 202610_ace_swepam_1h.txt
# Missing data values: Density and Speed = -9999.9, Temp. = -1.00e+05
#-------------------------------------------------------------------------
2026 10 01  0000   61314       0    0        2.3      288.4     1.68e+04
2026 10 01  0100   61314    3600    9    -9999.9    -9999.9    -1.00e+05
2026 10 01  0200   61314    7200    1        2.7      291.5     1.69e+04
2026 10 01  0300   61314   10800    0        4.5      292.2     1.25e+04
2026 10 01  0300   61314   10800    0        4.5      293.0     1.25e+04
"""


def test_parse_monthly_masks_bad_status_and_fills():
    s = cs.parse_monthly(MONTHLY)
    assert list(s.index.hour) == [0, 1, 2, 3] and str(s.index.tz) == "UTC"
    assert s.iloc[0] == 288.4 and np.isnan(s.iloc[1]) and np.isnan(s.iloc[2])
    assert s.iloc[3] == 293.0  # a duplicate timestamp keeps the last row


def test_model_loss_weights_reproduce_the_official_timse():
    from models.final_baseline import lead_timse_weights, official_weights

    f, y, _ = sample(n=300, leads=6)
    y[RNG.random(y.shape) < 0.2] = np.nan
    w = lead_timse_weights(y)
    for j in range(y.shape[1]):
        ok = np.isfinite(y[:, j])
        weighted = np.sum(w[ok, j] * (y[ok, j] - f[ok, j]) ** 2)
        assert weighted == pytest.approx(cs.percentile_timse(y[ok, j], f[ok, j]))
    ow = official_weights(y, 0.5)
    assert np.allclose(np.nanmean(np.where(np.isfinite(y), ow, np.nan), axis=0), 1.0)
    assert np.all(ow[~np.isfinite(y)] == 0)
    assert np.allclose(official_weights(y, 0.0)[np.isfinite(y)], 1)
