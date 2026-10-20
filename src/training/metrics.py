"""Threshold-Integrated Mean Squared Error (TIMSE), the competition's ranking metric.

Definition (Benchlab participant briefing, 1 Sept 2026; see docs/timse.md):

    thresholds  v*_1 <= ... <= v*_m   the sorted observed speeds
    S_j   = {i : v_i >= v*_j}
    M_j   = (1 / |S_j|) * sum_{i in S_j} (v_i - vhat_i)^2
    TIMSE = sum_{j=1}^{m-1} (v*_{j+1} - v*_j) * M_j

Lower is better. Units are (km/s)^3, and the value grows with the spread of the
observed speeds, so only compare scores computed over the same observations.
"""

import numpy as np


def _valid_pairs(actual, forecast):
    """Return the (observed, forecast) pairs that take part in scoring.

    Missing observations (NaN in `actual`) are dropped entirely: they are not
    thresholds and have no error. The organisers have said this rule may change;
    keep it in this one place.
    """
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    if actual.shape != forecast.shape:
        raise ValueError(f"actual has {actual.size} values but forecast has {forecast.size}")

    observed = ~np.isnan(actual)
    actual, forecast = actual[observed], forecast[observed]
    if not np.isfinite(actual).all():
        raise ValueError("actual contains infinite values")
    if not np.isfinite(forecast).all():
        raise ValueError("forecast is missing or non-finite where an observation exists")
    return actual, forecast


def timse(actual, forecast):
    """TIMSE of `forecast` against `actual` (both in km/s).

    Inputs are any array-likes of the same shape; they are flattened, so a
    (runs x 72) block is scored as one pooled set. NaN in `actual` marks a
    missing observation. Returns NaN when there are no observations, and 0.0
    with a single observation (the sum has no terms).
    """
    v, vhat = _valid_pairs(actual, forecast)
    n = v.size
    if n == 0:
        return float("nan")

    order = np.argsort(v, kind="stable")
    v = v[order]
    sq_err = (v - vhat[order]) ** 2

    # tail_sum[k] = sum of sq_err[k:]; tail_sum[n] = 0.
    tail_sum = np.concatenate([np.cumsum(sq_err[::-1])[::-1], [0.0]])
    # S_j includes every value tied with v*_j, so it starts at the first such index.
    start = np.searchsorted(v, v, side="left")
    tail_mse = tail_sum[start] / (n - start)

    gaps = np.diff(v)
    return float(np.dot(gaps, tail_mse[:-1]))
