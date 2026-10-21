"""Causal features for the recurrence baseline.

Every feature at position p (a forecast made at t0 = hour p) uses only values[:p],
the hours fully observed by t0. Lead k targets values[p + k].
"""

import numpy as np

HORIZON = 72
LEADS = np.arange(1, HORIZON + 1)


def last_observed(values, positions):
    """Last finite value strictly before each position (NaN if there is none)."""
    n = len(values)
    seen = np.where(np.isfinite(values), np.arange(n), -1)
    seen = np.maximum.accumulate(seen) if n else seen
    prev = positions - 1
    src = np.full(len(positions), -1)
    ok = (prev >= 0) & (prev < n)
    src[ok] = seen[prev[ok]]
    out = np.full(len(positions), np.nan)
    out[src >= 0] = values[src[src >= 0]]
    return out


def window_means(values, centers, half_width):
    """Mean of finite values in [c - half_width, c + half_width] for each center c."""
    n = len(values)
    finite = np.isfinite(values)
    csum = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    ccount = np.concatenate([[0], np.cumsum(finite)])
    lo = np.clip(centers - half_width, 0, n)
    hi = np.clip(centers + half_width + 1, 0, n)
    count = ccount[hi] - ccount[lo]
    total = csum[hi] - csum[lo]
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def check_lags(lag_hours, half_width):
    """A recurrence window must end before t0 for every lead, or it would see the future."""
    for lag in lag_hours:
        if lag - HORIZON - half_width < 1:
            raise ValueError(
                f"lag {lag} h is too short: its window reaches t0 at lead {HORIZON}"
            )


def design(values, positions, mean, lag_hours, half_width):
    """Design tensor (len(positions), 72, 2 + len(lag_hours)) for the per-lead regression.

    Columns: 1, last - mean, then (r_j - mean) for each lag, where last is the last
    observed speed before t0 and r_j is the mean speed within +-half_width hours of
    t0 + k - lag_j. A missing input becomes the mean (a zero column entry). Centring
    on the training mean keeps the intercept and slopes well separated.
    """
    check_lags(lag_hours, half_width)
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions)
    n = len(positions)
    x = np.zeros((n, HORIZON, 2 + len(lag_hours)))
    x[:, :, 0] = 1.0
    last = last_observed(values, positions)
    x[:, :, 1] = np.where(np.isfinite(last), last - mean, 0.0)[:, None]
    for j, lag in enumerate(lag_hours):
        centers = positions[:, None] + LEADS - lag
        r = window_means(values, centers, half_width)
        x[:, :, 2 + j] = np.where(np.isfinite(r), r - mean, 0.0)
    return x


def targets(values, positions, end=None):
    """values[p + k] for each position and lead; NaN if missing or at/after `end`.

    `end` (exclusive) defaults to len(values). A rolling refit at t0 = position P
    passes end=P, so it only learns from targets observed before t0.
    """
    values = np.asarray(values, dtype=float)
    end = len(values) if end is None else end
    idx = np.asarray(positions)[:, None] + LEADS
    out = np.full(idx.shape, np.nan)
    inside = (idx >= 0) & (idx < min(end, len(values)))
    out[inside] = values[idx[inside]]
    return out
