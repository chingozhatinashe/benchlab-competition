"""Reference models that use only the hourly speed series (see docs/baselines.md).

Every model follows the training.evaluate interface:

    fit(train)            hourly speed Series, training years only
    predict(history, t0)  72 forecasts for the hours labelled t0+1h .. t0+72h

The fitted models learn one set of coefficients per lead hour, from every hour of
the training years used as a t0. Samples whose target is missing are dropped.
Missing *inputs* fall back to the last observed speed or the training mean; the
stored data itself is never filled.
"""

import numpy as np
from sklearn.linear_model import Ridge

from training.evaluate import HORIZON

LEADS = np.arange(1, HORIZON + 1)
ROTATION_HOURS = 27 * 24  # one synodic solar rotation, rounded to whole days
RECURRENCE_HALF_WIDTH = 3  # hours either side of the lagged time
N_LAGS = 48


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


def lag_matrix(values, positions, n_lags):
    """values[p - n_lags : p] for each position p, NaN where out of range."""
    idx = positions[:, None] + np.arange(-n_lags, 0)
    out = np.full(idx.shape, np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def fill(x, fallback):
    return np.where(np.isfinite(x), x, fallback)


class Persistence:
    """Repeat the last observed hourly speed for all 72 hours."""

    def fit(self, train):
        # Fallback for a history with no valid hours at all.
        self.fallback = float(np.nanmean(train.to_numpy()))
        return self

    def predict(self, history, t0):
        observed = history.dropna()
        last = observed.iloc[-1] if len(observed) else self.fallback
        return np.full(HORIZON, last)


class _PerLeadModel:
    """Shared fit/predict plumbing: features at position p use only values[:p]."""

    def fit(self, train):
        values = train.to_numpy(dtype=float)
        self.mean = float(np.nanmean(values))
        positions = np.arange(1, len(values) - HORIZON)
        targets = values[positions[:, None] + LEADS]
        self._fit(self.features(values, positions), targets)
        return self

    def predict(self, history, t0):
        values = history.to_numpy(dtype=float)
        return self._predict(self.features(values, np.array([len(values)])))[0]


class ClimatologyBlend(_PerLeadModel):
    """v(t0+k) = mean + w_k * (last - mean), with w_k in [0, 1]."""

    def features(self, values, positions):
        return fill(last_observed(values, positions), self.mean)

    def _fit(self, last, targets):
        x = last - self.mean
        self.weights = np.empty(HORIZON)
        for k in range(HORIZON):
            ok = np.isfinite(targets[:, k])
            w = np.dot(x[ok], targets[ok, k] - self.mean) / np.dot(x[ok], x[ok])
            self.weights[k] = np.clip(w, 0.0, 1.0)

    def _predict(self, last):
        return self.mean + np.outer(last - self.mean, self.weights)


class Recurrence27(_PerLeadModel):
    """v(t0+k) = a_k + b_k * last + c_k * (speed one solar rotation before t0+k)."""

    def features(self, values, positions):
        last = fill(last_observed(values, positions), self.mean)
        centers = positions[:, None] + LEADS - ROTATION_HOURS
        recur = fill(window_means(values, centers, RECURRENCE_HALF_WIDTH), self.mean)
        return last, recur

    def _fit(self, feats, targets):
        last, recur = feats
        self.coef = np.empty((HORIZON, 3))
        for k in range(HORIZON):
            ok = np.isfinite(targets[:, k])
            design = np.column_stack([np.ones(ok.sum()), last[ok], recur[ok, k]])
            self.coef[k] = np.linalg.lstsq(design, targets[ok, k], rcond=None)[0]

    def _predict(self, feats):
        last, recur = feats
        a, b, c = self.coef.T
        return a + np.outer(last, b) + c * recur


class Autoregressive(_PerLeadModel):
    """Ridge regression per lead on the last 48 hourly speeds (fixed alpha, not tuned)."""

    alpha = 1.0

    def features(self, values, positions):
        last = fill(last_observed(values, positions), self.mean)
        lags = lag_matrix(values, positions, N_LAGS)
        return np.where(np.isfinite(lags), lags, last[:, None])

    def _fit(self, lags, targets):
        self.models = []
        for k in range(HORIZON):
            ok = np.isfinite(targets[:, k])
            self.models.append(Ridge(alpha=self.alpha).fit(lags[ok], targets[ok, k]))

    def _predict(self, lags):
        return np.column_stack([m.predict(lags) for m in self.models])


ALL = {
    "persistence": Persistence,
    "climatology_blend": ClimatologyBlend,
    "recurrence_27d": Recurrence27,
    "autoregressive_48h": Autoregressive,
}
