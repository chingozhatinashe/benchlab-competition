"""A standalone regime-switching forecaster: HMM-gated experts.

    forecast(k) = sum_s g_s * expert_s(k)       (soft; hard: the argmax state's expert)

g        P(state s at t0 | in-situ data up to t0): the forward filter of a hidden Markov
         model (models.regime_hmm) at the in-situ row labelled t0 - 1h. Causal, so it
         is known live (unlike the future-window labels of training.regime_models).
expert_s one per state: a per-lead weighted ridge regression of v(t0+k) - last speed on
         the last speed, the 27-day and 13.5-day recurrence means at t0+k, and
         optionally a few regressors. Expert s is fitted on every training run with
         weights TIMSE weight x g_s(t0), so it learns how runs that start in its
         state evolve.

Independent of the other models: no linear-model or hybrid forecast is used. With
one state (n_states=1) the gate is 1 and the model is one global regression: the
control that measures what regime switching adds.

Training drops isolated one-hour speed glitches from targets and recurrence terms
(feature_engineering.recurrence.despike), as the other models do. Missing speed
terms and regressors take their training mean (zero after centring); a run with
no last speed forecasts the training mean.

Experimental (training.regime_switching, docs/regime_switching.md); not used by
the live forecast.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import HORIZON, LEADS, last_observed, targets, window_means
from models.final_baseline import TAIL_FLOOR, VALID_RANGE, run_timse_weights, solve
from models.linear_model import _despiked
from models.regime_hmm import Config as HMMConfig
from models.regime_hmm import RegimeHMM, observations

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Config:
    n_states: int = 4
    autoregressive: bool = False  # the HMM's speed emission (see models.regime_hmm)
    features: tuple = ()  # optional regressors (feature_engineering.windows names)
    lags_days: tuple = (27.0, 13.5)
    half_width: int = 3
    ridge: float = 0.1  # relative to each slope column's weighted second moment
    despike: bool = True

    @property
    def lag_hours(self):
        return tuple(int(round(d * 24)) for d in self.lags_days)

    @property
    def n_columns(self):
        return 2 + len(self.lags_days) + len(self.features)  # intercept, last, recs, regs


class RegimeSwitchingModel:
    def __init__(self, config=None, hmm=None):
        self.config = config or Config()
        self.hmm = hmm

    # ------------------------------------------------------------------ gate
    def fit_hmm(self, insitu):
        """Fit the gate's HMM on an in-situ hourly frame (training hours only)."""
        cfg = self.config
        self.hmm = RegimeHMM(HMMConfig(n_states=cfg.n_states,
                                       autoregressive=cfg.autoregressive))
        self.hmm.fit(observations(insitu))
        return self

    def gate(self, insitu, t0s):
        """(n, K) filtered state probabilities at t0 - 1h; the stationary mix if unknown."""
        t0s = pd.DatetimeIndex(t0s)
        probs = self.hmm.filter(observations(insitu))
        row = insitu.index.get_indexer(t0s - pd.Timedelta(hours=1))
        out = np.repeat(self.hmm.start[None, :], len(t0s), axis=0)
        ok = row >= 0
        out[ok] = probs[row[ok]]
        return out

    # ------------------------------------------------------------------ design
    def _clean(self, values):
        return _despiked(values) if self.config.despike else np.asarray(values, dtype=float)

    def _raw(self, values, positions, regressors):
        """(n, 72, columns - 1) uncentred inputs and the last speed (n,)."""
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        clean = self._clean(values)
        last = last_observed(values, positions)
        n = len(positions)
        x = np.empty((n, HORIZON, cfg.n_columns - 1))
        x[:, :, 0] = last[:, None]
        for j, lag in enumerate(cfg.lag_hours):
            x[:, :, 1 + j] = window_means(clean, positions[:, None] + LEADS - lag, cfg.half_width)
        if cfg.features:
            reg = np.asarray(regressors, dtype=float).reshape(n, len(cfg.features))
            x[:, :, 1 + len(cfg.lag_hours):] = reg[:, None, :]
        return x, last

    def design(self, values, positions, regressors=None):
        """(n, 72, columns) centred design with an intercept; missing -> 0."""
        raw, last = self._raw(values, positions, regressors)
        x = raw - self.x_mean
        x = np.where(np.isfinite(x), x, 0.0)
        return np.concatenate([np.ones(x.shape[:2] + (1,)), x], axis=2), last

    # ------------------------------------------------------------------ fitting (dev)
    def fit(self, values, positions, gate, regressors=None, end=None):
        """Fit every expert on the runs at `positions`; targets at/after `end` ignored."""
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        hist_end = len(values) if end is None else end
        clean = self._clean(values[:hist_end])
        raw, last = self._raw(values[:hist_end], positions, regressors)
        self.x_mean = np.nanmean(raw, axis=0)  # (72, columns - 1)
        self.x_mean = np.where(np.isfinite(self.x_mean), self.x_mean, 0.0)
        self.mean = float(np.nanmean(clean))
        x, _ = self.design(values[:hist_end], positions, regressors)
        y = targets(clean, positions, end) - last[:, None]
        observed = np.isfinite(y)
        q = run_timse_weights(np.where(observed, y + last[:, None], np.nan))
        scale = q[observed].mean() if observed.any() else 1.0
        q = np.where(observed, np.maximum(q / max(scale, 1e-12), TAIL_FLOOR), 0.0)
        q = np.where(np.isfinite(last)[:, None], q, 0.0)
        y = np.where(q > 0, y, 0.0)
        gate = np.asarray(gate, dtype=float)
        self.coef = np.empty((cfg.n_states, HORIZON, x.shape[2]))
        self.share = gate.mean(axis=0)
        for s in range(cfg.n_states):
            w = q * gate[:, s][:, None]  # (n, 72)
            a = np.einsum("nk,nki,nkj->kij", w, x, x)
            b = np.einsum("nk,nki,nk->ki", w, x, y)
            self.coef[s] = solve(a, b, cfg.ridge)
        return self

    # ------------------------------------------------------------------ forecasting
    def experts(self, values, positions, regressors=None):
        """(K, n, 72) each expert's forecast."""
        x, last = self.design(values, positions, regressors)
        delta = np.einsum("nki,ski->snk", x, self.coef)
        out = last[None, :, None] + delta
        return np.where(np.isfinite(last)[None, :, None], out, self.mean)

    def forecast(self, values, positions, gate, regressors=None, hard=False):
        """(n, 72) gated forecast: soft mixture, or the most probable state's expert."""
        gate = np.asarray(gate, dtype=float)
        if hard:
            gate = np.eye(gate.shape[1])[gate.argmax(axis=1)]
        out = np.einsum("ns,snk->nk", gate, self.experts(values, positions, regressors))
        return np.clip(out, *VALID_RANGE)

    def predict(self, history, insitu, features=None, hard=False):
        """72 forecasts at t0 = the hour after `history` ends; `insitu` rows before t0."""
        values = np.asarray(history, dtype=float)
        t0 = insitu.index[-1] + pd.Timedelta(hours=1)
        g = self.gate(insitu, [t0])
        row = None
        if self.config.features:
            row = np.array([[np.nan if features is None else features.get(f, np.nan)
                             for f in self.config.features]])
        return self.forecast(values, np.array([len(values)]), g, row, hard)[0]

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        cfg = asdict(self.config)
        cfg["features"] = list(cfg["features"])
        cfg["lags_days"] = list(cfg["lags_days"])
        return {"model": "RegimeSwitchingModel", "format": FORMAT_VERSION, "config": cfg,
                "hmm": self.hmm.to_dict(), "x_mean": self.x_mean.tolist(),
                "coef": self.coef.tolist(), "mean": self.mean, "share": self.share.tolist()}

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "RegimeSwitchingModel" or d.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported RegimeSwitchingModel asset")
        cfg = dict(d["config"])
        cfg["features"] = tuple(cfg["features"])
        cfg["lags_days"] = tuple(cfg["lags_days"])
        model = cls(Config(**cfg), RegimeHMM.from_dict(d["hmm"]))
        model.x_mean = np.asarray(d["x_mean"], dtype=float)
        model.coef = np.asarray(d["coef"], dtype=float)
        model.mean = float(d["mean"])
        model.share = np.asarray(d["share"], dtype=float)
        return model

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
