"""Bayes model: the submitted linear model plus two regressors from the latent Bayes stream.

An option, not the submitted model (SWP_MODEL=bayes; user, 2026-11-16). The structural
model v3 (models.bayes_stream: slow wind + hierarchical stream arrivals) is fitted with the
arrivals latent (summed out) on all the training data by NUTS (training.bayes_model, dev
only: jax, numpyro). From N_DRAWS of its posterior draws, per run, numpy only:

    parr   P(N >= 1) x P(first arrival <= k): the stream timing, by lead k
    mean   a + rho^k (v0 - a) + sum_u P(N >= 1) pi_u E(x(u)) profile(k; u): the additive
           model's mean path (E and D at their population medians given x(u), R = exp(mu_R))

Each is a lead-aligned regressor "structural_bayes_<name>_lead_0h" of the linear model
(models.linear_model: lead k reads the column for its 6-hour block), refitted with them on
the same training runs. On the six yearly folds this set was 0.41% better than the
submitted model (4 of 6 folds; docs/bayes_features.md), short of the 1% rule. See
docs/bayes_model.md.
"""

import json
from pathlib import Path

import numpy as np

from models import bayes_stream as bs
from models.linear_model import LinearModel, lag_column, lead_lags

ASSET = Path(__file__).resolve().parents[1] / "assets" / "bayes_model.json"
FORMAT_VERSION = 1
SIGNALS = ("parr", "mean")
ALL_SIGNALS = ("ambient", "slow", "parr", "stream", "mean")
N_DRAWS = 32
CHUNK = 40  # runs per block in the stream expectation (memory: runs x draws x 84 x 72)


def signal_name(s):
    return f"structural_bayes_{s}"


def features(draws, inp, cfg, n_draws=N_DRAWS, seed=0):
    """{signal: (n, 72)} posterior-mean features of runs with inputs `inp`
    (BayesStreamModel.prepare): ambient, slow, parr, stream and mean (ALL_SIGNALS)."""
    rng = np.random.default_rng(seed)
    n_all = len(draws["rho_up"])
    idx = rng.choice(n_all, size=min(n_draws, n_all), replace=False)
    d = {k: np.asarray(draws[k])[idx] for k in bs.GLOBALS}
    x, amb, v0 = inp["x"], inp["amb"], inp["last"]
    x84 = bs.extend(x)
    n = len(v0)
    a = bs.ambient(d["w_a"].T, amb, np)  # (n, P)
    slow = bs.slow_mean(a, v0[:, None], d["rho_up"], d["rho_dn"], np)  # (n, P, 72)
    clp = bs.count_logp(x.mean(axis=1) @ d["w_N"].T, cfg.nmax, np)  # (n, P, nmax + 1)
    p_any = 1.0 - np.exp(clp[..., 0])
    pi = np.exp(bs.stick_logp(np.einsum("nup,Pp->nPu", x84, d["w_tau"]), np))  # (n, P, 84)
    parr = p_any[..., None] * np.cumsum(pi, axis=-1)[..., :bs.HORIZON]
    xc = x84[:, :, 1:]
    E = np.exp(d["mu_E"][:, None] + np.einsum("nuq,Pq->nPu", xc, d["w_E"]))  # (n, P, 84)
    D = np.exp(d["mu_D"][:, None] + np.einsum("nuq,Pq->nPu", xc, d["w_D"]))
    R = np.broadcast_to(np.exp(d["mu_R"])[None, :, None], E.shape)
    tau = np.broadcast_to(bs.ARRIVALS, E.shape)
    stream = np.empty((n, bs.HORIZON))
    for i in range(0, n, CHUNK):  # sum_u pi_u E_u profile_u(k), mean over draws
        sl = slice(i, min(i + CHUNK, n))
        prof = bs.profile(tau[sl], E[sl], R[sl], D[sl], np)  # (c, P, 84, 72)
        stream[sl] = (p_any[sl, :, None] * np.einsum("cPu,cPuk->cPk", pi[sl], prof)).mean(1)
    out = {"ambient": np.repeat(a.mean(axis=1)[:, None], bs.HORIZON, axis=1),
           "slow": slow.mean(axis=1), "parr": parr.mean(axis=1), "stream": stream}
    out["mean"] = out["slow"] + out["stream"]
    return out


def features_in_blocks(draws, inp, cfg, block=1000, n_draws=N_DRAWS):
    """features() over blocks of runs (memory), with the same draws for every block."""
    n = len(inp["last"])
    parts = [features(draws, {k: v[i:i + block] for k, v in inp.items()
                              if k in ("x", "amb", "last")}, cfg, n_draws)
             for i in range(0, n, block)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def lead_columns(signal, f):
    """{"<signal>_at_<L>h": f at lead max(-L, 1)} for the lead-aligned regressor (as
    training.structural.structural_columns)."""
    lags = np.unique(lead_lags(0, signal))
    return {lag_column(signal, lag): f[:, max(-int(lag), 1) - 1] for lag in lags}


def signal_columns(feats, signals=SIGNALS):
    """{column: (n,)} of the lead-aligned regressors of `signals`."""
    cols = {}
    for s in signals:
        cols.update(lead_columns(signal_name(s), feats[s]))
    return cols


class BayesModel:
    """The linear model with the Bayes stream regressors; `stream` holds the posterior draws
    and the input statistics, `linear` the refitted linear model."""

    def __init__(self, stream, linear, signals=SIGNALS):
        self.stream = stream
        self.linear = linear
        self.signals = tuple(signals)
        self.fit_info = {}

    @property
    def regressor_names(self):
        """The base regressors (the submitted linear model's input features)."""
        return tuple(self.stream.pre.config.input_features)

    @property
    def history_hours(self):
        return max(self.linear.config.history_hours, self.stream.config.ambient_hours + 1)

    def columns(self, values, positions, regressors):
        """{column: (n,)} Bayes regressors of runs at `positions` (base `regressors`)."""
        inp = self.stream.prepare(values, positions, regressors)
        draws = self.stream.draws
        feats = features_in_blocks(draws, inp, self.stream.bayes,
                                   n_draws=len(draws["rho_up"]))
        return signal_columns(feats, self.signals)

    def design(self, values, positions, regressors):
        """The linear model's regressor matrix: the base columns and the Bayes columns."""
        regressors = np.asarray(regressors, dtype=float).reshape(len(positions), -1)
        base = {n: regressors[:, j] for j, n in enumerate(self.regressor_names)}
        extra = self.columns(values, positions, regressors)
        return np.stack([extra[n] if n in extra else base[n]
                         for n in self.linear.config.input_features], axis=1)

    def forecast(self, values, positions, regressors):
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        return self.linear.forecast(values, positions,
                                    self.design(values, positions, regressors))

    def predict(self, history, features=None):
        """72 forecasts from the hourly speed history ending at t0 - 1h and one row of the
        base regressors ({name: value}; missing or NaN -> the training medians)."""
        values = np.asarray(history, dtype=float)
        features = features or {}
        row = np.array([[float(features.get(n, np.nan)) for n in self.regressor_names]])
        return self.forecast(values, np.array([len(values)]), row)[0]

    # ------------------------------------------------------------------ assets
    def to_dict(self):
        return {"model": "BayesModel", "format": FORMAT_VERSION, "signals": list(self.signals),
                "stream": self.stream.to_dict(), "linear": self.linear.to_dict(),
                "fit_info": self.fit_info}

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported bayes model format {d.get('format')}")
        m = cls(bs.BayesStreamModel.from_dict(d["stream"]), LinearModel.from_dict(d["linear"]),
                d["signals"])
        m.fit_info = d.get("fit_info", {})
        return m

    def save(self, path=ASSET):
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path=ASSET):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
