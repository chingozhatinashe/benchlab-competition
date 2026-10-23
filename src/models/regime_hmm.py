"""A hidden Markov model of solar-wind regimes: a Markov-switching autoregression.

Hidden states (4 by default, initialised as slow, intermediate, fast and ejecta)
follow a first-order Markov chain with a fixed transition matrix. Each hour's
observation, given the state s:

    speed        v_t ~ N(c_s + phi_s * v_prev, sigma_s^2)   v_prev: last speed within
                 MAX_GAP hours; with no recent speed, v_t ~ N(m_s, sd_s^2)
    log T ratio  ~ N(mu_s, tau_s^2)   (temperature / expected for the speed)
    log10 Bt     ~ N(nu_s, omega_s^2)

The three are independent given the state, and a missing value drops its term.
Fitted by EM (Baum-Welch), in numpy.

The forward filter is causal: P(state at hour h | observations up to h). For a
forecast at t0 the features use the filter at the in-situ row labelled t0 - 1h
(the rows are already shifted by the in-situ latency, feature_engineering.insitu),
so they only use data available at t0. Features (`features`):

    hmm_p_<state>     filtered probability of each state
    hmm_state_hours   hours since the most probable filtered state last changed
                      (capped at MAX_DURATION)

Experimental (training.regime_hmm, docs/regime_hmm.md); not used by the live forecast.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATE_NAMES = ("slow", "intermediate", "fast", "ejecta")
OBSERVED = ("speed", "log_t_ratio", "log_bt")
MAX_GAP = 6  # hours a previous speed may be carried for the autoregression
MAX_DURATION = 240.0
VAR_FLOOR = {"speed": 25.0, "log_t_ratio": 1e-3, "log_bt": 1e-4}
LOG_2PI = np.log(2 * np.pi)
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Config:
    n_states: int = 4
    max_iter: int = 50
    tol: float = 1e-4  # stop when the log-likelihood per hour improves less
    transition_prior: float = 1.0  # pseudo-counts on each transition (smoothing)
    autoregressive: bool = False  # speed AR(1) per state; False: speed ~ N(m_s, sd_s^2)


def observations(hourly):
    """(T, 3) speed, log T ratio and log10 Bt from an in-situ hourly frame."""
    bt = hourly["bt"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_bt = np.where(bt > 0, np.log10(bt), np.nan)
    return np.column_stack([hourly["speed"].to_numpy(dtype=float),
                            hourly["log_t_ratio"].to_numpy(dtype=float), log_bt])


def previous_speed(speed, max_gap=MAX_GAP):
    """The last observed speed strictly before each hour, within max_gap hours."""
    n = len(speed)
    seen = np.where(np.isfinite(speed), np.arange(n), -1)
    seen = np.maximum.accumulate(seen)
    src = np.r_[-1, seen[:-1]]
    ok = (src >= 0) & (np.arange(n) - src <= max_gap)
    out = np.full(n, np.nan)
    out[ok] = speed[src[ok]]
    return out


def initial_states(obs):
    """Heuristic hard labels to start EM: ejecta (cold, strong field), else by speed."""
    v, ratio, log_bt = obs.T
    with np.errstate(invalid="ignore"):
        ejecta = (ratio < np.log(0.5)) & (log_bt >= np.log10(8.0))
        state = np.where(v < 400, 0, np.where(v < 550, 1, 2))
    state = np.where(ejecta, 3, state)
    return state


class RegimeHMM:
    def __init__(self, config=None):
        self.config = config or Config()
        k = self.config.n_states
        if k != len(STATE_NAMES):
            self.names = tuple(f"s{i}" for i in range(k))
        else:
            self.names = STATE_NAMES

    # ------------------------------------------------------------------ emissions
    def log_emissions(self, obs):
        """(T, K) log density of each hour's observations under each state."""
        v, ratio, log_bt = obs.T
        prev = previous_speed(v)
        out = np.zeros((len(obs), self.config.n_states))

        def add(x, mean, var, rows=None):
            ok = np.isfinite(x) if rows is None else rows & np.isfinite(x)
            d = x[ok, None] - (mean[ok] if np.ndim(mean) == 2 else mean[None, :])
            out[ok] += -0.5 * (LOG_2PI + np.log(var)[None, :] + d**2 / var[None, :])

        ar = np.isfinite(prev) & self.config.autoregressive
        add(v, self.c[None, :] + self.phi[None, :] * prev[:, None], self.sigma2, ar)
        add(v, self.m, self.sd2, ~ar)
        add(ratio, self.mu, self.tau2)
        add(log_bt, self.nu, self.omega2)
        return out

    # ------------------------------------------------------------------ inference
    def _forward(self, log_b):
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        n, k = b.shape
        alpha = np.empty((n, k))
        scale = np.empty(n)
        a = self.start * b[0]
        scale[0] = a.sum()
        alpha[0] = a / scale[0]
        trans = self.trans
        for t in range(1, n):
            a = (alpha[t - 1] @ trans) * b[t]
            s = a.sum()
            if not s > 0:  # every state impossible: restart from the stationary mix
                a, s = self.start * b[t], (self.start * b[t]).sum()
            scale[t] = s
            alpha[t] = a / s
        loglik = float(np.log(scale).sum() + log_b.max(axis=1).sum())
        return alpha, scale, b, loglik

    def filter(self, obs):
        """(T, K) P(state at hour t | observations up to t)."""
        alpha, _, _, _ = self._forward(self.log_emissions(obs))
        return alpha

    def _backward(self, b, scale):
        n, k = b.shape
        beta = np.empty((n, k))
        beta[-1] = 1.0
        trans = self.trans
        for t in range(n - 2, -1, -1):
            beta[t] = trans @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
        return beta

    # ------------------------------------------------------------------ fitting (dev)
    def _m_step(self, obs, gamma, xi=None):
        v, ratio, log_bt = obs.T
        prev = previous_speed(v)
        k = self.config.n_states
        self.c, self.phi, self.sigma2 = np.empty(k), np.empty(k), np.empty(k)
        self.m, self.sd2 = np.empty(k), np.empty(k)
        self.mu, self.tau2 = np.empty(k), np.empty(k)
        self.nu, self.omega2 = np.empty(k), np.empty(k)
        ar = np.isfinite(prev) & np.isfinite(v)
        x = np.column_stack([np.ones(ar.sum()), prev[ar]])
        for s in range(k):
            w = gamma[ar, s]
            xtw = x.T * w
            coef = np.linalg.solve(xtw @ x + 1e-6 * np.eye(2), xtw @ v[ar])
            resid = v[ar] - x @ coef
            self.c[s], self.phi[s] = coef
            self.sigma2[s] = max((w * resid**2).sum() / w.sum(), VAR_FLOOR["speed"])
            for col, mean, var, floor in ((v, self.m, self.sd2, "speed"),
                                          (ratio, self.mu, self.tau2, "log_t_ratio"),
                                          (log_bt, self.nu, self.omega2, "log_bt")):
                ok = np.isfinite(col)
                ws = gamma[ok, s]
                mean[s] = (ws * col[ok]).sum() / ws.sum()
                var[s] = max((ws * (col[ok] - mean[s]) ** 2).sum() / ws.sum(),
                             VAR_FLOOR[floor])
        if xi is not None:
            counts = xi + self.config.transition_prior
            self.trans = counts / counts.sum(axis=1, keepdims=True)
        self.start = stationary(self.trans)

    def fit(self, obs, log=None):
        """EM on (T, 3) observations; hours with nothing observed are skipped."""
        obs = np.asarray(obs, dtype=float)
        k = self.config.n_states
        init = initial_states(obs) if k == len(STATE_NAMES) else _speed_quantiles(obs[:, 0], k)
        gamma = np.eye(k)[init]
        counts = np.full((k, k), self.config.transition_prior)
        np.add.at(counts, (init[:-1], init[1:]), 1.0)
        self.trans = counts / counts.sum(axis=1, keepdims=True)
        self._m_step(obs, gamma)
        self.history = []
        previous = -np.inf
        for it in range(self.config.max_iter):
            log_b = self.log_emissions(obs)
            alpha, scale, b, loglik = self._forward(log_b)
            beta = self._backward(b, scale)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = self.trans * (alpha[:-1].T @ (b[1:] * beta[1:] / scale[1:, None]))
            self._m_step(obs, gamma, xi)
            per_hour = loglik / len(obs)
            self.history.append(per_hour)
            if log:
                log(f"    EM {it + 1}: log-likelihood per hour {per_hour:.4f}")
            if per_hour - previous < self.config.tol:
                break
            previous = per_hour
        return self

    # ------------------------------------------------------------------ features
    def features(self, hourly, t0s):
        """DataFrame of HMM features at each t0, from the in-situ hourly frame.

        `hourly` must cover the hours before every t0 (rows labelled < t0).
        """
        probs = self.filter(observations(hourly))
        state = probs.argmax(axis=1)
        change = np.r_[True, state[1:] != state[:-1]]
        start = np.maximum.accumulate(np.where(change, np.arange(len(state)), 0))
        hours = np.minimum(np.arange(len(state)) - start + 1, MAX_DURATION)
        row = hourly.index.get_indexer(pd.DatetimeIndex(t0s) - pd.Timedelta(hours=1))
        ok = row >= 0
        out = pd.DataFrame(np.nan, index=pd.DatetimeIndex(t0s), columns=self.feature_names)
        values = np.column_stack([probs, hours])
        out.iloc[np.flatnonzero(ok)] = values[row[ok]]
        return out

    @property
    def feature_names(self):
        return [*(f"hmm_p_{n}" for n in self.names), "hmm_state_hours"]

    def summary(self):
        """Per-state parameters as a DataFrame (for checking the states make sense)."""
        stay = np.diag(self.trans)
        return pd.DataFrame({
            "speed mean": self.m, "speed sd": np.sqrt(self.sd2),
            "AR level c/(1-phi)": self.c / np.maximum(1 - self.phi, 1e-6), "AR phi": self.phi,
            "AR sd": np.sqrt(self.sigma2), "T ratio": np.exp(self.mu),
            "Bt (nT)": 10**self.nu, "mean stay (h)": 1 / np.maximum(1 - stay, 1e-9),
            "stationary share": self.start}, index=list(self.names))

    # ------------------------------------------------------------------ persistence
    _PARAMS = ("trans", "start", "c", "phi", "sigma2", "m", "sd2", "mu", "tau2", "nu", "omega2")

    def to_dict(self):
        return {"model": "RegimeHMM", "format": FORMAT_VERSION, "config": asdict(self.config),
                **{p: np.asarray(getattr(self, p)).tolist() for p in self._PARAMS}}

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "RegimeHMM" or d.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported RegimeHMM asset")
        model = cls(Config(**d["config"]))
        for p in cls._PARAMS:
            setattr(model, p, np.asarray(d[p], dtype=float))
        return model

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def stationary(trans):
    """The stationary distribution of a transition matrix."""
    w, v = np.linalg.eig(trans.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    p = np.abs(p)
    return p / p.sum()


def _speed_quantiles(speed, k):
    filled = np.where(np.isfinite(speed), speed, np.nanmedian(speed))
    edges = np.nanquantile(speed, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, filled)
