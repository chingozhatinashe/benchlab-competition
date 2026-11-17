"""Structural model v3: a Bayesian generative model of the next 72 h of solar-wind speed.

For a run at t0 (inputs x(u) for hour t0 + u as models.structural, held at lead 72 for
u = 73..84):

    slow wind     a = 250 + 250 sigmoid(w_a . [1, q20, rec27])            (ambient level)
                  S(k) = a + rho^k (v0 - a),  rho = rho_up above a, rho_dn below
    streams       N ~ Poisson(Lambda) truncated at N <= nmax, log Lambda = w_N . mean_k x(k)
      times       stick-breaking over the arrival hours u = 1..84: v_u = sigmoid(w_tau . x(u)),
                  pi_u ~ v_u prod_{s<u} (1 - v_s), normalised; a second arrival comes from
                  the same stick restricted to u >= tau_1 + GAP (the first to u <= 84 - GAP)
      marks       excess E = exp(mu_E + w_E . x'(tau) + sigma_E z), rise width
                  R = exp(mu_R + sigma_R z), decay D = exp(mu_D + w_D . x'(tau) + sigma_D z)
                  (x': x without its constant; Config.event_rise=False: R = exp(mu_R),
                  sigma_R = 0)
      profile     s_j(k) = a + E g((k - tau) / R) exp(-max(0, k - tau - 2R) / D), g logistic;
                  tau is the midpoint of the rise, as the detected arrivals
    speed         V = smoothmax(S, s_1, ..., s_N) + eta (log-sum-exp at TEMP km/s; rule
                  "max") or S + sum_j (s_j - a) + eta (rule "add"), eta an AR(1) process
                  (rho_eta, sigma_eta) started at 0 (the observed speed at t0); clipped to
                  VALID_RANGE. With N = 0 a path is the slow wind S + eta. The noise is added
                  after the streams, so the training likelihood is exact.

`draws` holds posterior draws (training.bayes_stream fits them with NumPyro NUTS). The
forecast simulates paths (draw, N, times, marks, AR noise) and returns a point decision:
the predictive mean, or the Bayes action under the official-weighted squared loss
(per lead sum w(y) y / sum w(y) over the paths, w from the fit period's weights).
The shared functions take an array namespace `xp` (numpy here, jax.numpy in training).
See docs/bayes_stream.md.
"""

from dataclasses import asdict, dataclass

import numpy as np

from models import structural as st
from models.final_baseline import VALID_RANGE, official_weights

FORMAT_VERSION = 1
HORIZON = st.HORIZON
LEADS = st.LEADS
N_ARRIVAL = 84  # arrival hours 1..84: a rise centred after t0 + 72 starts inside the window
ARRIVALS = np.arange(1, N_ARRIVAL + 1, dtype=float)
NEG = -1e4  # stands in for log 0 (finite, so gradients stay finite)
GLOBALS = ("w_a", "rho_up", "rho_dn", "rho_eta", "sigma_eta", "w_N", "w_tau", "mu_E", "w_E",
           "sigma_E", "mu_R", "sigma_R", "mu_D", "w_D", "sigma_D")
WEIGHT_GRID = np.arange(200.0, 1201.0, 5.0)  # km/s, for the official decision weights


@dataclass(frozen=True)
class Config:
    rule: str = "max"  # "max" (smooth max) or "add"
    nmax: int = 2  # most arrivals per window
    gap: float = 48.0  # h between two arrivals
    temp: float = 5.0  # km/s, smooth-max temperature
    latent: bool = False  # trained with the arrivals summed out (marks without event effects)
    event_rise: bool = True  # per-event rise widths; False: every event at exp(mu_R)
    # training priors (the draws carry their result): a fixed rise width in h instead of
    # fitting mu_R, and the prior sd of the decay inputs w_D
    rise_h: float | None = None
    w_D_sd: float = 0.3


# ---------------------------------------------------------------------- shared (numpy / jax)
def sigmoid(z, xp):
    return 1.0 / (1.0 + xp.exp(-z))


def log_sigmoid(z, xp):
    return -xp.logaddexp(0.0, -z)


def ambient(w_a, amb, xp):
    lo, hi = st.AMBIENT_RANGE
    return lo + (hi - lo) * sigmoid(amb @ w_a, xp)


def slow_mean(a, v0, rho_up, rho_dn, xp):
    """(..., 72): a + rho^k (v0 - a)."""
    e0 = v0 - a
    rho = xp.where(e0 > 0, rho_up, rho_dn)
    return a[..., None] + rho[..., None] ** LEADS * e0[..., None]


def profile(tau, E, R, D, xp):
    """(..., 72) stream excess for arrivals at hours `tau` (any shape)."""
    d = LEADS - tau[..., None]
    rise = sigmoid(d / R[..., None], xp)
    decay = xp.exp(-xp.maximum(0.0, d - 2.0 * R[..., None]) / D[..., None])
    return E[..., None] * rise * decay


def combine(slow, a, excess, present, cfg, xp):
    """Slow path (..., 72) with stream excesses (..., J, 72) where `present` (..., J)."""
    if cfg.rule == "add":
        return slow + (present[..., None] * excess).sum(axis=-2)
    t = cfg.temp
    out = slow / t
    for j in range(excess.shape[-2]):
        stream = xp.where(present[..., j, None] > 0, a[..., None] + excess[..., j, :], NEG)
        out = xp.logaddexp(out, stream / t)
    return t * out


def _logsumexp(z, axis, xp):
    m = z.max(axis=axis, keepdims=True)
    return (m + xp.log(xp.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def count_logp(log_lam, nmax, xp):
    """(..., nmax + 1) log P(N = n) for a Poisson(Lambda) truncated at nmax."""
    n = np.arange(nmax + 1, dtype=float)
    lgamma = np.cumsum(np.log(np.maximum(n, 1.0)))
    lp = log_lam[..., None] * n - lgamma
    return lp - _logsumexp(lp, -1, xp)[..., None]


def stick_logp(logit, xp):
    """(..., 84) log pi_u of the stick-breaking arrival hour, normalised over the window."""
    log_v = log_sigmoid(logit, xp)
    log_stay = log_sigmoid(-logit, xp)
    before = xp.cumsum(log_stay, axis=-1) - log_stay
    lp = log_v + before
    return lp - _logsumexp(lp, -1, xp)[..., None]


def restrict(lp, allowed, xp):
    """Renormalise log probabilities over the `allowed` hours."""
    lp = xp.where(allowed, lp, NEG)
    return lp - _logsumexp(lp, -1, xp)[..., None]


def ar1_loglik(r, rho, sigma, xp, gaps=None):
    """(...) Gaussian AR(1) log likelihood of residual paths r (..., 72), started at 0.
    `gaps` = (prev, lag, observed) from `gap_structure` (broadcastable to r) skips missing
    hours exactly: over a gap of g hours the step is rho^g with variance
    sigma^2 (1 - rho^2g) / (1 - rho^2)."""
    if gaps is None:
        prev = xp.concatenate([xp.zeros_like(r[..., :1]), r[..., :-1]], axis=-1)
        e = r - rho * prev
        return (-0.5 * (e / sigma) ** 2 - xp.log(sigma) - 0.5 * np.log(2 * np.pi)).sum(axis=-1)
    prev, lag, observed = gaps
    padded = xp.concatenate([xp.zeros_like(r[..., :1]), r], axis=-1)
    r_prev = xp.take_along_axis(padded, xp.broadcast_to(prev + 1, r.shape), axis=-1)
    var = sigma ** 2 * (1.0 - rho ** (2 * lag)) / (1.0 - rho ** 2)
    e = r - rho ** lag * r_prev
    ll = -0.5 * e ** 2 / var - 0.5 * xp.log(var) - 0.5 * np.log(2 * np.pi)
    return (observed * ll).sum(axis=-1)


def gap_structure(observed):
    """(prev, lag, observed) for ar1_loglik: the index of the previous observed hour (-1:
    t0, where the residual is 0) and the hours since it, for observed (n, 72) bool."""
    observed = np.asarray(observed, dtype=bool)
    k = np.arange(observed.shape[-1])
    seen = np.maximum.accumulate(np.where(observed, k, -1), axis=-1)
    prev = np.concatenate([np.full(observed.shape[:-1] + (1,), -1), seen[..., :-1]], axis=-1)
    return prev, (k - prev).astype(float), observed.astype(float)


def extend(x):
    """(n, 72, p) -> (n, 84, p): inputs at lead 72 held for the hours after the window."""
    return np.concatenate([x, np.repeat(x[:, -1:], N_ARRIVAL - HORIZON, axis=1)], axis=1)


# ---------------------------------------------------------------------- simulation (numpy)
def _categorical(rng, lp):
    """Sample indices from log probabilities lp (..., m)."""
    p = np.exp(lp - lp.max(axis=-1, keepdims=True))
    cdf = np.cumsum(p, axis=-1)
    u = rng.random(lp.shape[:-1])[..., None] * cdf[..., -1:]
    return np.minimum((u > cdf).sum(axis=-1), lp.shape[-1] - 1)


def simulate_paths(draws, inp, cfg, n_paths, rng, slow_only=False, details=False):
    """(n, P, 72) simulated speeds and (n, P) arrivals within hours 1..72, P = n_paths.
    Path j uses posterior draw idx[j] (a random subset or cycle of the draws)."""
    x, amb, v0 = inp["x"], inp["amb"], inp["last"]
    n_draws = len(draws["rho_up"])
    idx = rng.permutation(np.resize(np.arange(n_draws), n_paths))
    d = {k: draws[k][idx] for k in GLOBALS}
    x84 = extend(x)
    n = len(v0)
    a = ambient(d["w_a"].T, amb, np)  # (n, P)
    slow = slow_mean(a, v0[:, None], d["rho_up"], d["rho_dn"], np)
    eta = np.zeros((n, n_paths, HORIZON))
    prev = np.zeros((n, n_paths))
    for k in range(HORIZON):
        prev = d["rho_eta"] * prev + d["sigma_eta"] * rng.standard_normal((n, n_paths))
        eta[:, :, k] = prev
    if slow_only:
        return np.clip(slow + eta, *VALID_RANGE), np.zeros((n, n_paths))
    log_lam = x.mean(axis=1) @ d["w_N"].T
    count = _categorical(rng, count_logp(log_lam, cfg.nmax, np))  # (n, P)
    lp = stick_logp(np.einsum("nup,Pp->nPu", x84, d["w_tau"]), np)
    first = restrict(lp, (count[..., None] < 2) | (N_ARRIVAL - cfg.gap >= ARRIVALS), np)
    tau1 = _categorical(rng, first) + 1
    second = restrict(lp, tau1[..., None] + cfg.gap <= ARRIVALS, np)
    tau2 = _categorical(rng, second) + 1
    taus = np.stack([tau1, tau2], axis=-1).astype(float)  # (n, P, 2)
    present = np.stack([count >= 1, count >= 2], axis=-1).astype(float)
    xc = x84[:, :, 1:]
    rows = np.arange(n)[:, None, None]
    xt = xc[rows, taus.astype(int) - 1]  # (n, P, 2, p - 1)
    z = rng.standard_normal((3, n, n_paths, 2))
    E = np.exp(d["mu_E"][:, None] + np.einsum("nPjq,Pq->nPj", xt, d["w_E"])
               + d["sigma_E"][:, None] * z[0])
    R = np.exp(d["mu_R"][:, None] + d["sigma_R"][:, None] * z[1])
    D = np.exp(d["mu_D"][:, None] + np.einsum("nPjq,Pq->nPj", xt, d["w_D"])
               + d["sigma_D"][:, None] * z[2])
    excess = profile(taus, E, R, D, np)
    v = combine(slow, a, excess, present, cfg, np) + eta
    arrived = (present * (taus <= HORIZON)).sum(axis=-1)
    if details:
        return np.clip(v, *VALID_RANGE), arrived, taus, present
    return np.clip(v, *VALID_RANGE), arrived


def weight_table(y_fit, blend):
    """(72, len(WEIGHT_GRID)) official loss weights as a function of speed, per lead, from
    the fit runs' targets (weights rise with the speed's percentile)."""
    y = np.where(np.isfinite(y_fit), y_fit, np.nan)
    w = official_weights(y, blend)
    out = np.empty((HORIZON, len(WEIGHT_GRID)))
    for k in range(HORIZON):
        ok = np.isfinite(y[:, k])
        order = np.argsort(y[ok, k])
        out[k] = np.interp(WEIGHT_GRID, y[ok, k][order], w[ok, k][order])
    return out


def decisions(paths, table=None):
    """Point forecasts from paths (n, P, 72): the mean and the official-loss Bayes action."""
    out = {"mean": paths.mean(axis=1)}
    if table is not None:
        w = np.stack([np.interp(paths[:, :, k], WEIGHT_GRID, table[k])
                      for k in range(HORIZON)], axis=-1)
        out["official"] = (w * paths).sum(axis=1) / w.sum(axis=1)
    return out


def crps(paths, obs):
    """(n, 72) sample CRPS of paths (n, P, 72) against obs (n, 72) (NaN where missing)."""
    p = np.sort(paths, axis=1)
    m = p.shape[1]
    first = np.abs(p - obs[:, None, :]).mean(axis=1)
    i = np.arange(1, m + 1)[None, :, None]
    spread = (2 * (2 * i - m - 1) * p).sum(axis=1) / m ** 2
    return first - 0.5 * spread


def summarize_paths(draws, inp, cfg, table=None, obs=None, n_paths=400, chunk=100, seed=0,
                    slow_only=False):
    """Point forecasts, quantiles, P(arrival in 72 h), expected arrivals (and CRPS and the
    80% coverage against `obs`), simulated in chunks of runs."""
    rng = np.random.default_rng(seed)
    n = len(inp["last"])
    parts = []
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        sub = {k: (v[sl] if k in ("x", "amb", "last") else v) for k, v in inp.items()}
        paths, arrived = simulate_paths(draws, sub, cfg, n_paths, rng, slow_only)
        out = decisions(paths, table)
        q = np.quantile(paths, [0.1, 0.5, 0.9], axis=1)
        out.update(q10=q[0], q50=q[1], q90=q[2], p_any=(arrived > 0).mean(axis=1),
                   arrivals=arrived.mean(axis=1))
        if obs is not None:
            o = obs[sl]
            out["crps"] = crps(paths, o)
            out["covered"] = np.where(np.isfinite(o), (o >= q[0]) & (o <= q[2]), np.nan)
        parts.append(out)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class BayesStreamModel(st.StructuralModel):
    """Input preparation from models.structural; posterior draws from training.bayes_stream."""

    def __init__(self, config=None, bayes=None):
        super().__init__(config)
        self.bayes = bayes or Config()
        self.params = {}
        self.draws = None
        self.table = None

    def forecast(self, values, positions, regressors=None, decision="mean", n_paths=400,
                 seed=0):
        inp = self.prepare(values, positions, regressors)
        out = summarize_paths(self.draws, inp, self.bayes, self.table, n_paths=n_paths,
                              seed=seed)
        return out[decision]

    def to_dict(self):
        d = super().to_dict()
        d.update(format=FORMAT_VERSION, kind="bayes_stream", bayes=asdict(self.bayes),
                 draws={k: np.asarray(v).tolist() for k, v in self.draws.items()},
                 table=None if self.table is None else self.table.tolist())
        d.pop("params", None)
        return d

    @classmethod
    def from_dict(cls, d):
        base = st.StructuralModel.from_dict({**d, "params": {}})
        m = cls(base.config, Config(**d["bayes"]))
        m.pre = base.pre
        m.draws = {k: np.asarray(v, dtype=float) for k, v in d["draws"].items()}
        m.table = None if d["table"] is None else np.asarray(d["table"], dtype=float)
        return m
