"""Gaussian-process correction to the linear model, evaluated in numpy.

Status (2026-10-22): experimental and not used live. On the training folds the
ML-II-fitted version scored about 0.63 against 0.4614 for the linear model alone:
marginal likelihood picks short lengthscales and little noise, and the GP undoes
the linear model's TIMSE hedge (it corrects "quiet-looking" runs downwards, which
the storms among them make expensive). With lengthscale and noise chosen on the
folds instead, the best correction (one GP on each run's TIMSE-weighted mean
offset, the XGBoost hybrid's 25 inputs) reached 0.4587, a 0.6% gain; the
XGBoost hybrid gains 1.9%.

    forecast(k) = linear(k) + sum_j c_j(x) * phi_j(k)

linear(k)   the shipped linear model (models.linear_model), stored inside this asset
phi_j(k)    the leading principal shapes of the training residual trajectories
            (observed - linear, 72 leads), from a PCA weighted by each run's TIMSE
            weights (ResidualBasis); each training run's residual becomes K scores
c_j(x)      one zero-mean GP per score, Matern 5/2 kernel with one lengthscale per
            input (ARD); the inputs x are the standardised regressors plus the last
            observed speed and the linear model's mean offset from persistence

Each run's score is noisy with variance sigma_n^2 / w_i^gamma, w_i the run's mean
TIMSE weight: the GP counterpart of the TIMSE-weighted loss, fast runs fitted more
closely. gamma (Config.weight_power) tempers the heavy-tailed weights; gamma = 1
lets a few storm runs set near-zero noise and very short lengthscales.

Every mean is zero: the residual shapes are not centred and each GP has a zero
prior mean, so far from the training runs (or with no information) the correction
is zero and the forecast is the linear model's.

Hyper-parameters (sigma_f, lengthscales, sigma_n) maximise the marginal likelihood
(ML-II) on a random subsample of the fit runs; the posterior then conditions on
every fit run.

A GP needs complete inputs: a missing regressor takes its training median (the
linear anchor imputes its own). Fitting (dev only) needs scipy; forecasting is
numpy only: the posterior mean k(x*, X) alpha from the stored X and alpha.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from feature_engineering.recurrence import HORIZON, last_observed, targets
from models.final_baseline import VALID_RANGE
from models.linear_model import LinearModel

ASSET = Path(__file__).resolve().parents[1] / "assets" / "gaussian_process.json"
FORMAT_VERSION = 1
EXTRA_INPUTS = ("last", "offset")  # last observed speed; mean linear - last over leads
LOG_BOUNDS = {"lengthscale": (np.log(0.05), np.log(100.0)), "sf2": (np.log(1e-4), np.log(1e8)),
              "sn2": (np.log(1e-4), np.log(1e8))}


@dataclass(frozen=True)
class Config:
    features: tuple = ()  # regressor names (feature_engineering.windows)
    n_components: int = 5  # principal residual shapes used to forecast
    max_components: int = 8  # shapes fitted (selection scores fewer from the same fit)
    ml_runs: int = 1500  # runs in the ML-II subsample
    restarts: int = 2  # ML-II starts: lengthscales 1, then 3 (standardised units)
    max_iter: int = 200
    jitter: float = 1e-6
    seed: int = 0
    weight_power: float = 1.0  # noise variance sigma_n^2 / w^gamma

    def __post_init__(self):
        if not 1 <= self.n_components <= self.max_components <= HORIZON:
            raise ValueError("need 1 <= n_components <= max_components <= 72")

    @property
    def inputs(self):
        return (*self.features, *EXTRA_INPUTS)


# ---------------------------------------------------------------------- kernel
def scaled_sq_dist(a, b, lengthscales):
    a = np.asarray(a, dtype=float) / lengthscales
    b = np.asarray(b, dtype=float) / lengthscales
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d2, 0.0)


def matern52(a, b, lengthscales, sf2):
    """Matern 5/2 kernel with ARD lengthscales and signal variance sf2."""
    s = np.sqrt(5.0 * scaled_sq_dist(a, b, lengthscales))
    return sf2 * (1.0 + s + s**2 / 3.0) * np.exp(-s)


# ---------------------------------------------------------------------- output basis
class ResidualBasis:
    """Leading principal shapes of residual trajectories, weighted by TIMSE weights."""

    def __init__(self, shapes, mean):
        self.shapes = np.asarray(shapes, dtype=float)  # (72, K)
        self.mean = np.asarray(mean, dtype=float)  # (72,)

    @classmethod
    def fit(cls, residuals, weights, n_components):
        """Shapes of the weighted second moment (not centred: zero means no correction)."""
        q = np.where(np.isfinite(residuals), weights, 0.0)
        r = np.where(q > 0, residuals, 0.0)
        mean = np.zeros(r.shape[1])
        w = q.mean(axis=1)
        cov = (r * w[:, None]).T @ r / max(w.sum(), 1e-12)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:n_components]
        shapes = vecs[:, order]
        shapes *= np.sign(shapes.sum(axis=0) + 1e-12)  # a fixed sign: positive sum
        basis = cls(shapes, mean)
        basis.explained = vals[order] / max(vals.sum(), 1e-12)
        return basis

    def scores(self, residuals, weights):
        """Weighted least-squares scores per run from its observed hours, and run weights."""
        q = np.where(np.isfinite(residuals), weights, 0.0)
        r = np.where(q > 0, residuals - self.mean, 0.0)
        a = np.einsum("nk,kj,kl->njl", q, self.shapes, self.shapes)
        a += 1e-9 * np.eye(self.shapes.shape[1])
        b = np.einsum("nk,kj->nj", q * r, self.shapes)
        return np.linalg.solve(a, b[..., None])[..., 0], q.mean(axis=1)

    def trajectories(self, scores):
        k = scores.shape[1]
        return self.mean + scores @ self.shapes[:, :k].T


# ---------------------------------------------------------------------- GP (one output)
class GP:
    """Zero-mean GP regression with per-point noise sn2 / w; posterior mean only."""

    def __init__(self, lengthscales, sf2, sn2, x=None, alpha=None, mean=0.0):
        self.lengthscales = np.asarray(lengthscales, dtype=float)
        self.sf2 = float(sf2)
        self.sn2 = float(sn2)
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.alpha = None if alpha is None else np.asarray(alpha, dtype=float)
        self.mean = float(mean)

    def predict(self, x, chunk=2048):
        x = np.asarray(x, dtype=float)
        out = np.empty(len(x))
        for s in range(0, len(x), chunk):
            k = matern52(x[s : s + chunk], self.x, self.lengthscales, self.sf2)
            out[s : s + chunk] = k @ self.alpha
        return out + self.mean

    def condition(self, x, y, w, jitter=1e-6):
        """Posterior weights alpha = (K + diag(sn2 / w))^-1 y (zero prior mean)."""
        from scipy.linalg import cho_factor, cho_solve

        self.x = np.asarray(x, dtype=float)
        self.mean = 0.0
        a = matern52(self.x, self.x, self.lengthscales, self.sf2)
        a[np.diag_indices_from(a)] += self.sn2 / w + jitter * self.sf2
        self.alpha = cho_solve(cho_factor(a, lower=True, overwrite_a=True), y - self.mean)
        return self


def neg_log_marginal(log_params, x, y, w, jitter=1e-6):
    """-log p(y | x) and its gradient; log_params = (log lengthscales, log sf2, log sn2)."""
    from scipy.linalg import cho_factor, cho_solve

    d = x.shape[1]
    ls, sf2, sn2 = np.exp(log_params[:d]), np.exp(log_params[d]), np.exp(log_params[d + 1])
    n = len(y)
    s = np.sqrt(5.0 * scaled_sq_dist(x, x, ls))
    e = np.exp(-s)
    k = sf2 * (1.0 + s + s**2 / 3.0) * e
    noise = sn2 / w
    a = k.copy()
    a[np.diag_indices_from(a)] += noise + jitter * sf2
    try:
        c = cho_factor(a, lower=True)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros_like(log_params)
    alpha = cho_solve(c, y)
    logdet = 2.0 * np.log(np.diag(c[0])).sum()
    nll = 0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
    inv = cho_solve(c, np.eye(n))
    m = np.outer(alpha, alpha) - inv  # dL/dA = m / 2
    grad = np.empty_like(log_params)
    g = m * ((5.0 / 3.0) * sf2 * (1.0 + s) * e)  # dk/dlog l_d = g_ij * (dx_d / l_d)^2
    rows = g.sum(axis=1)
    for j in range(d):
        xd = x[:, j] / ls[j]
        grad[j] = -0.5 * (2.0 * (xd**2 * rows).sum() - 2.0 * xd @ g @ xd)
    grad[d] = -0.5 * (m * (k + jitter * sf2 * np.eye(n))).sum()
    grad[d + 1] = -0.5 * (np.diag(m) * noise).sum()
    return nll, grad


def fit_hyper(x, y, w, restarts=2, max_iter=200, jitter=1e-6):
    """ML-II: maximise the marginal likelihood over (lengthscales, sf2, sn2)."""
    from scipy.optimize import minimize

    d = x.shape[1]
    yc = np.asarray(y, dtype=float)  # zero prior mean
    var = max(np.average(yc**2, weights=w), 1e-6)
    bounds = ([LOG_BOUNDS["lengthscale"]] * d + [LOG_BOUNDS["sf2"], LOG_BOUNDS["sn2"]])
    best, evals = None, 0
    for r in range(restarts):
        start = np.concatenate([np.full(d, np.log(1.0 if r == 0 else 3.0)),
                                [np.log(0.5 * var), np.log(0.5 * var * np.mean(w))]])
        res = minimize(neg_log_marginal, start, args=(x, yc, w, jitter), jac=True,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter})
        evals += res.nfev
        if best is None or res.fun < best.fun:
            best = res
    p = best.x
    return GP(np.exp(p[:d]), np.exp(p[d]), np.exp(p[d + 1])), float(best.fun), evals


# ---------------------------------------------------------------------- the model
class GaussianProcessModel:
    """fit(values, positions, regressors) / predict(history, features)."""

    def __init__(self, config=None, base_model=None):
        if base_model is None:
            raise ValueError("the GP corrects a fitted LinearModel; pass base_model")
        self.config = config or Config()
        self.base_model = base_model
        self.gps = []
        self.basis = None

    @property
    def input_features(self):
        """Regressor columns the model takes: the GP's and the linear model's."""
        return tuple(dict.fromkeys((*self.config.features, *self.base_model.config.features)))

    def _columns(self, regressors, n, names):
        if not names:
            return np.zeros((n, 0))
        reg = np.asarray(regressors, dtype=float).reshape(n, len(self.input_features))
        return reg[:, [self.input_features.index(f) for f in names]]

    def anchor(self, values, positions, regressors=None):
        x = self._columns(regressors, len(positions), self.base_model.config.features)
        return self.base_model.forecast(values, positions, x if x.shape[1] else None)

    def raw_inputs(self, values, positions, regressors, anchor):
        """(n, d) unscaled GP inputs: regressors, last speed, mean anchor offset."""
        last = last_observed(np.asarray(values, dtype=float), np.asarray(positions))
        offset = np.nanmean(anchor, axis=1) - last
        return np.column_stack([self._columns(regressors, len(positions), self.config.features),
                                last, offset])

    def inputs(self, raw):
        """Standardised inputs with missing values at the training median."""
        x = (np.asarray(raw, dtype=float) - self.x_mean) / self.x_std
        return np.where(np.isfinite(x), x, self.x_median)

    # ------------------------------------------------------------------ fitting (dev)
    def fit(self, values, positions, regressors=None, end=None, log=None):
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        hist_end = len(values) if end is None else end
        anchor = self.anchor(values[:hist_end], positions, regressors)
        clean = self.base_model._clean(values[:hist_end])
        y = targets(clean, positions, end)
        q = self.base_model.loss_weights(y)
        raw = self.raw_inputs(values[:hist_end], positions, regressors, anchor)
        self.x_mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        self.x_std = np.where(std > 1e-12, std, 1.0)
        self.x_median = np.nanmedian((raw - self.x_mean) / self.x_std, axis=0)
        self.x_median = np.where(np.isfinite(self.x_median), self.x_median, 0.0)
        x = self.inputs(raw)
        residual = y - anchor
        self.basis = ResidualBasis.fit(residual, q, cfg.max_components)
        scores, w = self.basis.scores(residual, q)
        keep = w > 0
        w = w[keep] ** cfg.weight_power
        x, scores, w = x[keep], scores[keep], w / w.mean()
        rng = np.random.default_rng(cfg.seed)
        sub = rng.choice(len(x), size=min(cfg.ml_runs, len(x)), replace=False)
        self.gps = []
        self.ml_info = []
        for j in range(cfg.max_components):
            gp, nll, evals = fit_hyper(x[sub], scores[sub, j], w[sub], cfg.restarts,
                                       cfg.max_iter, cfg.jitter)
            gp.condition(x, scores[:, j], w, cfg.jitter)
            self.gps.append(gp)
            self.ml_info.append({"component": j, "nll": nll, "sf2": gp.sf2, "sn2": gp.sn2,
                                 "evals": evals})
            if log:
                log(f"    component {j}: explained {self.basis.explained[j]:.1%}, "
                    f"sf2 {gp.sf2:.3g}, sn2 {gp.sn2:.3g}")
        self.fit_info = {"runs": int(keep.sum()), "ml_runs": int(len(sub)),
                         "explained": [round(float(v), 4) for v in self.basis.explained]}
        return self

    # ------------------------------------------------------------------ forecasting
    def correction(self, x, n_components=None):
        k = n_components or self.config.n_components
        scores = np.column_stack([gp.predict(x) for gp in self.gps[:k]])
        return self.basis.trajectories(scores)

    def forecast(self, values, positions, regressors=None, n_components=None):
        """Forecasts (len(positions) x 72): the linear forecast plus the GP correction."""
        positions = np.asarray(positions)
        anchor = self.anchor(values, positions, regressors)
        x = self.inputs(self.raw_inputs(values, positions, regressors, anchor))
        return np.clip(anchor + self.correction(x, n_components), *VALID_RANGE)

    def predict(self, history, features=None):
        """72 forecasts from the hourly history ending at t0 - 1h and one feature row."""
        values = np.asarray(history, dtype=float)
        names = self.input_features
        row = np.array([[np.nan if features is None else _get(features, n) for n in names]])
        return self.forecast(values, np.array([len(values)]), row if names else None)[0]

    # ------------------------------------------------------------------ persistence
    def to_dict(self, n_components=None):
        k = n_components or self.config.n_components
        cfg = asdict(replace(self.config, n_components=k, max_components=k))
        cfg["features"] = list(cfg["features"])
        x = self.gps[0].x
        return {
            "model": "GaussianProcessModel", "format": FORMAT_VERSION, "config": cfg,
            "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
            "x_median": self.x_median.tolist(),
            "basis": {"shapes": np.round(self.basis.shapes[:, :k], 8).tolist(),
                      "mean": np.round(self.basis.mean, 6).tolist()},
            "x": np.round(x, 5).astype(np.float32).tolist(),
            "gps": [{"lengthscales": gp.lengthscales.tolist(), "sf2": gp.sf2, "sn2": gp.sn2,
                     "mean": gp.mean, "alpha": np.round(gp.alpha, 8).tolist()}
                    for gp in self.gps[:k]],
            "base_model": self.base_model.to_dict(),
            "fit_info": getattr(self, "fit_info", {}),
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT_VERSION or d.get("model") != "GaussianProcessModel":
            raise ValueError(f"unsupported model asset {d.get('model')!r} v{d.get('format')!r}")
        cfg = dict(d["config"])
        cfg["features"] = tuple(cfg["features"])
        model = cls(Config(**cfg), LinearModel.from_dict(d["base_model"]))
        model.x_mean = np.asarray(d["x_mean"], dtype=float)
        model.x_std = np.asarray(d["x_std"], dtype=float)
        model.x_median = np.asarray(d["x_median"], dtype=float)
        model.basis = ResidualBasis(d["basis"]["shapes"], d["basis"]["mean"])
        x = np.asarray(d["x"], dtype=float)
        model.gps = [GP(g["lengthscales"], g["sf2"], g["sn2"], x, g["alpha"], g["mean"])
                     for g in d["gps"]]
        model.fit_info = d.get("fit_info", {})
        if len(model.gps) != model.config.n_components:
            raise ValueError("asset has the wrong number of components")
        return model

    def save(self, path=ASSET, n_components=None):
        Path(path).write_text(json.dumps(self.to_dict(n_components), separators=(",", ":"))
                              + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path=ASSET):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _get(features, name):
    try:
        return float(features[name])
    except (KeyError, IndexError, TypeError, ValueError):
        return np.nan
