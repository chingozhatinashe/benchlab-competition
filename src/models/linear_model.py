"""Per-lead penalised generalised linear model (GLM) on speed history and regressors.

For each lead k = 1..72 h at t0:

    eta_k = offset + a_k + sum_j b_kj * s_j(k) + sum_i c_ki * x_i
    E[y_k] = g^-1(eta_k)

    s_j(k)   speed terms, always used: the last observed speed and the mean speed within
             +-half_width h of t0 + k - lag_j (27 d, 13.5 d), centred on the training
             mean after the family's transform (Box-Cox for Gaussian, log for the GLMs);
             a missing speed term becomes 0, i.e. the mean
    x_i      the selected regressors (feature_engineering.windows specs), standardised
             with the training means and standard deviations

Families (Config.family):
    gaussian          identity link on a Box-Cox target BC_lambda(y) (lambda = 1: none,
                      0: log). relative=True predicts BC(y) - BC(last) instead, so the
                      penalty shrinks towards persistence. The forecast is back-transformed
                      with a smearing estimate (Duan 1983): the mean of BC^-1(z + e) over
                      quantiles e of the training residuals, which removes the bias of
                      inverting a non-linear transform.
    gamma             log link on y; variance proportional to mu^2.
    inverse_gaussian  log link on y; variance proportional to mu^3.
    For the log-link families relative=True adds the offset log(last), so
    E[y] = last * exp(X beta).

Every family is fitted by iteratively reweighted least squares (IRLS), one batched
weighted ridge solve per iteration over all leads (Gaussian needs one). The loss
weights are each run's TIMSE weights (models.final_baseline.run_timse_weights) or
uniform, as prior weights. The ridge penalty is relative to each column's weighted
second moment and spares the intercept, as in the final baseline.

Training pipeline: regressors -> selection (Config.features) -> GLM; rows with a
missing selected regressor are dropped. Live pipeline: regressors -> imputation
with the training medians -> selection -> GLM. Inference is numpy only: one matrix
product and the inverse link.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from feature_engineering.recurrence import HORIZON, LEADS, last_observed, targets, window_means
from models.final_baseline import TAIL_FLOOR, VALID_RANGE, run_timse_weights, solve

ASSET = Path(__file__).resolve().parents[1] / "assets" / "linear_model.json"
FORMAT_VERSION = 1
FAMILIES = ("gaussian", "gamma", "inverse_gaussian")
WEIGHTS = ("run_timse", "uniform")
SMEAR_QUANTILES = np.linspace(0.025, 0.975, 39)
ETA_BOUNDS = (np.log(50.0), np.log(5000.0))  # keeps the log-link IRLS away from overflow


@dataclass(frozen=True)
class Config:
    features: tuple = ()  # selected regressor names (feature_engineering.windows)
    family: str = "gaussian"
    boxcox: float = 1.0  # Box-Cox lambda of the Gaussian target; 1 = untransformed
    relative: bool = False  # relative to persistence (target difference or log offset)
    weights: str = "run_timse"
    ridge: float = 1.0
    lags_days: tuple = (27.0, 13.5)
    half_width: int = 3
    max_iter: int = 30
    tol: float = 1e-6

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.weights not in WEIGHTS:
            raise ValueError(f"weights must be one of {WEIGHTS}")

    @property
    def log_link(self):
        return self.family != "gaussian"

    @property
    def lag_hours(self):
        return tuple(int(round(d * 24)) for d in self.lags_days)

    @property
    def history_hours(self):
        return max(self.lag_hours) + self.half_width + 48

    @property
    def n_speed(self):
        return 1 + len(self.lags_days)

    @property
    def n_columns(self):
        return 1 + self.n_speed + len(self.features)


def boxcox(y, lam):
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(y) if lam == 0 else (y**lam - 1.0) / lam


def inv_boxcox(z, lam):
    z = np.asarray(z, dtype=float)
    if lam == 0:
        return np.exp(np.clip(z, -50, 50))
    return np.maximum(lam * z + 1.0, 1e-9) ** (1.0 / lam)


def weighted_quantiles(values, weights, qs):
    """Quantiles of each column of values (n x k) under weights (n x k); zero weight = absent."""
    out = np.zeros((values.shape[1], len(qs)))
    for k in range(values.shape[1]):
        w = weights[:, k]
        ok = w > 0
        if not ok.any():
            continue
        v, w = values[ok, k], w[ok]
        order = np.argsort(v)
        v, w = v[order], w[order]
        cum = (np.cumsum(w) - 0.5 * w) / w.sum()
        out[k] = np.interp(qs, cum, v)
    return out


class LinearModel:
    """fit(values, positions, regressors) / predict(history, features)."""

    def __init__(self, config=None):
        self.config = config or Config()

    # ------------------------------------------------------------------ design
    def _speed_transform(self, v):
        lam = 0.0 if self.config.log_link else self.config.boxcox
        return boxcox(v, lam)

    def speed_terms(self, values, positions):
        """(n, 72, n_speed) transformed speed terms, centred; missing -> 0 (the mean)."""
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        centre = self._speed_transform(self.mean)
        out = np.zeros((len(positions), HORIZON, cfg.n_speed))
        last = last_observed(values, positions)
        s = self._speed_transform(last) - centre
        out[:, :, 0] = np.where(np.isfinite(s), s, 0.0)[:, None]
        for j, lag in enumerate(cfg.lag_hours):
            r = window_means(values, positions[:, None] + LEADS - lag, cfg.half_width)
            s = self._speed_transform(r) - centre
            out[:, :, 1 + j] = np.where(np.isfinite(s), s, 0.0)
        return out, last

    def design(self, values, positions, regressors):
        """(n, 72, n_columns): intercept, speed terms, standardised regressors.

        `regressors` (n x p, in Config.features order) must already be complete
        (fit: incomplete rows dropped; predict: imputed).
        """
        speed, last = self.speed_terms(values, positions)
        n = len(positions)
        x = np.ones((n, HORIZON, self.config.n_columns))
        x[:, :, 1 : 1 + self.config.n_speed] = speed
        if self.config.features:
            z = (np.asarray(regressors, dtype=float) - self.x_mean) / self.x_std
            x[:, :, 1 + self.config.n_speed :] = z[:, None, :]
        return x, last

    def _offset(self, last):
        """Per-run offset (log link) or base (Gaussian) for relative models; 0 otherwise."""
        cfg = self.config
        if not cfg.relative:
            return np.zeros(len(last))
        filled = np.where(np.isfinite(last), last, self.mean)
        return np.log(filled) if cfg.log_link else boxcox(filled, cfg.boxcox)

    # ------------------------------------------------------------------ fitting
    def loss_weights(self, y):
        observed = np.isfinite(y)
        if self.config.weights == "uniform":
            return observed.astype(float)
        w = run_timse_weights(y)
        scale = w[observed].mean() if observed.any() else 1.0
        return np.where(observed, np.maximum(w / max(scale, 1e-12), TAIL_FLOOR), 0.0)

    def fit(self, values, positions, regressors=None, end=None):
        """Fit on the runs at `positions` of the hourly speed array `values`.

        Targets at or after `end` (a position) are ignored, so a fold's fit never sees
        its validation hours. Runs with a missing selected regressor are dropped.
        """
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        x_raw = np.zeros((len(positions), 0)) if not cfg.features else np.asarray(
            regressors, dtype=float)
        if x_raw.shape != (len(positions), len(cfg.features)):
            raise ValueError(f"regressors shape {x_raw.shape}, expected "
                             f"({len(positions)}, {len(cfg.features)})")
        hist_end = len(values) if end is None else end
        self.mean = float(np.nanmean(values[:hist_end]))
        self.medians = (np.nanmedian(x_raw, axis=0) if cfg.features else np.zeros(0))
        complete = np.isfinite(x_raw).all(axis=1)
        x_raw, positions = x_raw[complete], positions[complete]
        self.x_mean = x_raw.mean(axis=0) if cfg.features else np.zeros(0)
        std = x_raw.std(axis=0) if cfg.features else np.zeros(0)
        self.x_std = np.where(std > 1e-12, std, 1.0)

        x, last = self.design(values, positions, x_raw)
        y = targets(values, positions, end)
        q = self.loss_weights(y)
        offset = self._offset(last)
        if cfg.relative:
            q = np.where(np.isfinite(last)[:, None], q, 0.0)  # no persistence to be relative to
        y0 = np.where(q > 0, y, 1.0)

        if cfg.log_link:
            self.coef = self._irls(x, y0, q, offset)
            self.smear = None
        else:
            t = boxcox(y0, cfg.boxcox) - offset[:, None]
            self.coef = self._wls(x, t, q)
            resid = t - np.einsum("nkf,kf->nk", x, self.coef)
            self.smear = weighted_quantiles(np.where(q > 0, resid, 0.0), q, SMEAR_QUANTILES)
        self.fit_info = {"runs": int(len(positions)), "dropped_incomplete": int((~complete).sum()),
                         "targets": int((q > 0).sum())}
        return self

    def _wls(self, x, t, w):
        a = np.einsum("nkf,nk,nkg->kfg", x, w, x)
        b = np.einsum("nkf,nk->kf", x, w * np.where(w > 0, t, 0.0))
        return solve(a, b, self.config.ridge)

    def _irls(self, x, y, q, offset):
        cfg = self.config
        mean_y = (q * y).sum(axis=0) / np.maximum(q.sum(axis=0), 1e-12)
        coef = np.zeros((HORIZON, x.shape[-1]))
        coef[:, 0] = np.log(np.maximum(mean_y, 1.0)) - offset.mean()
        for _ in range(cfg.max_iter):
            eta = np.clip(np.einsum("nkf,kf->nk", x, coef) + offset[:, None], *ETA_BOUNDS)
            mu = np.exp(eta)
            z = eta - offset[:, None] + (y - mu) / mu
            w = q if cfg.family == "gamma" else q / mu  # q * mu^2 / V(mu)
            new = self._wls(x, z, w)
            step = np.abs(new - coef).max()
            coef = new
            if step < cfg.tol:
                break
        return coef

    # ------------------------------------------------------------------ forecasting
    def impute(self, regressors):
        """Live pipeline: missing regressors -> training medians."""
        x = np.asarray(regressors, dtype=float).reshape(-1, len(self.config.features))
        return np.where(np.isfinite(x), x, self.medians)

    def forecast(self, values, positions, regressors=None):
        """Forecasts (len(positions) x 72), imputing missing regressors with the medians."""
        positions = np.asarray(positions)
        x_raw = (self.impute(regressors) if self.config.features
                 else np.zeros((len(positions), 0)))
        x, last = self.design(values, positions, x_raw)
        eta = np.einsum("nkf,kf->nk", x, self.coef)
        offset = self._offset(last)[:, None]
        if self.config.log_link:
            out = np.exp(np.clip(eta + offset, *ETA_BOUNDS))
        else:
            z = eta + offset
            out = inv_boxcox(z[:, :, None] + self.smear[None, :, :], self.config.boxcox).mean(-1)
        return np.clip(out, *VALID_RANGE)

    def predict(self, history, features=None):
        """72 forecasts from the hourly speed history ending at t0 - 1h and one feature row.

        `features` maps regressor names to values (missing or NaN -> median).
        """
        values = np.asarray(history, dtype=float)
        row = np.array([[np.nan if features is None else _get(features, n)
                         for n in self.config.features]])
        return self.forecast(values, np.array([len(values)]), row)[0]

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        cfg = asdict(self.config)
        cfg["features"] = list(cfg["features"])
        cfg["lags_days"] = list(cfg["lags_days"])
        return {
            "model": "LinearModel",
            "format": FORMAT_VERSION,
            "config": cfg,
            "mean": self.mean,
            "x_mean": np.round(self.x_mean, 10).tolist(),
            "x_std": np.round(self.x_std, 10).tolist(),
            "medians": np.round(self.medians, 10).tolist(),
            "coef": np.round(self.coef, 10).tolist(),
            "smear": None if self.smear is None else np.round(self.smear, 8).tolist(),
            "fit_info": getattr(self, "fit_info", {}),
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT_VERSION or d.get("model") != "LinearModel":
            raise ValueError(f"unsupported model asset {d.get('model')!r} v{d.get('format')!r}")
        cfg = dict(d["config"])
        cfg["features"] = tuple(cfg["features"])
        cfg["lags_days"] = tuple(cfg["lags_days"])
        model = cls(Config(**cfg))
        model.mean = float(d["mean"])
        model.x_mean = np.asarray(d["x_mean"], dtype=float)
        model.x_std = np.asarray(d["x_std"], dtype=float)
        model.medians = np.asarray(d["medians"], dtype=float)
        model.coef = np.asarray(d["coef"], dtype=float)
        model.smear = None if d["smear"] is None else np.asarray(d["smear"], dtype=float)
        model.fit_info = d.get("fit_info", {})
        if model.coef.shape != (HORIZON, model.config.n_columns):
            raise ValueError(f"coef shape {model.coef.shape}")
        return model

    def save(self, path=ASSET):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path=ASSET):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _get(features, name):
    try:
        v = features[name]
    except (KeyError, IndexError):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
