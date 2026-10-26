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

Config.optional names regressors that can be missing for long stretches (the
coronal-hole strip: the SDO archive has gaps of weeks to months). For them the
model also fits a submodel without them, on every run the other regressors cover,
and a run with an optional regressor missing is forecast by that submodel instead
of imputing the median.

Config.cap_quantile clips each regressor to its training [q, 1 - q] quantiles, in
fitting and forecasting, so an extreme value outside the training range (a 61 nT field
on 2026-01-20) can't be extrapolated linearly into a large forecast error.

Lead-aligned regressors (named "<signal>_lead_<D>h", e.g. ch_cm_lead_72h) take, for lead
k, the signal at lag L_k = D - k before t0, rounded up to the 6-hour image grid and at
least 0: the image about D hours before the hour being forecast (training.lead_aligned_ch).
The model reads them from the lagged columns "<signal>_at_<L>h" (Config.input_features),
one column per distinct L_k, and puts each lead's own column in its design. A missing
lagged value is filled from the nearest lag of the same run within LEAD_FILL_HOURS;
standardising, the cap and the median imputation are per input column.

Config.despike removes isolated one-hour glitches from the speed series
(feature_engineering.recurrence.despike) before it is used as a fitting target or
in the recurrence terms. The last observed speed stays raw: its hour can't be
checked against the hours after it. Scoring always uses the raw series.
"""

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from feature_engineering.recurrence import (
    HORIZON,
    LEADS,
    SPIKE_SIDE,
    despike,
    last_observed,
    targets,
    window_means,
)
from models.final_baseline import (
    TAIL_FLOOR,
    VALID_RANGE,
    official_weights,
    run_timse_weights,
    solve,
)

ASSET = Path(__file__).resolve().parents[1] / "assets" / "linear_model.json"
FORMAT_VERSION = 2
READABLE_FORMATS = (1, 2)  # 1: no submodel, no despike
FAMILIES = ("gaussian", "gamma", "inverse_gaussian")
WEIGHTS = ("run_timse", "uniform", "official")
SMEAR_QUANTILES = np.linspace(0.025, 0.975, 39)
ETA_BOUNDS = (np.log(50.0), np.log(5000.0))  # keeps the log-link IRLS away from overflow
LEAD_GRID = 6  # hours between the lagged columns of a lead-aligned regressor (image grid)
LEAD_FILL_HOURS = 24  # a missing lagged value is filled from the nearest lag within this
_LEAD = re.compile(r"^(?P<signal>.+)_lead_(?P<delay>\d+)h$")


def lead_feature(name):
    """(signal, delay h) for a lead-aligned regressor name like "ch_cm_lead_72h", else None."""
    m = _LEAD.match(name)
    return (m["signal"], int(m["delay"])) if m else None


def lead_lags(delay):
    """L_k for each lead k: delay - k rounded up to LEAD_GRID, at least 0."""
    return np.maximum(0, np.ceil((delay - LEADS) / LEAD_GRID).astype(int) * LEAD_GRID)


def lag_column(signal, lag):
    return f"{signal}_at_{int(lag)}h"


@dataclass(frozen=True)
class Config:
    features: tuple = ()  # selected regressor names (feature_engineering.windows)
    family: str = "gaussian"
    boxcox: float = 1.0  # Box-Cox lambda of the Gaussian target; 1 = untransformed
    relative: bool = False  # relative to persistence (target difference or log offset)
    weights: str = "run_timse"
    blend: float = 0.5  # weights="official": share of the percentile-TIMSE weights (rest uniform)
    ridge: float = 1.0
    lags_days: tuple = (27.0, 13.5)
    half_width: int = 3
    max_iter: int = 30
    tol: float = 1e-6
    despike: bool = False  # drop isolated one-hour glitches from targets and recurrence
    optional: tuple = ()  # features that may be missing: a submodel without them
    cap_quantile: float = 0.0  # clip each regressor to its training [q, 1 - q] quantiles; 0 = off

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if self.weights not in WEIGHTS:
            raise ValueError(f"weights must be one of {WEIGHTS}")
        if not 0.0 <= self.cap_quantile < 0.5:
            raise ValueError("cap_quantile must be in [0, 0.5)")
        if not set(self.optional) <= set(self.features):
            raise ValueError("optional features must be among the features")
        if self.despike:  # the cleaned recurrence window must end SPIKE_SIDE h before t0
            for lag in self.lag_hours:
                if lag - HORIZON - self.half_width < 1 + SPIKE_SIDE:
                    raise ValueError(f"lag {lag} h is too short for despiking")

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

    @property
    def input_features(self):
        """The regressor columns the model takes: features, lead-aligned ones expanded."""
        cols = []
        for f in self.features:
            lead = lead_feature(f)
            cols += ([f] if lead is None else
                     [lag_column(lead[0], lag) for lag in np.unique(lead_lags(lead[1]))])
        return tuple(dict.fromkeys(cols))

    def lead_columns(self, feature):
        """(72,) index into input_features of the column each lead uses for `feature`."""
        names = self.input_features
        lead = lead_feature(feature)
        if lead is None:
            return np.full(HORIZON, names.index(feature))
        return np.array([names.index(lag_column(lead[0], lag)) for lag in lead_lags(lead[1])])


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

    def _clean(self, values):
        return _despiked(values) if self.config.despike else values

    def speed_terms(self, values, positions):
        """(n, 72, n_speed) transformed speed terms, centred; missing -> 0 (the mean).

        The last observed speed is raw; the recurrence means use the despiked series
        (their windows end long before t0, so despiking them uses only past hours).
        """
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        centre = self._speed_transform(self.mean)
        out = np.zeros((len(positions), HORIZON, cfg.n_speed))
        last = last_observed(values, positions)
        s = self._speed_transform(last) - centre
        out[:, :, 0] = np.where(np.isfinite(s), s, 0.0)[:, None]
        clean = self._clean(values)
        for j, lag in enumerate(cfg.lag_hours):
            r = window_means(clean, positions[:, None] + LEADS - lag, cfg.half_width)
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
            for j, f in enumerate(self.config.features):
                x[:, :, 1 + self.config.n_speed + j] = z[:, self.config.lead_columns(f)]
        return x, last

    def fill_leads(self, regressors):
        """Missing lagged values of lead-aligned regressors from the nearest lag in the run."""
        cfg = self.config
        x = np.array(regressors, dtype=float, copy=True)
        names = cfg.input_features
        for f in cfg.features:
            lead = lead_feature(f)
            if lead is None:
                continue
            lags = np.unique(lead_lags(lead[1]))
            idx = [names.index(lag_column(lead[0], lag)) for lag in lags]
            block = x[:, idx]
            out = block.copy()
            for i, lag in enumerate(lags):
                # nearest lag first; on a tie the older image (larger lag)
                for j in sorted(range(len(lags)), key=lambda j: (abs(lags[j] - lag), -lags[j])):
                    if j == i or abs(lags[j] - lag) > LEAD_FILL_HOURS:
                        continue
                    gap = np.isnan(out[:, i])
                    if not gap.any():
                        break
                    out[gap, i] = block[gap, j]
            x[:, idx] = out
        return x

    def _select(self, x, names):
        """Columns `names` of an input_features matrix."""
        cols = self.config.input_features
        return x[:, [cols.index(c) for c in names]]

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
        if self.config.weights == "official":
            return official_weights(y, self.config.blend)
        w = run_timse_weights(y)
        scale = w[observed].mean() if observed.any() else 1.0
        return np.where(observed, np.maximum(w / max(scale, 1e-12), TAIL_FLOOR), 0.0)

    def fit(self, values, positions, regressors=None, end=None, run_weights=None):
        """Fit on the runs at `positions` of the hourly speed array `values`.

        Targets at or after `end` (a position) are ignored, so a fold's fit never sees
        its validation hours. Runs with a missing selected regressor are dropped.
        `run_weights` (one per run, optional) multiply the loss weights, e.g. to weight
        recent runs more (training.recency_weights).
        """
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        run_w = (np.ones(len(positions)) if run_weights is None
                 else np.asarray(run_weights, dtype=float))
        if run_w.shape != (len(positions),):
            raise ValueError(f"run_weights shape {run_w.shape}, expected ({len(positions)},)")
        n_in = len(cfg.input_features)
        x_raw = np.zeros((len(positions), 0)) if not cfg.features else np.asarray(
            regressors, dtype=float)
        if x_raw.shape != (len(positions), n_in):
            raise ValueError(f"regressors shape {x_raw.shape}, expected "
                             f"({len(positions)}, {n_in})")
        if cfg.features:
            x_raw = self.fill_leads(x_raw)
        hist_end = len(values) if end is None else end
        clean = self._clean(values[:hist_end])  # despiked up to `end` only
        self.mean = float(np.nanmean(clean))
        self.medians = (np.nanmedian(x_raw, axis=0) if cfg.features else np.zeros(0))
        self.fallback = None
        if cfg.optional:
            sub_cfg = replace(cfg, features=tuple(f for f in cfg.features
                                                  if f not in cfg.optional), optional=())
            self.fallback = LinearModel(sub_cfg).fit(
                values, positions, self._select(x_raw, sub_cfg.input_features), end, run_w)
        complete = np.isfinite(x_raw).all(axis=1)
        x_raw, positions, run_w = x_raw[complete], positions[complete], run_w[complete]
        self.x_low = self.x_high = None
        if cfg.cap_quantile > 0 and cfg.features:
            self.x_low = np.quantile(x_raw, cfg.cap_quantile, axis=0)
            self.x_high = np.quantile(x_raw, 1.0 - cfg.cap_quantile, axis=0)
            x_raw = self.cap(x_raw)
        self.x_mean = x_raw.mean(axis=0) if cfg.features else np.zeros(0)
        std = x_raw.std(axis=0) if cfg.features else np.zeros(0)
        self.x_std = np.where(std > 1e-12, std, 1.0)

        x, last = self.design(values, positions, x_raw)
        y = targets(clean, positions, end)
        q = self.loss_weights(y) * run_w[:, None]
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
        if cfg.despike:
            self.fit_info["despiked_hours"] = int(np.isnan(clean).sum()
                                                  - np.isnan(values[:hist_end]).sum())
        if self.fallback is not None:
            self.fit_info["fallback_runs"] = self.fallback.fit_info["runs"]
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
        x = np.asarray(regressors, dtype=float).reshape(-1, len(self.config.input_features))
        return np.where(np.isfinite(x), x, self.medians)

    def cap(self, x):
        """Regressors clipped to the training quantile bounds (Config.cap_quantile)."""
        if getattr(self, "x_low", None) is None:
            return x
        return np.clip(x, self.x_low, self.x_high)

    def forecast(self, values, positions, regressors=None):
        """Forecasts (len(positions) x 72).

        A run missing an optional regressor goes to the submodel without it; other
        missing regressors are imputed with the training medians.
        """
        positions = np.asarray(positions)
        cfg = self.config
        raw = (self.fill_leads(np.asarray(regressors, dtype=float).reshape(
            len(positions), len(cfg.input_features)))
               if cfg.features else np.zeros((len(positions), 0)))
        out = self._forecast(values, positions, raw)
        rows = self.uses_fallback(raw)
        if rows.any():
            sub = self._select(raw[rows], self.fallback.config.input_features)
            out[rows] = self.fallback.forecast(values, positions[rows], sub)
        return out

    def uses_fallback(self, regressors):
        """Which rows the submodel forecasts (an optional regressor is missing)."""
        cfg = self.config
        if getattr(self, "fallback", None) is None:
            return np.zeros(len(np.asarray(regressors)), dtype=bool)
        raw = self.fill_leads(np.asarray(regressors, dtype=float).reshape(
            -1, len(cfg.input_features)))
        keep = set(self.fallback.config.input_features)
        opt = [i for i, c in enumerate(cfg.input_features) if c not in keep]
        return np.isnan(raw[:, opt]).any(axis=1)

    def _forecast(self, values, positions, raw):
        x_raw = self.cap(self.impute(raw)) if self.config.features else raw
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
                         for n in self.config.input_features]])
        return self.forecast(values, np.array([len(values)]), row)[0]

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        cfg = asdict(self.config)
        cfg["features"] = list(cfg["features"])
        cfg["lags_days"] = list(cfg["lags_days"])
        cfg["optional"] = list(cfg["optional"])
        fallback = getattr(self, "fallback", None)
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
            "fallback": None if fallback is None else fallback.to_dict(),
            "x_low": None if getattr(self, "x_low", None) is None
            else np.round(self.x_low, 10).tolist(),
            "x_high": None if getattr(self, "x_high", None) is None
            else np.round(self.x_high, 10).tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("format") not in READABLE_FORMATS or d.get("model") != "LinearModel":
            raise ValueError(f"unsupported model asset {d.get('model')!r} v{d.get('format')!r}")
        cfg = dict(d["config"])
        cfg["features"] = tuple(cfg["features"])
        cfg["lags_days"] = tuple(cfg["lags_days"])
        cfg["optional"] = tuple(cfg.get("optional", ()))
        model = cls(Config(**cfg))
        model.mean = float(d["mean"])
        model.x_mean = np.asarray(d["x_mean"], dtype=float)
        model.x_std = np.asarray(d["x_std"], dtype=float)
        model.medians = np.asarray(d["medians"], dtype=float)
        model.coef = np.asarray(d["coef"], dtype=float)
        model.smear = None if d["smear"] is None else np.asarray(d["smear"], dtype=float)
        model.fit_info = d.get("fit_info", {})
        model.fallback = cls.from_dict(d["fallback"]) if d.get("fallback") else None
        model.x_low = None if d.get("x_low") is None else np.asarray(d["x_low"], dtype=float)
        model.x_high = None if d.get("x_high") is None else np.asarray(d["x_high"], dtype=float)
        if model.config.cap_quantile > 0 and model.config.features and model.x_low is None:
            raise ValueError("asset sets cap_quantile but has no cap bounds")
        if model.config.optional and model.fallback is None:
            raise ValueError("asset lists optional features but has no fallback submodel")
        if model.coef.shape != (HORIZON, model.config.n_columns):
            raise ValueError(f"coef shape {model.coef.shape}")
        return model

    def save(self, path=ASSET):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path=ASSET):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


_DESPIKED = {}


def _despiked(values):
    """despike(values), cached: selection cleans the same series thousands of times."""
    v = np.ascontiguousarray(values, dtype=float)
    key = (v.shape, hash(v.tobytes()))
    out = _DESPIKED.get(key)
    if out is None:
        if len(_DESPIKED) >= 8:
            _DESPIKED.clear()
        out = _DESPIKED[key] = despike(v)
        out.flags.writeable = False
    return out


def _get(features, name):
    try:
        v = features[name]
    except (KeyError, IndexError):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan
