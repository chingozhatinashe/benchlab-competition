"""Structural solar-wind model: a steady ambient wind plus stream arrivals, fitted in one step.

At t0, for leads k = 1..72 (x(u): the inputs for hour t0 + u, lead-aligned as in the
linear model):

    v(k) = a + e0 * exp(-k / tau_c) + sum_{u <= k} P_first(u) * A(u) * g(k - u)

    ambient a        250 + 250 * sigmoid(w_amb . [1, q20 - mean, rec27 - mean]); q20 is the
                     20th percentile of the speed over the last 27 days, rec27 the 27-day
                     recurrence mean over the 72 leads (both / 100 km/s)
    current stream   e0 = last speed - a, relaxing with tau_c = 1 + 24 softplus(w_dec . x(1))
                     h above the ambient (a single tau_neg below it)
    new arrival      hazard h(u) = sigmoid(w_haz . x(u)); only the first arrival in the 72 h
                     counts: P_first(u) = h(u) prod_{s < u} (1 - h(s))
    its speed        peak excess A(u) = 100 softplus(w_amp . x(u)) km/s
    its duration     profile g(d) = (1 - exp(-(d + 1) / r)) exp(-d / tau_s), peak 1, with a
                     fitted rise time r and decay time tau_s (h)
The output is the expected speed (it suits the RMSE part of the official metric), clipped
to VALID_RANGE. x(u) = [1, last, rec27(u), rec13.5(u) (centred, / 100 km/s), the
standardised regressors at lead u, a missing flag per optional regressor].

All parameters are fitted together by minimising the linear model's loss (official
weights, blend 0.25) over every training run and lead: training.structural does that with
PyTorch (dev only). This module is numpy only; `forward` takes an `ops` namespace so the
same equation runs in numpy here and in torch for training.
Inputs reuse models.linear_model (lead filling, imputation, regressor cap, standardisation,
recurrence terms, despiking). See docs/structural_model.md.
"""

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

from feature_engineering.recurrence import HORIZON, targets
from models.final_baseline import VALID_RANGE, official_weights
from models.linear_model import Config as LinearConfig
from models.linear_model import LinearModel

FORMAT_VERSION = 1
LEADS = np.arange(1, HORIZON + 1, dtype=float)
AMBIENT_RANGE = (250.0, 500.0)
TAU_SCALE = 24.0  # h; decay times are 1 + TAU_SCALE * softplus(.)
RISE_SCALE = 12.0
AMP_SCALE = 100.0  # km/s
SPEED_SCALE = 100.0  # km/s, for the speed inputs
LOSS_SCALE = 1e4  # (km/s)^2, keeps the loss near 1

NUMPY = SimpleNamespace(
    exp=np.exp, sigmoid=lambda z: 0.5 * (1.0 + np.tanh(0.5 * z)),
    softplus=lambda z: np.logaddexp(0.0, z), cumsum=lambda x, axis: np.cumsum(x, axis=axis),
    clip=np.clip, where=np.where, einsum=np.einsum, max=np.max,
)


@dataclass(frozen=True)
class Config:
    features: tuple = ()  # regressors, as models.linear_model (lead-aligned names allowed)
    optional: tuple = ()  # may be missing: zero (the mean) plus a missing flag
    exclude: tuple = ()  # regressors left out of x(u) (ablations)
    cap_quantile: float = 0.01
    blend: float = 0.25
    lags_days: tuple = (27.0, 13.5)
    half_width: int = 3
    despike: bool = True
    arrivals: bool = True  # False: ambient + decay only
    decay_inputs: bool = True  # False: one tau_c for every run
    ambient_hours: int = 27 * 24
    ambient_quantile: float = 0.2
    ridge: float = 1e-3  # on every weight except the intercepts

    def linear(self):
        return LinearConfig(features=tuple(self.features), optional=tuple(self.optional),
                            relative=True, weights="official", blend=self.blend,
                            lags_days=tuple(self.lags_days), half_width=self.half_width,
                            despike=self.despike, cap_quantile=self.cap_quantile)

    @property
    def used(self):
        return tuple(f for f in self.features if f not in self.exclude)

    @property
    def flags(self):
        return tuple(f for f in self.optional if f not in self.exclude)

    @property
    def n_inputs(self):
        return 4 + len(self.used) + len(self.flags)

    def shapes(self):
        p = self.n_inputs
        return {"amb": 3, "dec": p if self.decay_inputs else 1, "neg": 1,
                "haz": p, "amp": p, "rise": 1, "sdec": 1}


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


def initial_params(cfg, hazard=-5.0):
    """Physical starting values: a ~ 380 km/s following q20, tau_c ~ 24 h, an arrival
    hazard of sigmoid(hazard) per hour, amplitude ~ 150 km/s, rise ~ 12 h, decay ~ 48 h."""
    p = {k: np.zeros(n) for k, n in cfg.shapes().items()}
    p["amb"][:2] = [0.08, 1.6]
    p["dec"][0] = _inv_softplus(23.0 / TAU_SCALE)
    p["neg"][0] = _inv_softplus(23.0 / TAU_SCALE)
    p["haz"][0] = hazard
    p["amp"][0] = _inv_softplus(1.5)
    p["rise"][0] = _inv_softplus(11.0 / RISE_SCALE)
    p["sdec"][0] = _inv_softplus(47.0 / TAU_SCALE)
    return p


def penalty_mask(cfg):
    """1 for weights under the ridge, 0 for intercepts and scalar shape parameters."""
    out = {k: np.ones(n) for k, n in cfg.shapes().items()}
    for k in out:
        out[k][0] = 0.0
    return out


def forward(p, inp, cfg, ops=NUMPY, parts=False):
    """(n, 72) expected speeds. `p`: parameter arrays; `inp`: from StructuralModel.prepare
    (arrays of the same kind as ops expects)."""
    x, last, amb = inp["x"], inp["last"], inp["amb"]
    lo, hi = AMBIENT_RANGE
    a = lo + (hi - lo) * ops.sigmoid(amb @ p["amb"])
    e0 = last - a
    if cfg.decay_inputs:
        tau_pos = 1.0 + TAU_SCALE * ops.softplus(x[:, 0, :] @ p["dec"])
    else:
        tau_pos = 1.0 + TAU_SCALE * ops.softplus(p["dec"][0] + 0.0 * last)
    tau_neg = 1.0 + TAU_SCALE * ops.softplus(p["neg"][0] + 0.0 * last)
    tau = ops.where(e0 > 0, tau_pos, tau_neg)
    k = inp["leads"]
    current = a[:, None] + e0[:, None] * ops.exp(-k[None, :] / tau[:, None])
    out = current
    stream = 0.0 * current
    if cfg.arrivals:
        zh = x @ p["haz"]
        h = ops.sigmoid(zh)
        log_stay = -ops.softplus(zh)  # log(1 - h)
        before = ops.cumsum(log_stay, 1) - log_stay
        first = h * ops.exp(before)
        amp = AMP_SCALE * ops.softplus(x @ p["amp"])
        r = 1.0 + RISE_SCALE * ops.softplus(p["rise"][0])
        ts = 1.0 + TAU_SCALE * ops.softplus(p["sdec"][0])
        d, ahead = inp["lag"], inp["ahead"]
        g = (1.0 - ops.exp(-(d + 1.0) / r)) * ops.exp(-d * ahead / ts) * ahead
        g = g / ops.max(g)
        stream = ops.einsum("nu,ku->nk", first * amp, g)
        out = current + stream
    v = ops.clip(out, *VALID_RANGE)
    if parts:
        return v, {"ambient": a, "current": current, "stream": stream}
    return v


class StructuralModel:
    """fit statistics here; the parameters come from training.structural (torch)."""

    def __init__(self, config=None):
        self.config = config or Config()
        self.pre = LinearModel(self.config.linear())
        self.params = None

    # ------------------------------------------------------------------ inputs
    def ambient_quantile(self, values, positions):
        """The speed's ambient_quantile over the ambient_hours before each position."""
        cfg = self.config
        q = (pd.Series(np.asarray(values, dtype=float))
             .rolling(cfg.ambient_hours, min_periods=cfg.ambient_hours // 4)
             .quantile(cfg.ambient_quantile).to_numpy())
        prev = np.asarray(positions) - 1
        out = np.full(len(prev), np.nan)
        ok = (prev >= 0) & (prev < len(q))
        out[ok] = q[prev[ok]]
        return out

    def set_statistics(self, values, positions, regressors, end=None):
        """The input statistics from the fit runs (mean speed, regressor medians, caps,
        means and standard deviations), as LinearModel.fit computes them."""
        cfg, pre = self.config, self.pre
        values = np.asarray(values, dtype=float)
        hist_end = len(values) if end is None else end
        pre.mean = float(np.nanmean(pre._clean(values[:hist_end])))
        n_in = len(pre.config.input_features)
        raw = (pre.fill_leads(np.asarray(regressors, dtype=float).reshape(-1, n_in))
               if cfg.features else np.zeros((len(positions), 0)))
        pre.medians = np.nanmedian(raw, axis=0) if cfg.features else np.zeros(0)
        pre.x_low = pre.x_high = None
        if cfg.cap_quantile > 0 and cfg.features:
            pre.x_low = np.nanquantile(raw, cfg.cap_quantile, axis=0)
            pre.x_high = np.nanquantile(raw, 1.0 - cfg.cap_quantile, axis=0)
        capped = pre.cap(pre.impute(raw)) if cfg.features else raw
        pre.x_mean = capped.mean(axis=0) if cfg.features else np.zeros(0)
        std = capped.std(axis=0) if cfg.features else np.zeros(0)
        pre.x_std = np.where(std > 1e-12, std, 1.0)
        return self

    def prepare(self, values, positions, regressors):
        """Model inputs for runs at `positions` (missing regressors imputed, optional ones
        zeroed and flagged)."""
        cfg, pre = self.config, self.pre
        positions = np.asarray(positions)
        n = len(positions)
        lin = pre.config
        raw = (pre.fill_leads(np.asarray(regressors, dtype=float).reshape(
            n, len(lin.input_features))) if cfg.features else np.zeros((n, 0)))
        x_raw = pre.cap(pre.impute(raw)) if cfg.features else raw
        design, last = pre.design(values, positions, x_raw)
        cols = [np.ones((n, HORIZON)), design[:, :, 1] / SPEED_SCALE,
                design[:, :, 2] / SPEED_SCALE, design[:, :, 3] / SPEED_SCALE]
        missing = {f: np.isnan(raw[:, lin.lead_columns(f)]) for f in cfg.optional}
        for j, f in enumerate(cfg.features):
            if f in cfg.exclude:
                continue
            z = design[:, :, 1 + lin.n_speed + j]
            cols.append(np.where(missing[f], 0.0, z) if f in missing else z)
        for f in cfg.flags:
            cols.append(missing[f].astype(float))
        x = np.stack(cols, axis=-1)
        q20 = self.ambient_quantile(pre._clean(np.asarray(values, dtype=float)), positions)
        q20 = np.where(np.isfinite(q20), q20, pre.mean)
        amb = np.stack([np.ones(n), (q20 - pre.mean) / SPEED_SCALE,
                        design[:, :, 2].mean(axis=1) / SPEED_SCALE], axis=1)
        d = LEADS[:, None] - LEADS[None, :]
        return {"x": x, "amb": amb, "last": np.where(np.isfinite(last), last, pre.mean),
                "has_last": np.isfinite(last), "leads": LEADS,
                "lag": np.maximum(d, 0.0), "ahead": (d >= 0).astype(float)}

    def training_targets(self, values, positions, end=None, has_last=None):
        """(y, q): despiked targets before `end` and the official loss weights (zero for
        runs without a last speed, as the relative linear model)."""
        values = np.asarray(values, dtype=float)
        hist_end = len(values) if end is None else end
        y = targets(self.pre._clean(values[:hist_end]), positions, end)
        q = official_weights(y, self.config.blend)
        if has_last is not None:
            q = np.where(has_last[:, None], q, 0.0)
        return np.where(q > 0, y, 0.0), q

    # ------------------------------------------------------------------ forecasting
    def forecast(self, values, positions, regressors=None, parts=False):
        inp = self.prepare(values, positions, regressors)
        return forward(self.params, inp, self.config, parts=parts)

    # ------------------------------------------------------------------ assets
    def to_dict(self):
        pre = self.pre
        return {"format": FORMAT_VERSION, "config": {k: list(v) if isinstance(v, tuple) else v
                                                     for k, v in asdict(self.config).items()},
                "mean": pre.mean, "medians": pre.medians.tolist(),
                "x_low": None if pre.x_low is None else pre.x_low.tolist(),
                "x_high": None if pre.x_high is None else pre.x_high.tolist(),
                "x_mean": pre.x_mean.tolist(), "x_std": pre.x_std.tolist(),
                "params": {k: np.asarray(v).tolist() for k, v in self.params.items()}}

    @classmethod
    def from_dict(cls, d):
        cfg = Config(**{k: tuple(v) if isinstance(v, list) else v
                        for k, v in d["config"].items()})
        m = cls(cfg)
        pre = m.pre
        pre.mean = float(d["mean"])
        pre.medians = np.asarray(d["medians"], dtype=float)
        pre.x_low = None if d["x_low"] is None else np.asarray(d["x_low"], dtype=float)
        pre.x_high = None if d["x_high"] is None else np.asarray(d["x_high"], dtype=float)
        pre.x_mean = np.asarray(d["x_mean"], dtype=float)
        pre.x_std = np.asarray(d["x_std"], dtype=float)
        m.params = {k: np.asarray(v, dtype=float) for k, v in d["params"].items()}
        return m
