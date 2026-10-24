"""Gradient-boosted trees (XGBoost) pooled over all 72 leads, evaluated in numpy.

One model serves every lead: each training row is one (run, lead k) pair with
the features

    lead       k (hours ahead)
    last       last observed speed before t0
    rec_<d>d   mean speed within +-half_width h of t0 + k - d days (d = 27, 13.5)
    <name>     the selected regressors (feature_engineering.windows), the same
               value for all 72 rows of a run

and the target speed(t0 + k) - anchor(k): the trees learn corrections to an
anchor forecast (Config.base):

    persistence  anchor = last observed speed
    linear       anchor = the linear model's forecast (models.linear_model, stored
                 inside this asset); the rows also get its offset from persistence
                 (`anchor` = linear - last). This hybrid keeps the linear model's
                 ability to extrapolate, and the trees add non-linear corrections.

Rows are weighted by each run's TIMSE weights (as the linear model),
which makes squared error the competition loss. Missing values (NaN) follow the
default branch each split learned. A regressor that was never missing in training
has no learned default branch (xgboost picks one arbitrarily), so a missing value
of such a regressor is filled with its training median instead (`impute_values`).

Training (dev only) uses xgboost; it is imported inside `fit`. The image never
imports it: the fitted trees are stored as flat arrays in assets/xgboost_model.json
and evaluated by `TreeEnsemble` in numpy, matching xgboost's own predictions
(float32 comparisons, x < threshold goes left, NaN takes the default branch).

Despiking (feature_engineering.recurrence.despike) cleans the fitting targets and
the recurrence windows, as in the linear model; the last observed speed stays raw.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from feature_engineering.recurrence import (
    HORIZON,
    LEADS,
    SPIKE_SIDE,
    last_observed,
    targets,
    window_means,
)
from models.final_baseline import TAIL_FLOOR, VALID_RANGE, run_timse_weights
from models.linear_model import LinearModel, _despiked

ASSET = Path(__file__).resolve().parents[1] / "assets" / "xgboost_model.json"
FORMAT_VERSION = 1
BASES = ("persistence", "linear")


@dataclass(frozen=True)
class Config:
    features: tuple = ()  # selected regressor names (feature_engineering.windows)
    max_depth: int = 4
    min_child_weight: float = 5.0  # in units of mean row weight
    eta: float = 0.05
    n_trees: int = 400
    subsample: float = 0.8
    colsample: float = 0.8
    reg_lambda: float = 1.0
    max_bin: int = 256
    lags_days: tuple = (27.0, 13.5)
    half_width: int = 3
    despike: bool = True
    leads_per_run: int = 12  # training rows per run: a random subset of the 72 leads
    seed: int = 0
    base: str = "persistence"  # what the trees correct: persistence or the linear model

    def __post_init__(self):
        if self.base not in BASES:
            raise ValueError(f"base must be one of {BASES}")
        for lag in self.lag_hours:
            if lag - HORIZON - self.half_width < 1 + (SPIKE_SIDE if self.despike else 0):
                raise ValueError(f"lag {lag} h is too short: its window would reach t0")
        if not 1 <= self.leads_per_run <= HORIZON:
            raise ValueError("leads_per_run must be in 1..72")

    @property
    def lag_hours(self):
        return tuple(int(round(d * 24)) for d in self.lags_days)

    @property
    def history_hours(self):
        return max(self.lag_hours) + self.half_width + 48

    @property
    def columns(self):
        anchor = ("anchor",) if self.base == "linear" else ()
        return ("lead", "last", *(f"rec_{d:g}d" for d in self.lags_days), *anchor,
                *self.features)


# ---------------------------------------------------------------------- trees in numpy
class TreeEnsemble:
    """XGBoost trees as flat node arrays; predict() reproduces Booster.predict."""

    def __init__(self, left, right, feature, threshold, default_left, value, roots, base_score,
                 depth):
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.feature = np.asarray(feature, dtype=np.int64)
        self.threshold = np.asarray(threshold, dtype=np.float32)
        self.default_left = np.asarray(default_left, dtype=bool)
        self.value = np.asarray(value, dtype=np.float32)
        self.roots = np.asarray(roots, dtype=np.int64)
        self.base_score = float(base_score)
        self.depth = int(depth)

    @classmethod
    def from_booster(cls, booster, n_trees=None):
        """From an xgboost Booster (its JSON model), keeping the first n_trees trees."""
        model = json.loads(booster.save_raw("json"))["learner"]
        base = model["learner_model_param"]["base_score"].strip("[]")
        trees = model["gradient_booster"]["model"]["trees"]
        if n_trees is not None:
            trees = trees[:n_trees]
        parts = {k: [] for k in ("left", "right", "feature", "threshold", "default", "value")}
        roots, offset, depth = [], 0, 0
        for t in trees:
            left = np.asarray(t["left_children"], dtype=np.int64)
            right = np.asarray(t["right_children"], dtype=np.int64)
            leaf = left == -1
            cond = np.asarray(t["split_conditions"], dtype=np.float32)
            parts["left"].append(np.where(leaf, -1, left + offset))
            parts["right"].append(np.where(leaf, -1, right + offset))
            parts["feature"].append(np.where(leaf, 0, t["split_indices"]))
            parts["threshold"].append(np.where(leaf, 0, cond))
            parts["default"].append(np.asarray(t["default_left"], dtype=bool))
            parts["value"].append(np.where(leaf, cond, 0))  # a leaf's value is its condition
            roots.append(offset)
            offset += len(left)
            depth = max(depth, _tree_depth(left, right))
        cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
        return cls(cat["left"], cat["right"], cat["feature"], cat["threshold"], cat["default"],
                   cat["value"], roots, float(base), depth)

    @property
    def n_trees(self):
        return len(self.roots)

    def apply(self, x, chunk=2048):
        """The leaf node each row reaches in each tree: (rows, trees) node indices."""
        x = np.asarray(x, dtype=np.float32)
        out = np.empty((len(x), self.n_trees), dtype=np.int64)
        for start in range(0, len(x), chunk):
            xs = x[start : start + chunk]
            rows = np.arange(len(xs))[:, None]
            node = np.broadcast_to(self.roots, (len(xs), self.n_trees)).copy()
            for _ in range(self.depth):
                inner = self.left[node] >= 0
                if not inner.any():
                    break
                v = xs[rows, self.feature[node]]
                go_left = np.where(np.isnan(v), self.default_left[node], v < self.threshold[node])
                node = np.where(inner, np.where(go_left, self.left[node], self.right[node]), node)
            out[start : start + chunk] = node
        return out

    def predict(self, x, chunk=2048):
        """Sum of leaf values over trees plus the base score, for each row of x."""
        out = np.full(len(x), self.base_score, dtype=np.float64)
        if not self.n_trees:
            return out
        return out + self.value[self.apply(x, chunk)].sum(axis=1, dtype=np.float64)

    def to_dict(self):
        return {"base_score": self.base_score, "depth": self.depth,
                "roots": self.roots.tolist(), "left": self.left.tolist(),
                "right": self.right.tolist(), "feature": self.feature.tolist(),
                # float32 values are exact in float64, so the round trip is lossless
                "threshold": self.threshold.astype(float).tolist(),
                "default_left": self.default_left.astype(int).tolist(),
                "value": self.value.astype(float).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["left"], d["right"], d["feature"], d["threshold"], d["default_left"],
                   d["value"], d["roots"], d["base_score"], d["depth"])


def _tree_depth(left, right):
    depth, level = 0, [0]
    while level:
        nxt = [c for n in level if left[n] >= 0 for c in (left[n], right[n])]
        depth += bool(nxt)
        level = nxt
    return depth


# ---------------------------------------------------------------------- the model
class XGBoostModel:
    """fit(values, positions, regressors) / predict(history, features)."""

    def __init__(self, config=None, base_model=None):
        self.config = config or Config()
        if self.config.base == "linear" and base_model is None:
            raise ValueError("base='linear' needs a fitted LinearModel")
        self.base_model = base_model if self.config.base == "linear" else None
        self.trees = None
        self.booster = None  # only after fit() in this process (dev)
        self.impute_values = {}  # regressors never missing in training -> median

    @property
    def input_features(self):
        """Regressor columns the model takes: the trees' and the linear model's."""
        lin = self.base_model.config.features if self.base_model is not None else ()
        return tuple(dict.fromkeys((*self.config.features, *lin)))

    def _columns(self, regressors, n, names):
        if not names:
            return None
        reg = np.asarray(regressors, dtype=float).reshape(n, len(self.input_features))
        return reg[:, [self.input_features.index(f) for f in names]]

    def anchor(self, values, positions, regressors=None):
        """(n, 72) forecast the trees correct (NaN where persistence has no speed)."""
        positions = np.asarray(positions)
        if self.base_model is None:
            last = last_observed(np.asarray(values, dtype=float), positions)
            return np.repeat(last[:, None], HORIZON, axis=1)
        x = self._columns(regressors, len(positions), self.base_model.config.features)
        out = np.empty((len(positions), HORIZON))
        chunk = 4000  # bounds the linear model's (runs x 72 x columns) design in memory
        for s in range(0, len(positions), chunk):
            out[s : s + chunk] = self.base_model.forecast(
                values, positions[s : s + chunk], None if x is None else x[s : s + chunk])
        return out

    def _clean(self, values):
        return _despiked(values) if self.config.despike else values

    def rows(self, values, positions, regressors=None, leads=None, anchor=None):
        """Design rows: (n * m, columns) float32, one per (run, lead), and `last` per run.

        `regressors` has the input_features columns; `leads` is (m,) for the same
        leads in every run (default all 72), or (n, m); `anchor` is the anchor at
        those leads (needed for base='linear').
        """
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        n = len(positions)
        leads = LEADS if leads is None else np.asarray(leads)
        lead = np.broadcast_to(leads, (n, leads.shape[-1]))
        m = lead.shape[1]
        clean = self._clean(values)
        last = last_observed(values, positions)
        x = np.empty((n, m, len(cfg.columns)), dtype=np.float32)
        x[:, :, 0] = lead
        x[:, :, 1] = last[:, None]
        for j, lag in enumerate(cfg.lag_hours):
            x[:, :, 2 + j] = window_means(clean, positions[:, None] + lead - lag, cfg.half_width)
        col = 2 + len(cfg.lag_hours)
        if cfg.base == "linear":
            x[:, :, col] = np.asarray(anchor) - last[:, None]
            col += 1
        if cfg.features:
            reg = self._columns(regressors, n, cfg.features)
            for j, f in enumerate(cfg.features):
                if f in self.impute_values:
                    reg[:, j] = np.where(np.isnan(reg[:, j]), self.impute_values[f], reg[:, j])
            x[:, :, col:] = reg[:, None, :]
        return x.reshape(n * m, -1), last

    def fit_imputation(self, regressors, n):
        """Medians for the regressors that are never missing in these training runs."""
        reg = self._columns(regressors, n, self.config.features)
        self.impute_values = {} if reg is None else {
            f: float(np.median(reg[:, j])) for j, f in enumerate(self.config.features)
            if np.isfinite(reg[:, j]).all()}

    # ------------------------------------------------------------------ fitting (dev)
    def training_set(self, values, positions, regressors=None, end=None, rng=None,
                     run_weights=None):
        """(rows, target, weight) for fitting: a random subset of leads per run.

        `run_weights` (one per run, optional) multiply the rows' weights.
        """
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        hist_end = len(values) if end is None else end
        clean = self._clean(values[:hist_end])
        y = targets(clean, positions, end)
        q = run_timse_weights(y)
        observed = np.isfinite(y)
        scale = q[observed].mean() if observed.any() else 1.0
        q = np.where(observed, np.maximum(q / max(scale, 1e-12), TAIL_FLOOR), 0.0)
        if run_weights is not None:
            q = q * np.asarray(run_weights, dtype=float)[:, None]
        rng = np.random.default_rng(cfg.seed) if rng is None else rng
        pick = np.argsort(rng.random((len(positions), HORIZON)), axis=1)[:, : cfg.leads_per_run]
        pick.sort(axis=1)
        anchor = np.take_along_axis(self.anchor(values[:hist_end], positions, regressors),
                                    pick, axis=1)
        x, _ = self.rows(values[:hist_end], positions, regressors, LEADS[pick], anchor)
        y = np.take_along_axis(y, pick, axis=1) - anchor
        w = np.take_along_axis(q, pick, axis=1)
        keep = (np.isfinite(y) & (w > 0)).ravel()
        return x[keep], y.ravel()[keep], w.ravel()[keep]

    def params(self, nthread=0):
        """xgboost parameters; nthread 0 = all cores, or $XGB_NTHREAD if set (dev)."""
        import os

        nthread = nthread or int(os.environ.get("XGB_NTHREAD", "0"))
        cfg = self.config
        return {"objective": "reg:squarederror", "tree_method": "hist",
                "max_depth": cfg.max_depth, "min_child_weight": cfg.min_child_weight,
                "eta": cfg.eta, "subsample": cfg.subsample, "colsample_bytree": cfg.colsample,
                "lambda": cfg.reg_lambda, "max_bin": cfg.max_bin, "seed": cfg.seed,
                "nthread": nthread}

    def fit(self, values, positions, regressors=None, end=None, n_trees=None, nthread=0,
            run_weights=None):
        """Fit on the runs at `positions`; targets at or after `end` are ignored.

        `run_weights` (one per run, optional) multiply the rows' weights.
        """
        import xgboost as xgb

        hist_end = len(values) if end is None else end
        self.mean = float(np.nanmean(self._clean(np.asarray(values, dtype=float)[:hist_end])))
        self.fit_imputation(regressors, len(positions))
        x, y, w = self.training_set(values, positions, regressors, end,
                                    run_weights=run_weights)
        data = xgb.QuantileDMatrix(x, label=y, weight=w, max_bin=self.config.max_bin)
        del x
        self.booster = xgb.train(self.params(nthread), data,
                                 num_boost_round=n_trees or self.config.n_trees)
        self.trees = TreeEnsemble.from_booster(self.booster, self.config.n_trees)
        self.fit_info = {"runs": int(len(positions)), "rows": int(len(y)),
                         "trees": self.trees.n_trees}
        return self

    # ------------------------------------------------------------------ forecasting
    def forecast(self, values, positions, regressors=None, n_trees=None):
        """Forecasts (len(positions) x 72).

        With a booster from fit() in this process, xgboost predicts (faster on large
        batches; n_trees scores the first n trees); otherwise the numpy trees do.
        """
        positions = np.asarray(positions)
        anchor = self.anchor(values, positions, regressors)
        x, _ = self.rows(values, positions, regressors, anchor=anchor)
        if self.booster is not None:
            import xgboost as xgb

            n = n_trees or self.config.n_trees
            delta = self.booster.predict(xgb.DMatrix(x), iteration_range=(0, n))
        else:
            if n_trees not in (None, self.trees.n_trees):
                raise ValueError("n_trees needs the booster from fit()")
            delta = self.trees.predict(x)
        out = anchor + delta.reshape(len(positions), HORIZON)
        # no observed speed at all (persistence anchor): forecast the training mean
        out = np.where(np.isfinite(anchor), out, self.mean)
        return np.clip(out, *VALID_RANGE)

    def predict(self, history, features=None):
        """72 forecasts from the hourly history ending at t0 - 1h and one feature row."""
        values = np.asarray(history, dtype=float)
        names = self.input_features
        row = np.array([[np.nan if features is None else _get(features, n) for n in names]])
        return self.forecast(values, np.array([len(values)]), row if names else None)[0]

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        cfg = asdict(self.config)
        cfg["features"] = list(cfg["features"])
        cfg["lags_days"] = list(cfg["lags_days"])
        return {"model": "XGBoostModel", "format": FORMAT_VERSION, "config": cfg,
                "mean": self.mean, "trees": self.trees.to_dict(),
                "base_model": None if self.base_model is None else self.base_model.to_dict(),
                "impute_values": self.impute_values,
                "fit_info": getattr(self, "fit_info", {})}

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT_VERSION or d.get("model") != "XGBoostModel":
            raise ValueError(f"unsupported model asset {d.get('model')!r} v{d.get('format')!r}")
        cfg = dict(d["config"])
        cfg["features"] = tuple(cfg["features"])
        cfg["lags_days"] = tuple(cfg["lags_days"])
        base = d.get("base_model")
        model = cls(Config(**cfg), LinearModel.from_dict(base) if base else None)
        model.mean = float(d["mean"])
        model.impute_values = {k: float(v) for k, v in d.get("impute_values", {}).items()}
        model.trees = TreeEnsemble.from_dict(d["trees"])
        model.fit_info = d.get("fit_info", {})
        if model.trees.n_trees == 0:
            raise ValueError("asset has no trees")
        return model

    def save(self, path=ASSET):
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n",
                              encoding="utf-8")

    @classmethod
    def load(cls, path=ASSET):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _get(features, name):
    try:
        return float(features[name])
    except (KeyError, IndexError, TypeError, ValueError):
        return np.nan
