"""The reference baseline: blended 27-day recurrence with sub-rotation lags.

Forecast for lead k = 1..72 h at t0:

    R_k = a_k + b_k * (last - mu) + sum_j c_kj * (r_j(k) - mu)     recurrence regression
    F_k = w_k * last + (1 - w_k) * R_k                             persistence blend

    last     the last observed hourly speed before t0
    r_j(k)   mean speed within +-half_width h of t0 + k - lag_j. The lags are one solar
             rotation (27 d) plus optional sub-rotation lags (13.5 d, 9 d) for several
             streams per rotation
    mu       the training-years mean; a missing input is replaced by mu

The regression coefficients are a blend of two fits:

    coef = rolling_weight * rolling + (1 - rolling_weight) * frozen
    frozen   fitted once on the training years and baked into assets/
    rolling  refitted at every t0 on the last `window_days` of history before t0
             (per lead; falls back to frozen if the window has too few targets)

Both fits can use a tail-weighted loss. Each sample's squared error is weighted by
g(v)^tail_power, where v is the observed target and g(v) = integral of du / N(>= u)
over the training speeds. With tail_power = 1 the weighted squared error is, up to a
constant, the TIMSE of the training set; tail_power = 0 is ordinary least squares.

The persistence weights w_k (clipped to [0, 1]) are fitted out of sample: the
regression is fitted without the last `stack_days` of the fitting data, run
walk-forward over those days, and w_k is the loss-weighted least-squares blend of
persistence and that forecast. The regression is then refitted on all the data.

All hyper-parameters live in Config. Choose them on the training years only
(training.final_baseline tune), never by looking at eval scores.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import HORIZON, design, last_observed, targets

ASSET = Path(__file__).resolve().parents[1] / "assets" / "final_baseline.json"
FORMAT_VERSION = 1
VALID_RANGE = (200.0, 1500.0)  # sanity clip on the output, km/s
TAIL_FLOOR = 1e-3  # smallest tail weight, so no observed hour is ignored entirely
CYCLE_HOURS = 6  # platform runs at 00/06/12/18Z
LOSSES = ("run_timse", "timse")


@dataclass(frozen=True)
class Config:
    lags_days: tuple = (27.0, 13.5, 9.0)  # recurrence lags; 27 d is one rotation
    half_width: int = 3  # hours either side of each lagged time
    window_days: float = 30.0  # rolling refit window
    rolling_weight: float = 0.5  # 0 = frozen only, 1 = rolling only
    tail_power: float = 0.0  # 0 = least squares, 1 = TIMSE-weighted
    loss: str = "run_timse"  # tail weights from each run's own thresholds, or "timse" (pooled)
    blend_persistence: bool = True  # fit the persistence blend weights w_k
    stack_days: float = 365.0  # out-of-sample period used to fit w_k
    min_samples: int = 48  # fewest observed targets per lead for a rolling fit
    ridge: float = 1e-6  # relative ridge on the slopes, for numerical stability only

    def __post_init__(self):
        object.__setattr__(self, "lags_days", tuple(float(d) for d in self.lags_days))
        if not 0.0 <= self.rolling_weight <= 1.0:
            raise ValueError("rolling_weight must be in [0, 1]")
        if self.tail_power < 0:
            raise ValueError("tail_power must be >= 0")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")

    @property
    def lag_hours(self):
        return tuple(int(round(d * 24)) for d in self.lags_days)

    @property
    def window_hours(self):
        return int(round(self.window_days * 24))

    @property
    def history_hours(self):
        """Hours of history predict() uses: the rolling window plus the longest lag."""
        return self.window_hours + max(self.lag_hours, default=0) + self.half_width + 48

    @property
    def n_features(self):
        return 2 + len(self.lag_hours)


class TailWeights:
    """g(v) = integral of du / N(>= u) over a reference set of speeds, scaled to mean 1.

    For a set of observations, sum_i g(v_i) * e_i^2 approximates TIMSE: each hour's
    squared error counts once for every threshold at or below its speed, and each
    threshold is weighted by its gap over the number of hours above it.
    """

    def __init__(self, knots, g):
        self.knots = np.asarray(knots, dtype=float)
        self.g = np.asarray(g, dtype=float)

    @classmethod
    def fit(cls, speeds, step=1.0):
        v = np.sort(np.asarray(speeds, dtype=float))
        v = v[np.isfinite(v)]
        if v.size < 2:
            return cls([0.0, 1.0], [1.0, 1.0])
        knots = np.arange(np.floor(v[0]), np.ceil(v[-1]) + step, step)
        n_above = v.size - np.searchsorted(v, knots, side="left")
        g = np.concatenate([[0.0], np.cumsum(step / np.maximum(n_above[:-1], 1))])
        g /= np.mean(np.interp(v, knots, g))
        return cls(knots, g)

    def __call__(self, speeds, power):
        """Sample weights g(v)^power; 0 where v is missing."""
        speeds = np.asarray(speeds, dtype=float)
        observed = np.isfinite(speeds)
        if power == 0:
            return observed.astype(float)
        g = np.interp(np.where(observed, speeds, self.knots[0]), self.knots, self.g)
        return np.where(observed, np.maximum(g, TAIL_FLOOR) ** power, 0.0)


LEAD_INTERVALS = 100  # percentile intervals of the official TIMSE (organisers' spec, 6.1)


def lead_timse_weights(y):
    """Per-hour weights that turn squared errors into each lead's official percentile TIMSE.

    The organisers' metric (training.competition_score) scores each lead over all runs:
    TIMSE = sum_k (q_{k+1} - q_k) * MSE{v >= q_k} / (v_max - v_min), with q_k the 101
    linear-interpolation quantiles of that lead's observations. That equals
    sum_i w_i (v_i - f_i)^2 with w_i = sum over q_k <= v_i of (q_{k+1} - q_k) / (N_k * span),
    so these weights depend only on the observations, like run_timse_weights, but pool the
    runs at each lead instead of scoring each run on its own. `y` is (runs x leads);
    missing hours get 0, and a lead with fewer than two distinct speeds gets 0.
    """
    y = np.asarray(y, dtype=float)
    w = np.zeros_like(y)
    for j in range(y.shape[1]):
        ok = np.flatnonzero(np.isfinite(y[:, j]))
        if len(ok) < 2:
            continue
        order = ok[np.argsort(y[ok, j], kind="stable")]
        v = y[order, j]
        span = v[-1] - v[0]
        if span <= 0:
            continue
        k = np.arange(LEAD_INTERVALS + 1)
        lo, rem = np.divmod(k * (len(v) - 1), LEAD_INTERVALS)  # exact positions
        q = v[lo] + (rem / LEAD_INTERVALS) * (v[np.minimum(lo + 1, len(v) - 1)] - v[lo])
        first = np.searchsorted(v, q[:-1], side="left")
        step = np.zeros(len(v) + 1)
        np.add.at(step, first, np.diff(q) / ((len(v) - first) * span))
        w[order, j] = np.cumsum(step)[:-1]
    return w


def official_weights(y, blend):
    """Loss weights for the official per-lead score: (1 - blend) + blend * t on observed
    hours, t = lead_timse_weights scaled to mean 1 per lead. The official S_tau is
    sqrt(RMSE_tau * sqrt(TIMSE_tau)), so its loss mixes plain squared error (blend 0) and
    the percentile TIMSE (blend 1); the mix is a setting chosen on the folds."""
    y = np.asarray(y, dtype=float)
    observed = np.isfinite(y)
    t = lead_timse_weights(y)
    scale = np.where(observed, t, 0.0).sum(axis=0) / np.maximum(observed.sum(axis=0), 1)
    t = t / np.where(scale > 0, scale, 1.0)
    return np.where(observed, (1.0 - blend) + blend * t, 0.0)


def run_timse_weights(y):
    """Per-hour weights that turn squared errors into each run's own TIMSE.

    TIMSE weights depend only on the observations, so for a run with observed
    speeds v (NaN = missing) and any forecast, sum_i q_i * (v_i - f_i)^2 equals
    training.metrics.timse(v, f). Hour i gets sum over thresholds v*_l <= v_i
    (l < m - 1) of (v*_{l+1} - v*_l) / N_l, with N_l the hours at or above v*_l.
    `y` is (runs x hours); runs with fewer than 2 observations get zero weights.
    """
    y = np.asarray(y, dtype=float)
    n, h = y.shape
    observed = np.isfinite(y)
    m = observed.sum(axis=1)
    order = np.argsort(np.where(observed, y, np.inf), axis=1, kind="stable")
    s = np.take_along_axis(np.where(observed, y, np.inf), order, axis=1)
    idx = np.arange(h)
    inside = idx[:-1] < (m[:, None] - 1)  # threshold gaps l = 0 .. m-2
    with np.errstate(invalid="ignore"):
        gap = np.where(inside, np.diff(s, axis=1), 0.0)
    # N_l counts every hour at or above v*_l, so a tied threshold counts from its first copy
    is_first = np.ones((n, h), dtype=bool)
    is_first[:, 1:] = s[:, 1:] != s[:, :-1]
    first = np.maximum.accumulate(np.where(is_first, idx, 0), axis=1)
    n_above = np.maximum(m[:, None] - first[:, :-1], 1)
    term = np.where(inside, gap / n_above, 0.0)
    cum = np.concatenate([np.cumsum(term, axis=1), np.zeros((n, 1))], axis=1)
    # a tied speed belongs to every threshold up to the last copy of its value
    is_last = np.ones((n, h), dtype=bool)
    is_last[:, :-1] = s[:, 1:] != s[:, :-1]
    last = np.minimum.accumulate(np.where(is_last, idx, h - 1)[:, ::-1], axis=1)[:, ::-1]
    last = np.minimum(last, np.maximum(m[:, None] - 2, 0))
    sorted_w = np.take_along_axis(cum, last, axis=1)
    sorted_w = np.where((idx < m[:, None]) & (m[:, None] >= 2), sorted_w, 0.0)
    w = np.zeros((n, h))
    np.put_along_axis(w, order, sorted_w, axis=1)
    return w


def solve(a, b, ridge):
    """Batched solve of (a + ridge) x = b, with the ridge on the slope terms only."""
    n_feat = a.shape[-1]
    reg = np.zeros_like(a)
    diag = np.einsum("...ii->...i", a)
    idx = np.arange(1, n_feat)
    reg[..., idx, idx] = ridge * diag[..., idx]
    reg[..., np.arange(n_feat), np.arange(n_feat)] += 1e-9  # keeps empty systems solvable
    return np.linalg.solve(a + reg, b[..., None])[..., 0]


def _cumsum0(x):
    return np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])


def cycle_positions(index, start, end):
    """Positions in `index` of the 6-hourly t0s in [start, end) whose 72 h target fits."""
    index = pd.DatetimeIndex(index)
    pos = np.arange(len(index))
    keep = (index >= start) & (index < end) & (index.hour % CYCLE_HOURS == 0)
    keep &= pos + HORIZON < len(index)
    return pos[keep]


class FinalBaseline:
    """fit(train) / predict(history, t0), the training.evaluate interface."""

    def __init__(self, config=None):
        self.config = config or Config()

    # ------------------------------------------------------------------ fitting
    def fit(self, train):
        """Fit the frozen coefficients and the persistence weights on `train` only."""
        cfg = self.config
        values = np.asarray(train, dtype=float)
        self.prepare(values)
        self.persistence_weight = np.zeros(HORIZON)

        stack = int(round(cfg.stack_days * 24))
        if cfg.blend_persistence and 0 < stack <= len(values) // 2:
            split = len(values) - stack
            self.coef = self.fit_frozen(values[:split])
            ps = cycle_positions(train.index, train.index[split], train.index[-1])
            self.persistence_weight = self.fit_persistence_weight(values, ps)

        self.coef = self.fit_frozen(values)
        self.fit_info = {
            "start": f"{train.index[0]:%Y-%m-%dT%H:%MZ}",
            "end": f"{train.index[-1] + pd.Timedelta(hours=1):%Y-%m-%dT%H:%MZ}",
            "hours": int(len(values)),
            "observed_hours": int(np.isfinite(values).sum()),
        }
        return self

    def prepare(self, values):
        """Statistics of the fitting data: the fill mean and the loss-weight scales."""
        values = np.asarray(values, dtype=float)
        self.mean = float(np.nanmean(values))
        self.tail = TailWeights.fit(values)
        y = targets(values, np.arange(len(values)))
        observed = np.isfinite(y)
        self.run_scale = float(run_timse_weights(y)[observed].mean()) if observed.any() else 1.0

    def fit_frozen(self, values):
        """Per-lead (tail-)weighted least squares on every hour of `values` as a t0."""
        cfg = self.config
        positions = np.arange(1, len(values))
        x = design(values, positions, self.mean, cfg.lag_hours, cfg.half_width)
        y = targets(values, positions)
        q = self.weights(y)
        qy = np.where(q > 0, q * y, 0.0)
        a = np.zeros((HORIZON, cfg.n_features, cfg.n_features))
        b = np.zeros((HORIZON, cfg.n_features))
        for i in range(0, len(positions), 4096):  # chunks keep the temporaries small
            xc, qc = x[i : i + 4096], q[i : i + 4096]
            a += np.einsum("nkf,nk,nkg->kfg", xc, qc, xc)
            b += np.einsum("nkf,nk->kf", xc, qy[i : i + 4096])
        return solve(a, b, cfg.ridge)

    def fit_persistence_weight(self, values, positions):
        """w_k from walk-forward forecasts over `positions`, under the training loss."""
        regression = self.forecast(values, positions, blend=False)
        last = last_observed(values, positions)
        y = targets(values, positions)
        return self.blend_weight(regression, last, y)

    def weights(self, y):
        """Sample weights for whole runs of targets y (runs x 72) under the configured loss.

        "timse" weights each hour by g(v) over the training speeds (pooled thresholds);
        "run_timse" by its share of its own run's TIMSE, computed from the run's observed
        hours. Both are scaled by a constant fixed at fit time (mean 1 on the training
        data) and raised to tail_power, so tail_power = 0 is ordinary least squares.
        """
        cfg = self.config
        if cfg.tail_power == 0 or cfg.loss == "timse":
            return self.tail(y, cfg.tail_power)
        w = run_timse_weights(y) / max(self.run_scale, 1e-12)
        return np.where(np.isfinite(y), np.maximum(w, TAIL_FLOOR) ** cfg.tail_power, 0.0)

    def blend_weight(self, regression, last, y):
        q = self.weights(y)
        last = np.where(np.isfinite(last), last, np.nan)[:, None]
        ok = (q > 0) & np.isfinite(last)
        d = np.where(ok, last - regression, 0.0)
        e = np.where(ok, y - regression, 0.0)
        num = (q * d * e).sum(axis=0)
        den = (q * d * d).sum(axis=0)
        return np.clip(np.where(den > 0, num / np.maximum(den, 1e-12), 0.0), 0.0, 1.0)

    # --------------------------------------------------------------- forecasting
    def rolling(self, values, positions):
        """Design rows at each t0 and the rolling coefficients refitted there.

        For the t0 at position P, lead k learns from sample positions p in
        [P - window, P - k): the targets values[p + k] all lie before t0. Rows whose
        whole 72 h run ends before t0 (p < P - 72) are summed with cumulative sums,
        so many t0s cost one pass. The 72 rows just before t0 have runs cut short by
        t0; recent_rows adds them with loss weights from their observed part only.
        Returns (x_t0, coef, ok): ok marks leads with at least min_samples targets.
        """
        cfg = self.config
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        window = cfg.window_hours
        base = max(int(positions.min()) - window, 0)
        span = np.arange(base, int(positions.max()) + 1)
        x = design(values, span, self.mean, cfg.lag_hours, cfg.half_width)
        x_t0 = x[positions - base]
        coef = np.zeros((len(positions), HORIZON, cfg.n_features))
        ok = np.zeros((len(positions), HORIZON), dtype=bool)
        if window <= 0 or cfg.rolling_weight == 0:
            return x_t0, coef, ok

        y = targets(values, span)
        q = self.weights(y)  # used only for rows whose run ends before t0
        qy = np.where(q > 0, q * y, 0.0)
        local = positions - base
        lo = np.clip(local - window, 0, len(span))
        hi = np.clip(local - HORIZON, lo, len(span))  # rows p < P - 72
        a_recent, b_recent, n_recent = self.recent_rows(x, y, local, lo)
        for k in range(HORIZON):
            xk, qk = x[:, k], q[:, k]
            a = _cumsum0(np.einsum("nf,n,ng->nfg", xk, qk, xk))
            b = _cumsum0(xk * qy[:, k, None])
            count = _cumsum0((qk > 0).astype(float))
            coef[:, k] = solve(
                a[hi] - a[lo] + a_recent[:, k], b[hi] - b[lo] + b_recent[:, k], cfg.ridge
            )
            ok[:, k] = count[hi] - count[lo] + n_recent[:, k] >= cfg.min_samples
        return x_t0, coef, ok

    def recent_rows(self, x, y, local, lo, chunk=64):
        """Normal-equation terms from the rows p = P-72 .. P-1 before each t0 at P.

        Their runs end after t0, so targets at or after t0 are dropped and the loss
        weights are recomputed from the targets that remain.
        """
        n_feat = x.shape[-1]
        offset = np.arange(1, HORIZON + 1)  # row p = P - offset
        # lead k + 1 is observed by t0 if p + k + 1 <= P - 1, i.e. k <= offset - 2
        seen = np.arange(HORIZON)[None, :] <= offset[:, None] - 2
        a = np.zeros((len(local), HORIZON, n_feat, n_feat))
        b = np.zeros((len(local), HORIZON, n_feat))
        n = np.zeros((len(local), HORIZON))
        for i in range(0, len(local), chunk):
            rows = local[i : i + chunk, None] - offset[None, :]
            inside = (rows >= lo[i : i + chunk, None]) & (rows >= 0)
            rows = np.clip(rows, 0, None)
            yt = np.where(inside[..., None] & seen[None], y[rows], np.nan)
            qt = self.weights(yt.reshape(-1, HORIZON)).reshape(yt.shape)
            xt = x[rows]
            a[i : i + chunk] = np.einsum("cokf,cok,cokg->ckfg", xt, qt, xt, optimize=True)
            b[i : i + chunk] = np.einsum(
                "cokf,cok->ckf", xt, np.where(qt > 0, qt * yt, 0.0), optimize=True
            )
            n[i : i + chunk] = (qt > 0).sum(axis=1)
        return a, b, n

    def combine(self, x_t0, rolling_coef, rolling_ok, rolling_weight=None):
        """Regression forecast R with coefficients blended from rolling and frozen."""
        lam = self.config.rolling_weight if rolling_weight is None else rolling_weight
        coef = np.where(
            rolling_ok[..., None], lam * rolling_coef + (1 - lam) * self.coef, self.coef
        )
        return np.einsum("pkf,pkf->pk", x_t0, coef)  # the intercept carries the level

    def finish(self, regression, last, weight=None):
        """Persistence blend and sanity clip."""
        w = self.persistence_weight if weight is None else weight
        last = np.where(np.isfinite(last), last, self.mean)
        out = w * last[:, None] + (1 - w) * regression
        return np.clip(out, *VALID_RANGE)

    def forecast(self, values, positions, blend=True):
        """Forecasts (len(positions) x 72) for a t0 at each position, using values[:P] only."""
        values = np.asarray(values, dtype=float)
        positions = np.asarray(positions)
        regression = self.combine(*self.rolling(values, positions))
        if not blend:
            return regression
        return self.finish(regression, last_observed(values, positions))

    def predict(self, history, t0=None):
        """72 forecasts for the hours after `history` (every hour fully observed by t0).

        `history` is an hourly Series ending at t0 - 1h; missing hours are NaN.
        """
        values = np.asarray(history, dtype=float)
        return self.forecast(values, np.array([len(values)]))[0]

    # ------------------------------------------------------------- persistence
    def to_dict(self):
        cfg = asdict(self.config)
        cfg["lags_days"] = list(cfg["lags_days"])
        return {
            "model": "FinalBaseline",
            "format": FORMAT_VERSION,
            "config": cfg,
            "mean": self.mean,
            "run_scale": self.run_scale,
            "coef": np.round(self.coef, 8).tolist(),
            "persistence_weight": np.round(self.persistence_weight, 8).tolist(),
            "tail": {"knots": self.tail.knots.tolist(), "g": np.round(self.tail.g, 8).tolist()},
            "fit_info": getattr(self, "fit_info", {}),
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format {d.get('format')!r}")
        cfg = dict(d["config"])
        cfg["lags_days"] = tuple(cfg["lags_days"])
        model = cls(Config(**cfg))
        model.mean = float(d["mean"])
        model.run_scale = float(d["run_scale"])
        model.coef = np.asarray(d["coef"], dtype=float)
        model.persistence_weight = np.asarray(d["persistence_weight"], dtype=float)
        model.tail = TailWeights(d["tail"]["knots"], d["tail"]["g"])
        model.fit_info = d.get("fit_info", {})
        expected = (HORIZON, model.config.n_features)
        if model.coef.shape != expected or model.persistence_weight.shape != (HORIZON,):
            raise ValueError(f"coef shape {model.coef.shape}, expected {expected}")
        return model

    def save(self, path=ASSET):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path=ASSET):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
