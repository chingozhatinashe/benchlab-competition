"""A two-stage CME detector on LASCO C2 images (item 4, path (ii); docs/cme_detector.md).

Stage A, the image detector: per C2 image, P(a wide CME began in the last 3 h).
    Inputs: the image's sector brightenings (feature_engineering.coronagraph.excess)
    sorted from largest to smallest within each ring, plus the share of brightened
    outer sectors. Sorting makes the inputs independent of the position angle: a CME
    is a CME whichever way it leaves the Sun. Trained on the CDAW catalogue (wide
    CMEs, IMAGE_MIN_WIDTH), which is published months late: the labels are used for
    training only; live runs see only the images.
Stage B, arrival at L1: per t0, P(an in-situ CME starts at L1 in (t0 + a, t0 + b]) for
    the horizons in HORIZONS. Inputs: the largest stage-A probability among images
    taken in each of the past-day bins BINS, and the largest GOES flare peaking within
    FLARE_HOURS before each image (an on-disk eruption: a front-side, possibly
    Earth-directed CME). Trained on the CMEs found in the ACE data
    (training.solar_wind_events).
Physics enters as constraints, not as a model: every input of both stages is
monotone increasing (a brighter or more widespread brightening, or a larger flare,
never lowers a probability), and stage B only sees images 0-4 days old (CME transit
times to L1 are 1-4 days), binned by day so it learns the arrival delay itself.

Both stages are xgboost binary classifiers, trained in dev and evaluated in numpy
(models.xgboost_model.TreeEnsemble), so the image needs no xgboost.
"""

import numpy as np
import pandas as pd

from feature_engineering import coronagraph as cg
from models.xgboost_model import TreeEnsemble

IMAGE_MIN_WIDTH = 270  # CDAW angular width (deg) of the CMEs stage A detects
IMAGE_LABEL_HOURS = 3  # an image is positive if such a CME's onset is in (t - 3 h, t]
FLARE_HOURS = 3
NO_FLARE = -9.0  # log10 flux used when no flare peaked near an image
BINS = ((0, 24), (24, 48), (48, 72), (72, 96))  # hours before t0
HORIZONS = ((0, 24), (24, 48), (48, 72))  # hours after t0
IMAGE_PARAMS = {"n_trees": 300, "max_depth": 3, "eta": 0.05, "subsample": 0.8}
ARRIVAL_PARAMS = {"n_trees": 200, "max_depth": 2, "eta": 0.05, "subsample": 0.8}
HOUR_NS = 3600 * 10**9


def image_features(cells):
    """Stage-A inputs per C2 image: (frame indexed by image time, NaN rows dropped)."""
    ex = cg.excess(cells[cg.cell_columns()]).dropna()
    rings = [np.sort(ex[[f"r{r}s{s}" for s in range(cg.N_SECTORS)]].to_numpy(), axis=1)[:, ::-1]
             for r in (1, 0)]
    share = (rings[0] > cg.EXCESS_LEVEL).mean(axis=1, keepdims=True)
    cols = ([f"outer{k}" for k in range(cg.N_SECTORS)] + [f"inner{k}" for k in range(cg.N_SECTORS)]
            + ["share"])
    return pd.DataFrame(np.hstack([*rings, share]), index=ex.index, columns=cols)


def image_labels(times, onsets, hours=IMAGE_LABEL_HOURS):
    """True where a catalogue onset falls in (t - hours, t]."""
    on = np.sort(pd.DatetimeIndex(onsets).asi8)
    t = pd.DatetimeIndex(times).asi8
    return np.searchsorted(on, t, side="right") > np.searchsorted(on, t - hours * HOUR_NS,
                                                                  side="right")


def flare_near(times, flares, hours=FLARE_HOURS):
    """log10 of the largest flare peak in [t - hours, t] per time (NO_FLARE if none).

    `flares`: peak_time | peak_flux. Images count for t0 only if taken by
    t0 - coronagraph.LATENCY, later than the flare list's own latency, so every flare
    used here was listed by t0.
    """
    if flares is None or not len(flares):
        return np.full(len(times), NO_FLARE)
    f = flares.sort_values("peak_time")
    ft = pd.DatetimeIndex(f["peak_time"]).asi8
    fv = np.log10(np.clip(f["peak_flux"].to_numpy(dtype=float), 1e-9, None))
    t = pd.DatetimeIndex(times).asi8
    hi = np.searchsorted(ft, t, side="right")
    lo = np.searchsorted(ft, t - hours * HOUR_NS, side="left")
    out = range_max(fv, lo, hi)
    return np.where(np.isnan(out), NO_FLARE, out)


def range_max(values, lo, hi):
    """max(values[lo[i]:hi[i]]) per i, ignoring NaN; NaN for an empty range (sparse table)."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.asarray(lo), np.asarray(hi)
    out = np.full(len(lo), np.nan)
    length = hi - lo
    ok = length > 0
    if not ok.any():
        return out
    levels = [values]
    while 2 ** len(levels) <= len(values):
        prev, half = levels[-1], 2 ** (len(levels) - 1)
        levels.append(np.fmax(prev[:-half], prev[half:]))
    j = np.floor(np.log2(np.where(ok, length, 1))).astype(int)
    for level in np.unique(j[ok]):
        m = ok & (j == level)
        tab = levels[level]
        out[m] = np.fmax(tab[lo[m]], tab[hi[m] - 2 ** level])
    return out


def bin_max(times, values, t0s, lo_h, hi_h):
    """Largest value among times in (t0 - hi_h, t0 - lo_h], counting only times taken
    by t0 - coronagraph.LATENCY; NaN where there are none."""
    order = np.argsort(pd.DatetimeIndex(times).asi8, kind="stable")
    t = pd.DatetimeIndex(times).asi8[order]
    v = np.asarray(values, dtype=float)[order]
    t0 = pd.DatetimeIndex(t0s).asi8
    upper = np.minimum(t0 - lo_h * HOUR_NS, t0 - cg.LATENCY.value)
    return range_max(v, np.searchsorted(t, t0 - hi_h * HOUR_NS, side="right"),
                     np.searchsorted(t, upper, side="right"))


def arrival_inputs(times, p, flare, t0s):
    """Stage-B inputs per t0: the largest stage-A probability and flare per day bin."""
    cols = {}
    for name, v in (("p", p), ("flare", flare)):
        for lo_h, hi_h in BINS:
            cols[f"{name}_{lo_h}_{hi_h}h"] = bin_max(times, v, t0s, lo_h, hi_h)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(t0s))


def arrival_labels(onsets, t0s):
    """(len(t0s), len(HORIZONS)) booleans: an L1 CME onset in (t0 + a, t0 + b]."""
    on = np.sort(pd.DatetimeIndex(onsets).asi8)
    t0 = pd.DatetimeIndex(t0s).asi8
    return np.column_stack([
        np.searchsorted(on, t0 + b * HOUR_NS, side="right")
        > np.searchsorted(on, t0 + a * HOUR_NS, side="right") for a, b in HORIZONS])


class MonotoneClassifier:
    """A binary xgboost classifier with every input monotone increasing, in numpy."""

    def __init__(self, params, seed=0):
        self.params = params
        self.seed = seed
        self.trees = None
        self.offset = 0.0

    def fit(self, x, y, nthread=0):
        import os

        import xgboost as xgb

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        p = self.params
        params = {"objective": "binary:logistic", "tree_method": "hist",
                  "max_depth": p["max_depth"], "eta": p["eta"], "subsample": p["subsample"],
                  "monotone_constraints": "(" + ",".join(["1"] * x.shape[1]) + ")",
                  "seed": self.seed,
                  "nthread": nthread or int(os.environ.get("XGB_NTHREAD", "0"))}
        data = xgb.DMatrix(x, label=y)
        booster = xgb.train(params, data, num_boost_round=p["n_trees"])
        self.trees = TreeEnsemble.from_booster(booster)
        # the stored base score is a probability for binary:logistic: match the margin
        probe = x[: min(len(x), 256)]
        margin = booster.predict(xgb.DMatrix(probe), output_margin=True)
        self.offset = float(np.mean(margin - self.trees.predict(probe)))
        return self

    def predict(self, x):
        margin = self.trees.predict(np.asarray(x, dtype=np.float32)) + self.offset
        return 1.0 / (1.0 + np.exp(-margin))

    def to_dict(self):
        return {"params": self.params, "seed": self.seed, "offset": self.offset,
                "trees": self.trees.to_dict()}

    @classmethod
    def from_dict(cls, d):
        out = cls(d["params"], d["seed"])
        out.offset = float(d["offset"])
        out.trees = TreeEnsemble.from_dict(d["trees"])
        return out
