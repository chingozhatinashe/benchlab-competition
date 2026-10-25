"""The CME detector's outputs in the hybrid, on the yearly folds (item 4, path (ii)).

    python -m training.cme_detector --pilot --folds 2024   # one fold, timed, not cached
    python -m training.cme_detector [--max-seconds 240]    # every fold; resumable
    python -m training.cme_detector --summary

models.cme_detector turns the LASCO C2 images (training.pull_lasco_images) into
P(wide CME in the image) (stage A, trained on the CDAW catalogue) and then into
P(a CME reaches L1 in the next 0-24, 24-48, 48-72 h) (stage B, trained on the CMEs
found in the ACE data, with GOES flares as the on-disk confirmation).

Per fold (validation [S, E)), only data before S trains anything:
- stage A is fitted on the images before S. For the training years themselves its
  probabilities are cross-fitted by year (each year's images scored by a detector
  fitted on the other training years), so stage B and the hybrid train on honest,
  out-of-sample detector outputs, as they will see them live;
- stage B likewise: fitted on t0 < S - 72 h (its labels look 72 h ahead),
  cross-fitted by year for the training t0s;
- the hybrid is refitted (shipped settings, shipped linear model) with the outputs
  added to its 30 regressors.

Variants:
    shipped            the shipped hybrid
    hybrid + arrival   the three stage-B probabilities (cme_l1_0_24h, ...)
    hybrid + detector  the stage-A maxima per day bin (cme_p_0_24h ... cme_p_72_96h)
    hybrid + both      the seven together
Score: per-run mean TIMSE / persistence, geometric mean over folds; also on CME runs
and the rest (as training.coronagraph_features). Also per fold: stage A's AUC on the
fold's images and stage B's AUC on the fold's t0s. Decision rule (set before running):
more than 1% better on the six-fold score, winning at least 4 of 6 folds, with the CME
runs better and the other runs not worse by more than 0.5%.
Results: src/experiments/cme_detector.csv.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from models import cme_detector as cd
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import coronagraph_features as cf
from training import pull_lasco_images as pl
from training import regime_models as rm
from training import solar_wind_events as ev
from training.datasets import DATA_START, TRAIN_END
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cme_detector_cache"
FLARES = pl.OUT / "flares.csv"
CDAW = Path(__file__).resolve().parents[1] / "data" / "lasco_cme" / "cmes.csv"
ARRIVAL = tuple(f"cme_l1_{a}_{b}h" for a, b in cd.HORIZONS)
DETECTOR = tuple(f"cme_p_{a}_{b}h" for a, b in cd.BINS)
SETS = {"arrival": ARRIVAL, "detector": DETECTOR, "both": (*ARRIVAL, *DETECTOR)}
YEAR_STARTS = pd.DatetimeIndex([pd.Timestamp(f"{y}-10-20T00:00Z") for y in range(2016, 2026)])
_REFERENCE = {}  # fold start -> (shipped hybrid forecast, ratio)


def log(msg):
    print(msg, flush=True)


def variant_names():
    return ("shipped", *(f"hybrid + {k}" for k in SETS))


def year_block(times):
    """Index of the training year (from 20 October) each time falls in."""
    return np.searchsorted(YEAR_STARTS, pd.DatetimeIndex(times), side="right") - 1


class Inputs:
    """Everything the detector needs, built once: image features, labels, flares, onsets."""

    def __init__(self):
        cells = pl.load_cells()
        c2 = cells[cells["instrument"] == "c2"].set_index("time")
        self.images = cd.image_features(c2.dropna(subset=["r0s0"]))
        self.times = self.images.index
        cdaw = pd.read_csv(CDAW)
        onsets = pd.to_datetime(cdaw.loc[cdaw["width"] >= cd.IMAGE_MIN_WIDTH, "time"], utc=True)
        self.image_y = cd.image_labels(self.times, onsets)
        flares = pd.read_csv(FLARES)
        flares["peak_time"] = pd.to_datetime(flares["peak_time"], utc=True)
        self.flare = cd.flare_near(self.times, flares)
        self.l1_onsets = self._l1_onsets()

    @staticmethod
    def _l1_onsets():
        path = CACHE_DIR / "l1_onsets.pkl"
        if path.exists():
            return pd.read_pickle(path)
        idx = pd.date_range(DATA_START, TRAIN_END, freq="h", inclusive="left")
        cme, _, _ = ev.hourly_events(idx[0], idx[-1] + pd.Timedelta(hours=1), idx)
        onsets = cme.index[cme.to_numpy() & ~cme.shift(1, fill_value=False).to_numpy()]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.Series(onsets).to_pickle(path)
        return pd.Series(onsets)


def cross_fit(fit_rows, fit_blocks, predict_rows, fit, predict):
    """Out-of-block predictions for the fitted rows and a full-fit prediction for the rest.

    `fit(rows)` returns a model; `predict(model, rows)` its predictions. Rows are index
    arrays into one table; blocks are the training years of `fit_rows`.
    """
    oof = np.full(len(fit_rows), np.nan)
    for b in np.unique(fit_blocks):
        inside = fit_blocks == b
        if (~inside).sum() == 0:
            continue
        oof[inside] = predict(fit(fit_rows[~inside]), fit_rows[inside])
    return oof, predict(fit(fit_rows), predict_rows)


def detector_table(inputs, t0s, start, end, diag):
    """The detector outputs for every t0 in `t0s` (< end), using only data before `start`."""
    times, x, y = inputs.times, inputs.images.to_numpy(), inputs.image_y
    train = np.flatnonzero(times < start)
    later = np.flatnonzero((times >= start) & (times < end))

    def fit_a(rows):
        return cd.MonotoneClassifier(cd.IMAGE_PARAMS).fit(x[rows], y[rows])

    oof, p_later = cross_fit(train, year_block(times[train]), later, fit_a,
                             lambda m, rows: m.predict(x[rows]))
    p = np.full(len(times), np.nan)
    p[train], p[later] = oof, p_later
    if y[later].any():
        from sklearn.metrics import roc_auc_score

        diag["stage A AUC"] = float(roc_auc_score(y[later], p_later))

    t0s = pd.DatetimeIndex(t0s[t0s < end])
    z = cd.arrival_inputs(times, p, inputs.flare, t0s)
    out = pd.DataFrame(index=t0s)
    for (a, b), name in zip(cd.BINS, DETECTOR, strict=True):
        out[name] = z[f"p_{a}_{b}h"].to_numpy()
    zz = z.to_numpy()
    lab = cd.arrival_labels(inputs.l1_onsets, t0s)
    b_fit = np.flatnonzero(t0s < start - pd.Timedelta(hours=72))
    b_rest = np.flatnonzero(t0s >= start - pd.Timedelta(hours=72))
    for h, name in enumerate(ARRIVAL):
        def fit_b(rows, h=h):
            return cd.MonotoneClassifier(cd.ARRIVAL_PARAMS).fit(zz[rows], lab[rows, h])

        oof, q_rest = cross_fit(b_fit, year_block(t0s[b_fit]), b_rest, fit_b,
                                lambda m, rows: m.predict(zz[rows]))
        q = np.full(len(t0s), np.nan)
        q[b_fit], q[b_rest] = oof, q_rest
        out[name] = q
        val = (t0s >= start) & (t0s < end - pd.Timedelta(hours=72))
        if lab[val, h].any():
            from sklearn.metrics import roc_auc_score

            diag[f"stage B AUC {name}"] = float(roc_auc_score(lab[val, h], q[val]))
    return out


def score_fold(data, inputs, start, end, timer, names):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos6, pos_h = data.series.index.get_indexer(fit6), data.series.index.get_indexer(fit_h)
    t = time.perf_counter()
    diag = {}
    table = detector_table(inputs, data.hourly, start, end, diag)
    timer["detector"] = timer.get("detector", 0.0) + time.perf_counter() - t

    def x(t0s, cols):
        base = data.table.loc[t0s, [c for c in cols if c not in table.columns]]
        both = pd.concat([base, table.loc[t0s, [c for c in cols if c in table.columns]]], axis=1)
        return both[list(cols)].to_numpy(dtype=float)

    cfg = data.linear_cfg
    lin = LinearModel(cfg).fit(data.values, pos6, x(fit6, cfg.input_features), end=start_pos)

    def hybrid(extra):
        hcfg = replace(data.hybrid_cfg, features=(*data.hybrid_cfg.features, *extra))
        hyb = XGBoostModel(hcfg, lin)
        hyb.fit(data.values, pos_h, x(fit_h, hyb.input_features), end=start_pos)
        return hyb.forecast(data.values, fold["pos"], x(fold["t0s"], hyb.input_features))

    out = dict(diag)
    if start not in _REFERENCE:
        f0 = hybrid(())
        _REFERENCE[start] = (f0, data.ratio(fold, f0))
    ref, out["hybrid shipped"] = _REFERENCE[start]
    for name in names:
        t = time.perf_counter()
        f = hybrid(SETS[name.removeprefix("hybrid + ")])
        out[name] = data.ratio(fold, f)
        for k, v in cf.route_ratios(data, fold, f, ref).items():
            out[f"{name} {k}"] = v
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return out


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "cme_detector.csv", index=False, float_format="%.6g")
    table = {}
    ref = "hybrid shipped"
    for name in variant_names()[1:]:
        if name not in df:
            continue
        geo = float(np.exp(np.log(df[name]).mean()))
        geo_ref = float(np.exp(np.log(df[ref]).mean()))
        table[name] = {"score": geo, "vs shipped": geo / geo_ref - 1,
                       "folds won": int((df[name] < df[ref]).sum()),
                       "CME runs vs shipped": float(np.exp(np.log(df[f"{name} cme"]).mean())) - 1,
                       "other runs vs shipped":
                           float(np.exp(np.log(df[f"{name} other"]).mean())) - 1}
    table = pd.DataFrame(table).T
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df[[c for c in df.columns if not c.endswith((" cme", " other"))]]
            .T.to_string())
        log(f"\nover {len(df)} folds:\n{table.to_string()}")
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = rm.Data()
    inputs = Inputs()
    log(f"data ready in {time.perf_counter() - started:.0f} s; {len(inputs.times)} C2 images, "
        f"{int(inputs.image_y.sum())} positive; {len(inputs.l1_onsets)} CME onsets at L1")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    names = variant_names()[1:]
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        cached = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = tuple(n for n in names if n not in cached)
        if not todo:
            rows.append(cached)
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        new = score_fold(data, inputs, start, end, timer, todo)
        r = {"fold": f"{start:%Y-%m}", **cached, **new}
        if not pilot:
            path.write_text(json.dumps(r), encoding="utf-8")
        log(f"  fold {r['fold']} ({time.perf_counter() - t:.0f} s): " + ", ".join(
            f"{k} {v:.4f}" for k, v in new.items() if not k.endswith((" cme", " other"))))
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB; six folds -> ~"
                f"{6 * (time.perf_counter() - t) / 60:.0f} min")
            return r
        rows.append(r)
    return summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summarize([json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(CACHE_DIR.glob("*.json"))])
    else:
        run(args.folds, args.pilot, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
