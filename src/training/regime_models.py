"""Feasibility study: one model per solar-wind regime, routed by a regime classifier.

    python -m training.regime_models --pilot --folds 2024   # one fold, timed, not cached
    python -m training.regime_models [--max-seconds 240]    # every fold; resumable
    python -m training.regime_models --summary              # scores from the cached folds

Regimes (training.solar_wind_events.ROUTES) are labelled from what happens during
each run's 72 target hours, so a router must predict them. Per fold
(training.folds.YEARLY; fitting on the hours before the fold start):

    global      the shipped linear model and XGBoost hybrid (as training.compare_history)
    per regime  the same configs fitted on one regime's runs only; a regime with
                fewer than MIN_RUNS fit runs uses the global model
    classifier  P(regime) at t0 from the hybrid's regressors plus the speed terms:
                multinomial logistic regression and XGBoost (multi:softprob)

Only fit runs whose 72 target hours end before the fold start get a regime label
(per-regime models, classifiers), so no label looks past the start.

Every validation run's true regime, probabilities and member forecasts are cached
in src/data/regime_cache/{year}.pkl; the combinations (oracle, hard and soft
routing, soft routing shrunk towards the global model) are computed from them
without refitting (`scores`). Results: src/experiments/regime_models.csv; the
analysis is src/experiments/regime_models.ipynb, write-up docs/regime_models.md. Dev-only;
nothing here ships.
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import HORIZON, LEADS, last_observed, window_means
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import linear_model as tl
from training import solar_wind_events as ev
from training.datasets import DATA_START, TRAIN_END
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "regime_cache"
ROUTES = ev.ROUTES
N_ROUTES = len(ROUTES)
MEMBERS = ("linear", "hybrid")
CLASSIFIERS = ("logistic", "xgboost")
MIN_RUNS = 200  # 6-hourly fit runs a regime needs for its own model (50 days)
SHRINK_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------- data
class Data(ch.Data):
    """compare_history.Data plus a regime label for every hourly training run."""

    def __init__(self):
        super().__init__()
        path = CACHE_DIR / "labels.pkl"
        if path.exists():
            labels = pd.read_pickle(path)
        else:
            t = time.perf_counter()
            cme, possible, arrivals = ev.hourly_events(DATA_START, TRAIN_END, self.series.index)
            pos = self.series.index.get_indexer(self.hourly)
            regime = ev.label_runs(pos, self.values, cme, possible, arrivals)
            labels = pd.DataFrame({"regime": regime}, index=self.hourly)
            labels["route"] = pd.Categorical(labels["regime"].map(ev.ROUTE_OF), ROUTES)
            path.parent.mkdir(parents=True, exist_ok=True)
            labels.to_pickle(path)
            log(f"labelled {len(labels)} hourly runs in {time.perf_counter() - t:.0f} s")
        self.labels = labels
        self.lags = self.hybrid_cfg.lag_hours
        self.half_width = self.hybrid_cfg.half_width

    def route(self, t0s):
        """Route index 0..4 per run."""
        return self.labels.loc[t0s, "route"].cat.codes.to_numpy()

    def classifier_x(self, t0s):
        """Inputs available at t0: the hybrid's regressors plus the speed terms."""
        pos = self.series.index.get_indexer(t0s)
        last = last_observed(self.values, pos)
        cols = [self.table.loc[t0s, list(self.hybrid_cfg.features)].to_numpy(dtype=float),
                last[:, None]]
        for lag in self.lags:  # each recurrence term over the 72 target hours: mean, max
            rec = window_means(self.values, pos[:, None] + LEADS - lag, self.half_width)
            with warnings.catch_warnings():  # an all-missing window gives NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                cols += [np.nanmean(rec, axis=1)[:, None], np.nanmax(rec, axis=1)[:, None]]
        return np.hstack(cols)


# ---------------------------------------------------------------------- classifiers
def fit_logistic(x, y):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=3000))
    return model.fit(x, y)


def proba_logistic(model, x):
    p = np.zeros((len(x), len(ROUTES)))
    p[:, model.classes_] = model.predict_proba(x)  # a class absent in fitting gets 0
    return p


XGB_CLASSIFIER = {"objective": "multi:softprob", "num_class": len(ROUTES), "max_depth": 3,
                  "eta": 0.05, "min_child_weight": 5, "subsample": 0.8,
                  "colsample_bytree": 0.8, "seed": 0}
XGB_ROUNDS = 200


def fit_xgb(x, y):
    import os

    import xgboost as xgb

    params = {**XGB_CLASSIFIER, "nthread": int(os.environ.get("XGB_NTHREAD", "0"))}
    return xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=XGB_ROUNDS)


def proba_xgb(model, x):
    import xgboost as xgb

    return model.predict(xgb.DMatrix(x))


# ---------------------------------------------------------------------- one fold
def run_fold(data, start, end, timer):
    """Fit everything for one fold; return the validation runs' arrays."""
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    labelled_end = start - pd.Timedelta(hours=HORIZON)  # targets end before the start

    def clock(name, t):
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t

    out = {"t0s": fold["t0s"], "y": fold["y"].astype(np.float32), "q": fold["q"],
           "scored": fold["scored"], "ref": fold["ref"], "route": data.route(fold["t0s"]),
           "regime": data.labels.loc[fold["t0s"], "regime"].to_numpy()}

    # global models (the reference): fitted exactly as in compare_history
    t = time.perf_counter()
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    lin = LinearModel(data.linear_cfg).fit(
        data.values, data.series.index.get_indexer(fit6),
        data.x(fit6, data.linear_cfg.features), end=start_pos)
    out["global_linear"] = lin.forecast(
        data.values, fold["pos"], data.x(fold["t0s"], data.linear_cfg.features))
    clock("global linear", t)
    t = time.perf_counter()
    hyb = XGBoostModel(data.hybrid_cfg, lin)
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    hyb.fit(data.values, data.series.index.get_indexer(fit_h),
            data.x(fit_h, hyb.input_features), end=start_pos)
    xh = data.x(fold["t0s"], hyb.input_features)
    out["global_hybrid"] = hyb.forecast(data.values, fold["pos"], xh)
    clock("global hybrid", t)

    # per-regime models
    lab6 = fit6[fit6 < labelled_end]
    lab_h = fit_h[fit_h < labelled_end]
    route6, route_h = data.route(lab6), data.route(lab_h)
    n = len(fold["t0s"])
    regime_linear = np.empty((len(ROUTES), n, HORIZON), dtype=np.float32)
    regime_hybrid = np.empty((len(ROUTES), n, HORIZON), dtype=np.float32)
    fit_runs = {}
    for r, name in enumerate(ROUTES):
        sel6, sel_h = lab6[route6 == r], lab_h[route_h == r]
        fit_runs[name] = int(len(sel6))
        if len(sel6) < MIN_RUNS:
            log(f"    {name}: {len(sel6)} fit runs < {MIN_RUNS}; using the global models")
            regime_linear[r], regime_hybrid[r] = out["global_linear"], out["global_hybrid"]
            continue
        t = time.perf_counter()
        lin_r = LinearModel(data.linear_cfg).fit(
            data.values, data.series.index.get_indexer(sel6),
            data.x(sel6, data.linear_cfg.features), end=start_pos)
        regime_linear[r] = lin_r.forecast(
            data.values, fold["pos"], data.x(fold["t0s"], data.linear_cfg.features))
        clock("regime linear", t)
        t = time.perf_counter()
        hyb_r = XGBoostModel(data.hybrid_cfg, lin_r)
        hyb_r.fit(data.values, data.series.index.get_indexer(sel_h),
                  data.x(sel_h, hyb_r.input_features), end=start_pos)
        regime_hybrid[r] = hyb_r.forecast(data.values, fold["pos"], xh)
        clock("regime hybrid", t)
    out.update(regime_linear=regime_linear, regime_hybrid=regime_hybrid, fit_runs=fit_runs)
    out["global_linear"] = out["global_linear"].astype(np.float32)
    out["global_hybrid"] = out["global_hybrid"].astype(np.float32)

    # classifiers, on the labelled 6-hourly fit runs
    t = time.perf_counter()
    x_fit, x_val = data.classifier_x(lab6), data.classifier_x(fold["t0s"])
    clock("classifier inputs", t)
    t = time.perf_counter()
    out["p_logistic"] = proba_logistic(fit_logistic(x_fit, route6), x_val)
    clock("logistic", t)
    t = time.perf_counter()
    out["p_xgboost"] = proba_xgb(fit_xgb(x_fit, route6), x_val)
    clock("xgboost classifier", t)
    out["prior"] = np.bincount(route6, minlength=len(ROUTES)) / len(route6)
    return out


# ---------------------------------------------------------------------- combinations
def mix(probs, members):
    """Σ_r P(r) × member r: probs (n, R), members (R, n, 72) -> (n, 72)."""
    return np.einsum("nr,rnk->nk", np.asarray(probs, dtype=float),
                     np.asarray(members, dtype=float))


def one_hot(index, size=N_ROUTES):
    out = np.zeros((len(index), size))
    out[np.arange(len(index)), index] = 1.0
    return out


def forecasts(f, shrink=None):
    """{method: (n, 72) forecast} for one cached fold."""
    shrink = shrink or {}
    out = {f"global {m}": f[f"global_{m}"] for m in MEMBERS}
    for m in MEMBERS:
        members = f[f"regime_{m}"]
        out[f"oracle {m}"] = mix(one_hot(f["route"]), members)
        out[f"prior {m}"] = mix(np.repeat(f["prior"][None], len(f["route"]), 0), members)
        for c in CLASSIFIERS:
            p = f[f"p_{c}"]
            out[f"hard {c} {m}"] = mix(one_hot(p.argmax(axis=1)), members)
            soft = mix(p, members)
            out[f"soft {c} {m}"] = soft
            a = shrink.get((c, m))
            if a is not None:
                out[f"shrunk soft {c} {m}"] = a * soft + (1 - a) * f[f"global_{m}"]
    return out


def ratio(f, forecast):
    s = tl.run_scores(forecast, f["y"], f["q"])
    return float(np.nanmean(s[f["scored"]]) / f["ref"])


def geo(values):
    return float(np.exp(np.mean(np.log(values))))


def choose_shrink(folds):
    """The weight on the soft forecast that minimises the geometric fold score."""
    best = {}
    for c in CLASSIFIERS:
        for m in MEMBERS:
            scores = {a: geo([ratio(f, a * mix(f[f"p_{c}"], f[f"regime_{m}"])
                                    + (1 - a) * f[f"global_{m}"]) for f in folds.values()])
                      for a in SHRINK_GRID}
            best[(c, m)] = float(min(scores, key=scores.get))
    return best


def scores(folds):
    """(DataFrame fold x method of ratios to persistence, shrink weights)."""
    shrink = choose_shrink(folds)
    rows = []
    for year, f in folds.items():
        rows.append({"fold": f"{year}-10", **{k: ratio(f, v)
                                               for k, v in forecasts(f, shrink).items()}})
    return pd.DataFrame(rows), shrink


def load_folds(years=None):
    out = {}
    for start, _ in YEARLY:
        path = CACHE_DIR / f"{start:%Y}.pkl"
        if (years is None or start.year in years) and path.exists():
            with path.open("rb") as fh:
                out[start.year] = pickle.load(fh)
    return out


def summarize(folds):
    df, shrink = scores(folds)
    df.to_csv(EXPERIMENTS / "regime_models.csv", index=False, float_format="%.6g")
    summary = pd.Series({k: geo(df[k]) for k in df.columns if k != "fold"}).sort_values()
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold (per-run mean TIMSE / persistence):\n"
            + df.set_index("fold").T.to_string())
        log(f"\ngeometric mean over {len(df)} folds:\n{summary.to_string()}")
    log(f"\nshrink weights on the soft forecast: {shrink}")
    return df


# ---------------------------------------------------------------------- driver
def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer = {}
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.pkl"
        if path.exists() and not pilot:
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        f = run_fold(data, start, end, timer)
        f["seconds"] = time.perf_counter() - t
        log(f"  fold {start:%Y-%m}: {f['seconds']:.0f} s; fit runs per regime {f['fit_runs']}")
        if pilot:
            ref = {k: ratio(f, v) for k, v in forecasts(f).items()}
            log("  scores: " + ", ".join(f"{k} {v:.4f}" for k, v in ref.items()))
            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB")
            log(f"  six folds -> ~{6 * f['seconds'] / 60:.0f} min (earlier folds fit on less)")
            return f
        with path.open("wb") as fh:
            pickle.dump(f, fh)
    folds = load_folds(years)
    if len(folds) < len([s for s, _ in YEARLY if years is None or s.year in years]):
        log("some folds are missing; run again")
        return None
    return summarize(folds)


def peak_mb():
    from training.xgboost_model import peak_mb as peak

    return peak()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="fold start years (default all six)")
    parser.add_argument("--pilot", action="store_true", help="one timed pass, not cached")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--summary", action="store_true", help="score the cached folds only")
    args = parser.parse_args(argv)
    if args.summary:
        summarize(load_folds(args.folds))
    else:
        run(args.folds, args.pilot, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
