"""Bayes stream's fitted quantities as regressors of the submitted linear model, yearly folds.

    python -m training.bayes_features --pilot --folds 2024   # one fold, timed, not cached
    python -m training.bayes_features [--max-seconds N]      # every fold; resumable
    python -m training.bayes_features --summary

Follow-up to docs/bayes_stream.md (user, 2026-11-15): `linear + bayes` gave the linear model
only the generative model's blended mean path, and tied the submitted model. The model is
good in quiet wind and bad at stream amplitudes, so here its parts go in separately, and the
linear model can take what helps. The fitted global parameters are the same for every run of
a fold (constant columns), so the features are what they imply run by run from each run's
inputs. Per run, averaged over N_DRAWS posterior draws of the fold's cached additive fit
(training.bayes_stream; no refit):

    ambient   the ambient level a = 250 + 250 sigmoid(w_a . [1, q20, rec27]), every lead
    slow      the slow-wind path a + rho^k (v0 - a)
    parr      P(N >= 1) x P(first arrival <= k): stream timing without the amplitude
    stream    the expected stream excess sum_u P(N >= 1) pi_u E(x(u)) profile(k; u)
              (one arrival: E and D at their population medians given x(u), R = exp(mu_R))
    mean      slow + stream: the additive model's mean path (the `linear + bayes` signal,
              without the simulation noise), for comparison

Each is a lead-aligned regressor "structural_bayes_<name>_lead_0h" (models.linear_model:
lead k reads the column for its 6-hour block). The fit runs' features come from the same
fold's posterior, fitted on the runs before the fold (as `linear + bayes`).
Variants, set before running: each signal alone; "slow + parr" (the quiet-wind part and the
timing, without amplitudes); "components" (ambient, slow, parr, stream); and "forward", a
greedy forward selection over the five by the geometric mean over folds (needs every fold:
in the pilot, by the pilot fold). Decision rule: keep if > 1% better than the reference and
winning >= 4 of 6 folds. The eval year is not used.
--fit latent takes the features from the latent-arrival fit (`bayes latent`; cached in
training.bayes_stream, 2024 so far) instead of the observed-arrival one.
Results: src/experiments/bayes_features.csv (cache: src/data/bayes_features_cache/);
latent: bayes_features_latent.csv (cache: bayes_features_cache/latent/).
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from models import bayes_model as bm
from models.bayes_model import features_in_blocks, lead_columns, signal_name
from models.linear_model import LinearModel
from training import bayes_stream as tbs
from training import competition_score as cs
from training import model_diagnostics as dg
from training import xgboost_model as tx
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "bayes_features_cache"
FIT = "observed"  # the bayes_stream fit the features come from (--fit)
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
SIGNALS = bm.ALL_SIGNALS
SETS = {**{s: (s,) for s in SIGNALS}, "slow + parr": ("slow", "parr"),
        "components": ("ambient", "slow", "parr", "stream")}


def cache_dir():
    return CACHE_DIR if FIT == "observed" else CACHE_DIR / FIT


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------- features
def feature_frame(feats, index):
    cols = {}
    for s, f in feats.items():
        cols.update(lead_columns(signal_name(s), f))
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------- study
def fold_parts(data, start, end):
    """Everything a fold needs: fit and validation runs, inputs, features, truth."""
    year = f"{start:%Y}"
    path = tbs.CACHE_DIR / f"{year}_{FIT}_draws.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run training.bayes_stream for the fold first")
    draws = dict(np.load(path))
    ff = tbs.fold_fit(data, start)
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    table, cols, model = ff["table"], ff["cols"], ff["model"]
    x_val = table.loc[val, cols].to_numpy(dtype=float)
    inp_fit = model.prepare(data.values, ff["pos_fit"], ff["x_fit"])
    inp_val = model.prepare(data.values, data.series.index.get_indexer(val), x_val)
    cfg = tbs.FITS[FIT]
    t = time.perf_counter()
    f_fit = feature_frame(features_in_blocks(draws, inp_fit, cfg), ff["fit6"])
    f_val = feature_frame(features_in_blocks(draws, inp_val, cfg), val)
    return {"ff": ff, "val": val, "x_val": x_val, "f_fit": f_fit, "f_val": f_val,
            "feature s": time.perf_counter() - t,
            "obs": cs.verifying(data.truth, val), "pers": cs.persistence(data.truth, val),
            "regime": data.run_regimes(val)}


def score_set(data, parts, signals):
    """Regime scores of the submitted config with `signals` added (none: the reference)."""
    ff = parts["ff"]
    cfg = replace(data.cfg, features=(*data.cfg.features,
                                      *(f"{signal_name(s)}_lead_0h" for s in signals)))
    cols = list(cfg.input_features)
    fit_tab = ff["table"].loc[ff["fit6"]].join(parts["f_fit"])
    val_tab = ff["table"].loc[parts["val"]].join(parts["f_val"])
    lin = LinearModel(cfg).fit(data.values, ff["pos_fit"],
                               fit_tab[cols].to_numpy(dtype=float), end=ff["start_pos"])
    f = lin.forecast(data.values, data.series.index.get_indexer(parts["val"]),
                     val_tab[cols].to_numpy(dtype=float))
    return dg.regime_scores(f, parts["obs"], parts["pers"], parts["regime"])


def gmean_rel(rows, years, name):
    rel = np.array([rows[y][name]["all"] / rows[y]["reference"]["all"] for y in years])
    return float(np.exp(np.log(rel).mean())), int((rel < 1).sum())


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = dg.Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    cache_dir().mkdir(parents=True, exist_ok=True)
    rows, parts_by_year, timer = {}, {}, {}
    folds = [(s, e) for s, e in YEARLY if years is None or s.year in years]
    for start, end in folds:
        year = f"{start:%Y}"
        path = cache_dir() / f"{year}.json"
        row = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = [n for n in ("reference", *SETS) if n not in row]
        if todo:
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
            t = time.perf_counter()
            parts = fold_parts(data, start, end)
            parts_by_year[year] = parts
            timer["features"] = timer.get("features", 0.0) + parts["feature s"]
            log(f"  {year}: features in {parts['feature s']:.0f} s")
            for name in todo:
                t1 = time.perf_counter()
                row[name] = score_set(data, parts, () if name == "reference" else SETS[name])
                timer["fits"] = timer.get("fits", 0.0) + time.perf_counter() - t1
                log(f"    {name}: S_final {row[name]['all']:.2f}")
            log(f"  {year}: {time.perf_counter() - t:.0f} s")
            if not pilot:
                path.write_text(json.dumps(row), encoding="utf-8")
        rows[year] = row
    years_done = sorted(rows)
    forward = forward_selection(data, rows, years_done, parts_by_year, pilot, timer)
    if pilot:
        log(f"  time: {', '.join(f'{k} {v:.0f} s' for k, v in timer.items())}; peak memory "
            f"{tx.peak_mb():.0f} MB; six folds -> ~"
            f"{6 * sum(timer.values()) / max(len(rows), 1) / 60:.0f} min after the data")
    return summarize(rows, forward, pilot)


def forward_selection(data, rows, years, parts_by_year, pilot, timer):
    """Greedy: add the signal that most lowers the geometric mean over `years`, while it
    lowers it. Each step's scores are cached in the folds' rows under "forward <set>"."""
    chosen, best = [], 1.0
    path_out = [("reference", 1.0)]
    while len(chosen) < len(SIGNALS):
        trial = {}
        for s in SIGNALS:
            if s in chosen:
                continue
            key = "forward " + " + ".join([*chosen, s])
            for y in years:
                if key not in rows[y]:
                    if y not in parts_by_year:
                        start, end = next((a, b) for a, b in YEARLY if f"{a:%Y}" == y)
                        parts_by_year[y] = fold_parts(data, start, end)
                    t = time.perf_counter()
                    rows[y][key] = score_set(data, parts_by_year[y], (*chosen, s))
                    timer["fits"] = timer.get("fits", 0.0) + time.perf_counter() - t
                    if not pilot:
                        (cache_dir() / f"{y}.json").write_text(json.dumps(rows[y]), "utf-8")
            trial[s] = gmean_rel(rows, years, key)[0]
        s_best = min(trial, key=trial.get)
        if trial[s_best] >= best:
            break
        chosen.append(s_best)
        best = trial[s_best]
        path_out.append((" + ".join(chosen), best))
        log(f"  forward: + {s_best} -> {100 * (best - 1):+.2f}%")
    for y in years:
        rows[y]["forward"] = (rows[y]["forward " + " + ".join(chosen)] if chosen
                              else rows[y]["reference"])
        rows[y]["forward chosen"] = chosen
        if not pilot:
            (cache_dir() / f"{y}.json").write_text(json.dumps(rows[y]), "utf-8")
    return {"chosen": chosen, "path": path_out}


def summarize(rows, forward, pilot=False):
    years = sorted(rows)
    names = ["reference", *SETS, "forward"]
    out, long = {}, []
    for name in names:
        g, won = gmean_rel(rows, years, name)
        out[name] = {"vs reference": g - 1, "folds won": won,
                     **{f"{r} vs reference": float(np.exp(np.mean(
                         [np.log(rows[y][name][r] / rows[y]["reference"][r]) for y in years]))
                         - 1) for r in dg.RUN_REGIMES},
                     "keep": bool(g - 1 < -KEEP_GAIN and won >= KEEP_FOLDS)}
        long += [{"fold": y, "variant": name, **rows[y][name]} for y in years]
    table = pd.DataFrame(out).T.sort_values("vs reference")
    if not pilot:
        pd.DataFrame(long).to_csv(EXPERIMENTS / ("bayes_features.csv" if FIT == "observed"
                                                 else f"bayes_features_{FIT}.csv"), index=False,
                                  float_format="%.6g")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 30):
        log(f"\nfolds {', '.join(years)}:\n{table.to_string()}")
        if forward:
            log("forward path: " + " -> ".join(f"{s} ({100 * (g - 1):+.2f}%)"
                                                for s, g in forward["path"]))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--fit", choices=("observed", "latent"), default="observed")
    args = parser.parse_args(argv)
    global FIT
    FIT = args.fit
    if args.summary:
        rows = {p.stem: json.loads(p.read_text("utf-8"))
                for p in sorted(cache_dir().glob("20??.json"))}
        summarize(rows, None)
        return
    run(args.folds, args.pilot, args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
