"""Select, train and score the Gaussian-process model (models.gaussian_process).

Status (2026-10-22): experimental and not used live. On the training folds the
ML-II-fitted version scored about 0.63 against 0.4614 for the linear model alone:
marginal likelihood picks short lengthscales and little noise, and the GP undoes
the linear model's TIMSE hedge (it corrects "quiet-looking" runs downwards, which
the storms among them make expensive). With lengthscale and noise chosen on the
folds instead, the best correction (one GP on each run's TIMSE-weighted mean
offset, the XGBoost hybrid's 25 inputs) reached 0.4587, a 0.6% gain; the
XGBoost hybrid gains 1.9%.

    python -m training.gaussian_process select --sample 500 --seed 0   # timed pilot
    python -m training.gaussian_process select [--max-seconds 480]     # full selection
    python -m training.gaussian_process train                          # fit the chosen config
    python -m training.gaussian_process evaluate                       # eval year, once

Selection never touches the eval year. It uses the walk-forward folds of the final
baseline (training.final_baseline.FOLDS): in each fold the linear anchor and the
GP are fitted on the 6-hourly runs before the fold (targets cut at the fold
start), then the fold's 6-hourly runs are forecast and scored as per-run mean TIMSE
relative to persistence, a geometric mean over folds (the same objective as the
linear model and the XGBoost hybrid).

Candidate regressor sets: A the linear model's, H the XGBoost hybrid's, T the
hybrid's 10 regressors with the most gain. For each set and fold the GP is fitted
with MAX components once; K = 3, 5, 8 are scored from that fit (first K shapes).

--sample N fits each fold on N random fit runs (scoring unchanged) and reports
the time each part took, with an estimate for the full selection.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import last_observed, targets
from models.gaussian_process import ASSET, Config, GaussianProcessModel, neg_log_marginal
from training import linear_model as tl
from training import xgboost_model as tx
from training.datasets import DATA_START, TRAIN_END, load_train

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "gaussian_process_config.json"
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "gp_cache"
COMPONENTS = (3, 5, 8)


def log(msg):
    print(msg, flush=True)


def candidate_sets():
    hybrid = json.loads(tx.config_path("linear").read_text(encoding="utf-8"))
    gain = pd.Series(hybrid["selection"]["gain_share"])
    top = [f for f in gain.sort_values(ascending=False).index if f in hybrid["features"]][:10]
    return {"A_linear": tuple(tl.load_config().features),
            "H_hybrid": tuple(hybrid["features"]),
            "T_top10": tuple(top)}


def _fold_result(folds, i, feats, sample, seed, timer):
    """Fit the GP for fold i on (a sample of) its fit runs; score every K."""
    f = folds.folds[i]
    fit = f["fit"]
    if sample:
        fit = np.sort(np.random.default_rng(seed + i).choice(fit, size=min(sample, len(fit)),
                                                             replace=False))
    model = GaussianProcessModel(Config(features=feats, seed=seed), folds.linear(i))
    t = time.perf_counter()
    model.fit(folds.values, folds.fit_pos[fit], folds.x(folds.fit_t0s[fit], model.input_features),
              end=f["start"], log=log)
    timer.add("fit", time.perf_counter() - t)
    timer.add("ml evals", sum(m["evals"] for m in model.ml_info), count=len(model.ml_info))
    t = time.perf_counter()
    x_val = folds.x(folds.val_t0s[f["val"]], model.input_features)
    ratios = {}
    for k in COMPONENTS:
        forecast = model.forecast(folds.values, folds.val_pos[f["val"]], x_val, n_components=k)
        score = tl.run_scores(forecast, f["y"], f["q"])
        ratios[str(k)] = float(np.nanmean(score[f["scored"]]) / f["ref"])
    timer.add("score", time.perf_counter() - t)
    return {"ratios": ratios, "fit_runs": int(len(fit)), "explained": model.fit_info["explained"],
            "lengthscales": [gp.lengthscales.round(3).tolist() for gp in model.gps],
            "inputs": list(model.config.inputs)}


def select(sample=None, seed=0, max_seconds=None, fresh=False):
    started = time.perf_counter()
    timer = tl.Timer()
    tag = f"pilot{sample}_seed{seed}" if sample else "full"
    cache = CACHE_DIR / tag
    if fresh and cache.exists():
        for f in cache.glob("*.json"):
            f.unlink()
    cache.mkdir(parents=True, exist_ok=True)

    series = load_train()
    runs = tl.training_runs(series)
    sets = candidate_sets()
    names = tuple(dict.fromkeys([*(n for v in sets.values() for n in v),
                                 *tl.load_config().features]))
    table = tx.feature_table(runs, names, CACHE_DIR / "table.pkl")
    folds = tx.Folds(series, table, runs)
    log(f"{len(runs)} 6-hourly runs ({tag}); fit runs per fold: "
        + ", ".join(str(min(sample or 10**9, len(f["fit"]))) for f in folds.folds)
        + f"; sets: {', '.join(f'{k} ({len(v)})' for k, v in sets.items())}")

    results = {}
    for name, feats in sets.items():
        for i in range(len(folds.folds)):
            path = cache / f"{name}_fold{i + 1}.json"
            if path.exists():
                results[(name, i)] = json.loads(path.read_text(encoding="utf-8"))
                continue
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
            log(f"  {name}, fold {i + 1}:")
            results[(name, i)] = r = _fold_result(folds, i, feats, sample, seed, timer)
            path.write_text(json.dumps(r), encoding="utf-8")
            log(f"  {name}, fold {i + 1}: " + ", ".join(f"K={k}: {v:.4f}"
                                                        for k, v in r["ratios"].items()))
    rows = []
    for name in sets:
        for k in COMPONENTS:
            ratios = [results[(name, i)]["ratios"][str(k)] for i in range(len(folds.folds))]
            rows.append({"set": name, "n_features": len(sets[name]), "n_components": k,
                         "objective": float(np.exp(np.mean(np.log(ratios)))),
                         "fold1": ratios[0], "fold2": ratios[1]})
    table_out = pd.DataFrame(rows).sort_values("objective")
    best = table_out.iloc[0]
    chosen = Config(features=sets[best["set"]], n_components=int(best["n_components"]),
                    seed=seed)
    suffix = f"_{tag}" if sample else ""
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    table_out.to_csv(EXPERIMENTS / f"gaussian_process_selection{suffix}.csv", index=False,
                     float_format="%.6g")
    out = {**asdict(chosen), "selection": {
        "set": best["set"], "fit_runs_per_fold": [results[(best["set"], i)]["fit_runs"]
                                                  for i in range(len(folds.folds))],
        "objective": "geometric mean over folds of per-run mean TIMSE / persistence",
        "folds": [[f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for a, b in tl.FOLDS],
        "score": round(float(best["objective"]), 5),
        "fold_ratios": [round(float(best["fold1"]), 5), round(float(best["fold2"]), 5)],
        "explained_fold2": results[(best["set"], 1)]["explained"]}}
    path = CONFIG_PATH.with_name(f"gaussian_process_config{suffix}.json")
    path.write_text(json.dumps(out, indent=2, default=list) + "\n", encoding="utf-8")
    log("\n" + table_out.to_string(index=False, float_format="{:.4f}".format))
    log(f"chosen: {best['set']}, K={int(best['n_components'])}, score {best['objective']:.4f};"
        f" written to {path}")
    report_time(timer, time.perf_counter() - started, sample, sets, folds)
    return chosen


def report_time(timer, elapsed, sample, sets, folds):
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", info.rss) / 1e6
    log(f"\nTiming: {elapsed:.0f} s in all, peak memory {peak:.0f} MB")
    for key in ("fit", "score"):
        if key in timer.total:
            log(f"  {key}: {timer.total[key]:.1f} s total, {timer.mean(key):.1f} s each "
                f"(x{timer.count[key]})")
    if not sample or "ml evals" not in timer.total:
        return
    cfg = Config()
    evals_per_gp = timer.total["ml evals"] / timer.count["ml evals"]
    fits = len(sets) * len(folds.folds)
    gps = fits * cfg.max_components
    rng = np.random.default_rng(0)
    step = {}
    for name, feats in sets.items():  # one ML-II step at the full subsample size
        d = len(feats) + 2
        x = rng.normal(size=(cfg.ml_runs, d))
        t = time.perf_counter()
        neg_log_marginal(np.zeros(d + 2), x, x[:, 0], np.ones(cfg.ml_runs))
        step[name] = time.perf_counter() - t
    ml = sum(step[n] for n in sets) * len(folds.folds) * cfg.max_components * evals_per_gp
    n_full = max(len(f["fit"]) for f in folds.folds)
    t = time.perf_counter()
    x = rng.normal(size=(n_full, 27))
    from models.gaussian_process import GP

    GP(np.ones(27), 1.0, 0.1).condition(x, x[:, 0], np.ones(n_full))
    posterior = (time.perf_counter() - t) * gps
    log(f"  ML-II: {evals_per_gp:.0f} likelihood evaluations per GP (pilot); one evaluation at "
        f"{cfg.ml_runs} runs takes " + ", ".join(f"{k} {v:.2f} s" for k, v in step.items()))
    log(f"  estimate for the full selection ({fits} fits x {cfg.max_components} GPs): "
        f"~{(ml + posterior) / 60:.0f} min (ML-II ~{ml / 60:.0f} min, posteriors on up to "
        f"{n_full} runs ~{posterior / 60:.0f} min); peak memory ~"
        f"{3 * n_full**2 * 8 / 1e6 + 300:.0f} MB")


# ---------------------------------------------------------------------- train / evaluate
def load_config(path=CONFIG_PATH):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d.pop("selection", None)
    d["features"] = tuple(d["features"])
    return Config(**d)


def train(config=None, out=ASSET):
    from models.linear_model import LinearModel

    config = config or load_config()
    series = load_train()
    t0s = tl.training_runs(series)
    model = GaussianProcessModel(replace(config, max_components=config.n_components),
                                 LinearModel.load())
    names = model.input_features
    table = tx.feature_table(t0s, names, CACHE_DIR / "table.pkl")
    x = table[list(names)].to_numpy(dtype=float)
    model.fit(series.to_numpy(dtype=float), series.index.get_indexer(t0s), x, log=log)
    model.fit_info.update({"start": f"{DATA_START:%Y-%m-%dT%H:%MZ}",
                           "end": f"{TRAIN_END:%Y-%m-%dT%H:%MZ}"})
    model.save(out)
    log(f"fitted {config} on {model.fit_info}; wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return model


def evaluate(model=None):
    """Score the trained GP on the eval year once, against the XGBoost hybrid, the
    linear model, the final baseline and persistence."""
    from models.final_baseline import FinalBaseline
    from models.linear_model import LinearModel
    from models.xgboost_model import XGBoostModel
    from training import solar_wind_events as ev
    from training.datasets import EVAL_END, load_hourly
    from training.evaluate import forecast_times, run_timse, score

    model = model or GaussianProcessModel.load()
    hybrid = XGBoostModel.load()
    linear = LinearModel.load()
    series = load_hourly()
    values = series.to_numpy(dtype=float)
    t0s = forecast_times()
    pos = series.index.get_indexer(t0s)
    names = tuple(dict.fromkeys(model.input_features + hybrid.input_features
                                + linear.config.features))
    table = tx.feature_table(t0s, names)
    actual = targets(values, pos)

    def cols(m):
        return table[list(m)].to_numpy()

    forecasts = {
        "gp": model.forecast(values, pos, cols(model.input_features)),
        "xgb_hybrid": hybrid.forecast(values, pos, cols(hybrid.input_features)),
        "linear_model": linear.forecast(values, pos, cols(linear.config.features)),
        "final_baseline": FinalBaseline.load().forecast(values, pos),
        "persistence": np.repeat(last_observed(values, pos)[:, None], actual.shape[1], 1),
    }
    metrics = ["timse", "run_timse_mean", "run_timse_median", "run_timse_p90", "rmse"]
    overall = pd.DataFrame({m: {k: score(f, actual)[k] for k in metrics}
                            for m, f in forecasts.items()}).T
    runs = pd.DataFrame({m: run_timse(f, actual) for m, f in forecasts.items()}, index=t0s)
    minutes = ev.load_conditions(TRAIN_END - pd.Timedelta(days=30), EVAL_END)
    hourly = minutes.resample("h").mean().reindex(series.index)
    events = ev.detect_icmes(hourly, ev.detect_shocks(minutes))
    cme = ev.cme_mask(hourly.index, events)
    possible = ev.cme_mask(hourly.index, events, ("possible ICME",))
    arrivals = ev.detect_stream_arrivals(hourly["speed"], cme | possible)
    regime = pd.Series(ev.classify_runs(t0s, actual, cme, arrivals, possible), index=t0s)
    by_regime = {}
    for r in ev.REGIMES:
        sel = (regime == r).to_numpy() & runs.notna().all(axis=1).to_numpy()
        if not sel.any():
            continue
        gp, lin, base, xg = (runs.loc[sel, c] for c in ("gp", "linear_model", "final_baseline",
                                                        "xgb_hybrid"))
        by_regime[r] = {"runs": int(sel.sum()),
                        "gp / baseline": float((gp / base).median()),
                        "gp / linear": float((gp / lin).median()),
                        "gp beats linear": float((gp < lin).mean()),
                        "gp / hybrid": float((gp / xg).median()),
                        "gp beats hybrid": float((gp < xg).mean())}
    by_regime = pd.DataFrame(by_regime).T
    bias = pd.DataFrame({m: np.nanmean(actual - f, axis=0)[[0, 23, 47, 71]]
                         for m, f in forecasts.items()}, index=["1 h", "24 h", "48 h", "72 h"])
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4g}".format):
        log(f"eval year: {len(t0s)} runs\n\n{overall}\n\nby regime (medians of per-run "
            f"ratios):\n{by_regime}\n\nmean residual (actual - forecast, km/s):\n{bias}")
    return overall, by_regime, bias


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["select", "train", "evaluate"])
    parser.add_argument("--sample", type=int, default=None, help="pilot: fit runs per fold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="clear this run's cache first")
    args = parser.parse_args(argv)
    if args.command == "select":
        select(args.sample, args.seed, args.max_seconds, args.fresh)
    elif args.command == "train":
        train()
    else:
        evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
