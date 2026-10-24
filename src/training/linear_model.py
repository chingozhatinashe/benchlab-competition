"""Select, train and score the linear model (models.linear_model).

    python -m training.linear_model select --sample 500 --seed 0   # timed pilot on 500 runs
    python -m training.linear_model select [--max-seconds 500]      # full selection, resumable
    python -m training.linear_model train                           # fit the chosen config
    python -m training.linear_model evaluate                        # eval year, once, at the end

Selection never touches the eval year. It uses the baseline's two walk-forward folds
(training.folds.SELECTION): fit on the runs before the fold (targets cut at the
fold start), forecast every run in the fold through the live pipeline (missing
regressors imputed with the fit's medians). The objective is per-run mean TIMSE
relative to persistence, as a geometric mean over the folds.

Stage 1 picks the model variant (family, Box-Cox lambda, relative to persistence,
loss weights, ridge) with a fixed shortlist of regressors from the screening.
Stage 2 runs forward selection over every windowed candidate
(feature_engineering.windows), one window per (signal, statistic), for that
variant, and stops when a step gains less than MIN_GAIN. The ridge is then
re-tuned on the selected set.

--sample N draws N runs at random (fixed seed) from the training runs and uses only
those; results go to *_pilot files, with the time each part took and an estimate
for the full run.

Data fixes (with_fixes), applied to every config that is scored or trained:
- despike: isolated one-hour glitches in the training speeds (e.g. 1,087 km/s at
  2022-10-05 02:00) are dropped from the fitting targets and recurrence inputs;
- the coronal-hole strip falls back to an image 6 h, then 12 h older
  (feature_engineering.windows), and is "optional": a run still missing it is
  forecast by a submodel fitted without it, not by imputing the median;
- each regressor is capped at its training [0.5%, 99.5%] quantiles (CAP_QUANTILE,
  chosen on the folds by training.regressor_cap), so an out-of-range input such as
  the 61 nT Bt of 2026-01-20 isn't extrapolated.
"""

import argparse
import itertools
import json
import sys
import time
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build, windows
from feature_engineering.recurrence import last_observed, targets
from models.final_baseline import CYCLE_HOURS, run_timse_weights
from models.linear_model import ASSET, Config, LinearModel
from training.datasets import DATA_START, TRAIN_END, load_train
from training.evaluate import MIN_RUN_POINTS
from training.folds import SELECTION as FOLDS

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "linear_model_config.json"
CAP_QUANTILE = 0.005  # regressor cap, chosen on the folds (training.regressor_cap)
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "linear_model_cache"
MIN_GAIN = 0.005
MAX_FEATURES = 25
SHORTLIST = ("bt_mean_6h", "ch_cm_at_24h", "ch_cm_at_48h", "ch_cm_at_72h", "density_mean_24h",
             "pressure_mean_6h", "icme_mean_24h", "bz_min_6h", "log_t_ratio_mean_6h")
VARIANTS = (
    [dict(family="gaussian", boxcox=lam, relative=rel)
     for lam in (1.0, 0.5, 0.0, -0.5) for rel in (False, True)]
    + [dict(family=f, relative=rel) for f in ("gamma", "inverse_gaussian") for rel in (False, True)]
)
WEIGHTS = ("run_timse", "uniform")
RIDGES = (0.0, 0.1, 1.0, 10.0)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------- data
def candidate_table(t0s, ch_fallback=windows.CH_FALLBACK_HOURS):
    """Every windowed candidate at each t0, from the pulled history (training.pull_features)."""
    from training import pull_features as pf

    inputs = build.Inputs(hourly=pf.load_hourly(), shocks=pf.load_shocks(),
                          coronal_holes=pf.load_coronal_holes(), flares=pf.load_flares(),
                          f107=pf.load_f107())
    return windows.window_table(inputs, t0s, windows.candidates(), ch_fallback=ch_fallback)


def with_fixes(config, on=True):
    """`config` with the data fixes: despiking, coronal-hole regressors optional, the cap."""
    if not on:
        return replace(config, despike=False, optional=(), cap_quantile=0.0)
    optional = tuple(f for f in config.features
                     if windows.WindowSpec.parse(f).signal == "ch_cm")
    return replace(config, despike=True, optional=optional, cap_quantile=CAP_QUANTILE)


class Folds:
    """Everything selection needs: speed, run positions, candidates, fold splits, references."""

    def __init__(self, series, table, positions, fixes=True):
        self.fixes = fixes
        self.values = series.to_numpy(dtype=float)
        self.index = series.index
        self.table = table
        self.positions = np.asarray(positions)
        self.folds = []
        for start, end in FOLDS:
            start_pos = self.index.get_loc(start)
            t = self.index[self.positions]
            fit = np.flatnonzero(t < start)
            val = np.flatnonzero((t >= start) & (t < end))
            y = targets(self.values, self.positions[val])
            q = run_timse_weights(y)
            last = last_observed(self.values, self.positions[val])
            # the same runs for every model: enough targets and a persistence forecast
            scored = (np.isfinite(y).sum(axis=1) >= MIN_RUN_POINTS) & np.isfinite(last)
            persistence = run_scores(np.repeat(last[:, None], y.shape[1], 1), y, q)
            self.folds.append({"start": start_pos, "fit": fit, "val": val, "y": y, "q": q,
                               "scored": scored, "ref": np.nanmean(persistence[scored])})

    def x(self, rows, features):
        if not features:
            return None
        return self.table[list(features)].to_numpy(dtype=float)[rows]

    def objective(self, config):
        """Geometric mean over folds of per-run mean TIMSE / persistence's (lower is better)."""
        config = with_fixes(config, self.fixes)
        ratios = []
        for f in self.folds:
            model = LinearModel(config).fit(self.values, self.positions[f["fit"]],
                                            self.x(f["fit"], config.input_features), end=f["start"])
            forecast = model.forecast(self.values, self.positions[f["val"]],
                                      self.x(f["val"], config.input_features))
            score = run_scores(forecast, f["y"], f["q"])
            ratios.append(np.nanmean(score[f["scored"]]) / f["ref"])
        return float(np.exp(np.mean(np.log(ratios)))), ratios


def run_scores(forecast, y, q):
    """Each run's TIMSE: sum of q * error^2 with q = run_timse_weights(y) (exact)."""
    err = np.where(np.isfinite(y), y - forecast, 0.0)
    return (q * err**2).sum(axis=1)


def training_runs(series, sample=None, seed=0):
    """6-hourly t0s in the training years (as in the live schedule), optionally sampled."""
    t0s = series.index[(series.index.hour % CYCLE_HOURS == 0) & (series.index >= DATA_START)
                       & (series.index < TRAIN_END)]
    if sample:
        rng = np.random.default_rng(seed)
        t0s = pd.DatetimeIndex(np.sort(rng.choice(t0s.to_numpy(), size=sample, replace=False)),
                               tz="UTC")
    return t0s


# ---------------------------------------------------------------------- selection
def stage1(folds, timer, cache=None, max_seconds=None, started=None):
    """The variant grid; each scored variant is cached, so a stopped run resumes.

    Returns None if `max_seconds` runs out before every variant is scored.
    """
    rows = []
    for i, (variant, weights, ridge) in enumerate(
            itertools.product(VARIANTS, WEIGHTS, RIDGES)):
        path = cache / f"stage1_{i:03d}.json" if cache else None
        if path is not None and path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            return None
        cfg = Config(features=SHORTLIST, weights=weights, ridge=ridge, **variant)
        t = time.perf_counter()
        obj, ratios = folds.objective(cfg)
        timer.add(f"stage 1 fit+score ({cfg.family})", time.perf_counter() - t)
        row = {**_describe(cfg), "objective": obj,
               **{f"fold{k + 1}": r for k, r in enumerate(ratios)}}
        rows.append(row)
        if path is not None:
            path.write_text(json.dumps(row, default=float), encoding="utf-8")
    df = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    return df


def stage2(folds, base, timer, cache=None, max_seconds=None, started=None):
    """Forward selection for the variant `base`. Returns (path, every scored candidate)."""
    pool = windows.candidates()
    chosen, path, scored = [], [], []
    current, _ = folds.objective(replace(base, features=()))
    path.append({"step": 0, "added": "(speed terms only)", "objective": current, "gain": np.nan})
    for step in range(1, MAX_FEATURES + 1):
        families = {windows.WindowSpec.parse(n).family for n in chosen}
        options = [s for s in pool if s.name not in chosen and s.family not in families]
        cached = cache / f"step{step:02d}.csv" if cache else None
        if cached is not None and cached.exists():
            step_rows = pd.read_csv(cached).to_dict("records")
        else:
            if max_seconds and time.perf_counter() - started > max_seconds:
                return path, scored, False
            step_rows = []
            t = time.perf_counter()
            for spec in options:
                obj, _ = folds.objective(replace(base, features=(*chosen, spec.name)))
                step_rows.append({"step": step, "candidate": spec.name, "family": spec.family,
                                  "window_h": spec.hours, "objective": obj})
            timer.add("stage 2 step", time.perf_counter() - t, count=1)
            timer.add("stage 2 candidate", time.perf_counter() - t, count=len(options))
            if cached is not None:
                pd.DataFrame(step_rows).to_csv(cached, index=False)
        scored += step_rows
        best = min(step_rows, key=lambda r: r["objective"])
        gain = 1 - best["objective"] / current
        log(f"  step {step}: best {best['candidate']} -> {best['objective']:.4f} "
            f"(gain {gain:.2%}, {len(step_rows)} candidates)")
        if gain < MIN_GAIN:
            break
        chosen.append(best["candidate"])
        current = best["objective"]
        path.append({"step": step, "added": best["candidate"], "objective": current, "gain": gain})
    return path, scored, True


def _describe(cfg):
    d = asdict(cfg)
    d.pop("features")
    d["lags_days"] = "/".join(f"{x:g}" for x in d["lags_days"])
    return d


class Timer:
    def __init__(self):
        self.total, self.count = {}, {}

    def add(self, key, seconds, count=1):
        self.total[key] = self.total.get(key, 0.0) + seconds
        self.count[key] = self.count.get(key, 0) + count

    def mean(self, key):
        return self.total[key] / max(self.count[key], 1)


def select(sample=None, seed=0, max_seconds=None, fresh=False):
    started = time.perf_counter()
    tracemalloc.start()
    timer = Timer()
    tag = f"pilot{sample}_seed{seed}" if sample else "full"
    cache = CACHE_DIR / tag
    if fresh and cache.exists():
        for f in cache.glob("*.csv"):
            f.unlink()
    cache.mkdir(parents=True, exist_ok=True)

    series = load_train()
    t0s = training_runs(series, sample, seed)
    t = time.perf_counter()
    table = candidate_table(t0s)
    timer.add("candidate table", time.perf_counter() - t)
    folds = Folds(series, table, series.index.get_indexer(t0s))
    log(f"{len(t0s)} runs ({tag}); folds: " + ", ".join(
        f"fit {len(f['fit'])} / validate {len(f['val'])}" for f in folds.folds))

    s1_path = cache / "stage1.csv"
    if s1_path.exists():
        s1 = pd.read_csv(s1_path)
    else:
        s1 = stage1(folds, timer, cache, max_seconds, started)
        if s1 is None:
            log("time limit reached in stage 1; run the same command again to continue")
            return None
        s1.to_csv(s1_path, index=False)
    best = s1.iloc[0]
    base = Config(family=best["family"], boxcox=float(best["boxcox"]),
                  relative=bool(best["relative"]), weights=best["weights"],
                  ridge=float(best["ridge"]))
    log("stage 1 (shortlist), best 8 of " + f"{len(s1)}:\n" + s1.head(8)[
        ["family", "boxcox", "relative", "weights", "ridge", "objective",
         *(f"fold{i + 1}" for i in range(len(FOLDS)))]
    ].to_string(index=False, float_format="{:.4f}".format))

    path, scored, done = stage2(folds, base, timer, cache, max_seconds, started)
    if not done:
        log("time limit reached; run the same command again to continue")
        return None
    features = tuple(p["added"] for p in path[1:])
    ridge_rows = []
    for ridge in RIDGES:
        obj, ratios = folds.objective(replace(base, features=features, ridge=ridge))
        ridge_rows.append({"ridge": ridge, "objective": obj})
    best_ridge = min(ridge_rows, key=lambda r: r["objective"])
    config = with_fixes(replace(base, features=features, ridge=best_ridge["ridge"]))
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    elapsed = time.perf_counter() - started

    suffix = f"_{tag}" if sample else ""
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    s1.to_csv(EXPERIMENTS / f"linear_model_variants{suffix}.csv", index=False, float_format="%.6g")
    pd.DataFrame(scored).to_csv(EXPERIMENTS / f"linear_model_selection{suffix}.csv", index=False,
                                float_format="%.6g")
    out = {**_describe(config), "features": list(config.features), "selection": {
        "runs": len(t0s), "sample_seed": seed if sample else None,
        "objective": "geometric mean over folds of per-run mean TIMSE / persistence",
        "folds": [[f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for a, b in FOLDS],
        "path": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()}
                 for p in path],
        "ridge": ridge_rows, "score": round(best_ridge["objective"], 5)}}
    cfg_path = CONFIG_PATH.with_name(f"linear_model_config{suffix}.json")
    cfg_path.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")

    log("\nselection path:\n" + pd.DataFrame(path).to_string(index=False,
                                                              float_format="{:.4f}".format))
    log(f"chosen: {config}\nwritten to {cfg_path}")
    report_time(timer, elapsed, peak, len(t0s), len(path) - 1)
    return config


def report_time(timer, elapsed, peak, runs, steps):
    full_runs = len(training_runs(load_train()))
    scale = full_runs / runs
    log(f"\nTiming ({runs} runs): {elapsed:.0f} s in all, peak traced memory {peak:.0f} MB")
    for key in timer.total:
        log(f"  {key}: {timer.total[key]:.1f} s total, {timer.mean(key):.3f} s each "
            f"(x{timer.count[key]})")
    if runs < full_runs:
        n_s1 = len(VARIANTS) * len(WEIGHTS) * len(RIDGES)
        s1 = sum(v for k, v in timer.total.items() if k.startswith("stage 1")) * scale
        cand = timer.mean("stage 2 candidate") * scale if "stage 2 candidate" in timer.total else 0
        n_cand = len(windows.candidates())
        for full_steps in (steps, 10, 15):
            evals = sum(n_cand - i for i in range(full_steps + 1))
            est = timer.total["candidate table"] * scale + s1 + cand * evals
            log(f"  estimate for {full_runs} runs with {full_steps} selection steps: "
                f"~{est / 60:.0f} min (stage 1: {n_s1} variants ~{s1 / 60:.0f} min; "
                f"stage 2: {evals} fits ~{cand * evals / 60:.0f} min), "
                f"peak memory ~{peak * scale:.0f} MB")


# ---------------------------------------------------------------------- train / evaluate
def load_config(path=CONFIG_PATH):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(features=tuple(d["features"]), family=d["family"], boxcox=float(d["boxcox"]),
                  relative=bool(d["relative"]), weights=d["weights"], ridge=float(d["ridge"]),
                  lags_days=tuple(float(x) for x in d["lags_days"].split("/")),
                  half_width=int(d["half_width"]))


def train(config=None, out=ASSET):
    config = with_fixes(config or load_config())
    series = load_train()
    t0s = training_runs(series)
    table = candidate_table(t0s)
    x = table[list(config.input_features)].to_numpy(dtype=float) if config.features else None
    model = LinearModel(config).fit(series.to_numpy(dtype=float),
                                    series.index.get_indexer(t0s), x)
    model.fit_info.update({"start": f"{DATA_START:%Y-%m-%dT%H:%MZ}",
                           "end": f"{TRAIN_END:%Y-%m-%dT%H:%MZ}"})
    model.save(out)
    log(f"fitted {config} on {model.fit_info}; wrote {out}")
    return model


def evaluate(model=None):
    """Score the trained model on the eval year once, against the final baseline and persistence.

    Every eval t0 goes through the live pipeline: regressors from data available by t0,
    missing ones imputed with the training medians.
    """
    from models.final_baseline import FinalBaseline
    from training import solar_wind_events as ev
    from training.datasets import EVAL_END, load_hourly
    from training.evaluate import forecast_times, run_timse, score

    model = model or LinearModel.load()
    series = load_hourly()
    values = series.to_numpy(dtype=float)
    t0s = forecast_times()
    pos = series.index.get_indexer(t0s)
    table = candidate_table(t0s)
    x = table[list(model.config.input_features)].to_numpy(dtype=float)
    actual = targets(values, pos)
    forecasts = {
        "linear_model": model.forecast(values, pos, x),
        "final_baseline": FinalBaseline.load().forecast(values, pos),
        "persistence": np.repeat(last_observed(values, pos)[:, None], actual.shape[1], 1),
    }
    metrics = ["timse", "run_timse_mean", "run_timse_median", "run_timse_p90", "rmse"]
    overall = pd.DataFrame({m: {k: score(f, actual)[k] for k in metrics}
                            for m, f in forecasts.items()}).T
    imputed = pd.Series(np.isnan(x).mean(axis=0), index=model.config.input_features)
    imputed["(runs forecast by the submodel)"] = model.uses_fallback(x).mean()

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
        lin, base, per = (runs.loc[sel, c] for c in runs)
        by_regime[r] = {"runs": int(sel.sum()),
                        "linear / baseline (median)": float((lin / base).median()),
                        "beats baseline": float((lin < base).mean()),
                        "linear / persistence (median)": float((lin / per).median()),
                        "baseline / persistence (median)": float((base / per).median())}
    by_regime = pd.DataFrame(by_regime).T
    bias = pd.DataFrame({m: np.nanmean(actual - f, axis=0)[[0, 23, 47, 71]]
                         for m, f in forecasts.items()}, index=["1 h", "24 h", "48 h", "72 h"])
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        log(f"eval year: {len(t0s)} runs\n\n{overall}\n\nby regime:\n{by_regime}\n\n"
            f"mean residual (actual - forecast, km/s):\n{bias}\n\n"
            f"share of eval runs with each regressor missing:\n{imputed}")
    return overall, by_regime, bias, imputed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["select", "train", "evaluate"])
    parser.add_argument("--sample", type=int, default=None, help="pilot: number of runs")
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
