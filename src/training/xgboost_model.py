"""Select, train and score the XGBoost model (models.xgboost_model).

    python -m training.xgboost_model select --sample-frac 0.1 --seed 0   # timed pilot
    python -m training.xgboost_model select [--max-seconds 900]          # full selection
    python -m training.xgboost_model train                               # fit the chosen config
    python -m training.xgboost_model evaluate                            # eval year, once

--base linear runs the same for the hybrid: the trees correct the linear model's
forecast instead of persistence. In each fold the linear model (the shipped config,
with the data fixes) is fitted on the fold's 6-hourly fit runs, exactly as in the
linear model's own selection; `train` uses the shipped linear model.

Selection never touches the eval year. It uses the walk-forward folds of the
final baseline (training.final_baseline.FOLDS): fit on the runs before the fold
(targets cut at the fold start), forecast the fold's 6-hourly runs, and score
per-run mean TIMSE relative to persistence, as a geometric mean over the folds
(the same objective as the linear model).

Fitting uses every HOUR of the training years as a forecast time, not only the
6-hourly live schedule, so every lead sees targets at every hour of day; each run
contributes a random subset of Config.leads_per_run leads (memory). Scoring stays
on the 6-hourly runs.

Stages:
  1. candidate regressor sets with default hyper-parameters
     A  the linear model's regressors
     B  one window per family (its best in the linear selection's first step)
     C  B plus coronal-hole strips at several lags and the eastern / disk strips
     D  every windowed candidate plus those strips
  2. max_depth x min_child_weight on the best set; tree counts are scored from one
     fit each (predicting with the first N trees)
  3. pruning: keep the fewest regressors that make up PRUNE_KEEP of the total gain,
     if the fold score doesn't get worse

--sample-frac F fits on a random fraction F of the hourly training runs (scoring
is unchanged) and reports the time and memory each part took, with an estimate
for the full selection. Its outputs carry a _pilot suffix.
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import windows
from feature_engineering.recurrence import last_observed, targets
from models.final_baseline import run_timse_weights
from models.xgboost_model import ASSET, Config, XGBoostModel
from training import linear_model as tl
from training.datasets import DATA_START, TRAIN_END, load_train
from training.final_baseline import FOLDS

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "xgboost_model_config.json"
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "xgboost_cache"
TREE_COUNTS = (5, 10, 15, 25, 50, 100, 200, 400, 800)
GRID = {  # stage 2: max_depth x min_child_weight, per base
    "persistence": ((3, 4, 6), (5.0, 100.0)),
    "linear": ((2, 3, 4), (100.0, 500.0)),
}
PRUNE_KEEP = 0.95  # share of the regressors' total gain the pruned set keeps
EXTRA_SPECS = tuple(
    [windows.WindowSpec("ch_cm", "at", h) for h in (0, 24, 48, 72)]
    + [windows.WindowSpec("ch_east", "at", h) for h in (0, 24)]
    + [windows.WindowSpec("ch_disk", "at", 0)]
)


def log(msg):
    print(msg, flush=True)


def peak_mb():
    import psutil

    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e6


# ---------------------------------------------------------------------- data
def hourly_runs(series, frac=None, seed=0):
    """Every hour of the training years as a forecast time, optionally sampled."""
    t0s = series.index[(series.index >= DATA_START) & (series.index < TRAIN_END)]
    if frac:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(t0s), size=int(round(frac * len(t0s))), replace=False))
        t0s = t0s[keep]
    return t0s


def candidate_sets():
    """{name: tuple of regressor names} for stage 1."""
    linear = tuple(tl.load_config().features)
    step1 = pd.read_csv(EXPERIMENTS / "linear_model_selection.csv")
    step1 = step1[step1["step"] == 1]
    best = step1.loc[step1.groupby("family")["objective"].idxmin()].sort_values("objective")
    one_per_family = tuple(best["candidate"])
    extra = tuple(s.name for s in EXTRA_SPECS)
    every = tuple(s.name for s in windows.candidates())
    return {
        "A_linear": linear,
        "B_one_per_family": one_per_family,
        "C_family_plus_ch": tuple(dict.fromkeys(one_per_family + extra)),
        "D_all": tuple(dict.fromkeys(every + extra)),
    }


def feature_table(t0s, names, cache=None):
    """The regressors `names` at each t0 (pickled in `cache` if given)."""
    if cache is not None and cache.exists():
        table = pd.read_pickle(cache)
        if table.index.equals(t0s) and set(names) <= set(table.columns):
            return table
    from feature_engineering import build
    from training import pull_features as pf

    inputs = build.Inputs(hourly=pf.load_hourly(), shocks=pf.load_shocks(),
                          coronal_holes=pf.load_coronal_holes(), flares=pf.load_flares(),
                          f107=pf.load_f107())
    specs = [windows.WindowSpec.parse(n) for n in names]
    table = windows.window_table(inputs, t0s, specs).astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_pickle(cache)
    return table


class Folds:
    """Fit on hourly runs before each fold; score the fold's 6-hourly runs."""

    def __init__(self, series, table, fit_t0s):
        self.series = series
        self._linear = {}
        self.values = series.to_numpy(dtype=float)
        self.index = series.index
        self.table = table
        val_t0s = tl.training_runs(series)
        self.val_pos = self.index.get_indexer(val_t0s)
        self.val_t0s = val_t0s
        self.fit_t0s = fit_t0s
        self.fit_pos = self.index.get_indexer(fit_t0s)
        self.folds = []
        for start, end in FOLDS:
            fit = np.flatnonzero(fit_t0s < start)
            val = np.flatnonzero((val_t0s >= start) & (val_t0s < end))
            y = targets(self.values, self.val_pos[val])
            q = run_timse_weights(y)
            last = last_observed(self.values, self.val_pos[val])
            scored = (np.isfinite(y).sum(axis=1) >= tl.MIN_RUN_POINTS) & np.isfinite(last)
            persistence = tl.run_scores(np.repeat(last[:, None], y.shape[1], 1), y, q)
            self.folds.append({"start": self.index.get_loc(start), "fit": fit, "val": val,
                               "y": y, "q": q, "scored": scored,
                               "ref": np.nanmean(persistence[scored])})

    def x(self, t0s, features):
        if not features:
            return None
        return self.table.loc[t0s, list(features)].to_numpy(dtype=np.float32)

    def linear(self, i):
        """The linear model for fold i, fitted on the fold's 6-hourly fit runs."""
        if i not in self._linear:
            from models.linear_model import LinearModel

            f = self.folds[i]
            cfg = tl.with_fixes(tl.load_config())
            fit = np.flatnonzero(self.val_t0s < self.index[f["start"]])
            self._linear[i] = LinearModel(cfg).fit(
                self.values, self.val_pos[fit], self.x(self.val_t0s[fit], cfg.features),
                end=f["start"])
        return self._linear[i]

    def objective(self, config, tree_counts=TREE_COUNTS, timer=None):
        """{n_trees: (geometric-mean score, [fold ratios])} and the gain per feature."""
        ratios = {n: [] for n in tree_counts}
        gain = pd.Series(0.0, index=list(config.columns))
        for i, f in enumerate(self.folds):
            t = time.perf_counter()
            base = self.linear(i) if config.base == "linear" else None
            model = XGBoostModel(replace(config, n_trees=max(tree_counts)), base)
            model.fit(self.values, self.fit_pos[f["fit"]],
                      self.x(self.fit_t0s[f["fit"]], model.input_features), end=f["start"])
            if timer is not None:
                timer.add("fit", time.perf_counter() - t)
                timer.peak = max(timer.peak, peak_mb())
            t = time.perf_counter()
            x_val = self.x(self.val_t0s[f["val"]], model.input_features)
            for n in tree_counts:
                forecast = model.forecast(self.values, self.val_pos[f["val"]], x_val, n_trees=n)
                score = tl.run_scores(forecast, f["y"], f["q"])
                ratios[n].append(np.nanmean(score[f["scored"]]) / f["ref"])
            if timer is not None:
                timer.add("score", time.perf_counter() - t)
            g = model.booster.get_score(importance_type="total_gain")
            for k, v in g.items():
                gain.iloc[int(k[1:])] += v
        result = {n: (float(np.exp(np.mean(np.log(r)))), r) for n, r in ratios.items()}
        return result, gain / max(gain.sum(), 1e-12)


class Timer(tl.Timer):
    def __init__(self):
        super().__init__()
        self.peak = 0.0


def _key(config):
    return hashlib.sha1(json.dumps(asdict(config), sort_keys=True, default=list)
                        .encode()).hexdigest()[:12]


def _scored(folds, config, label, cache, timer):
    """Rows (one per tree count) for `config`, cached by config so a re-run resumes."""
    path = cache / f"{label.replace('|', '__')}_{_key(config)}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["rows"], pd.Series(d["gain"])
    result, gain = folds.objective(config, timer=timer)
    rows = [{"label": label, "set": label.split("|")[0], "max_depth": config.max_depth,
             "min_child_weight": config.min_child_weight, "n_trees": n,
             "n_features": len(config.features), "objective": obj, "fold1": r[0], "fold2": r[1]}
            for n, (obj, r) in result.items()]
    path.write_text(json.dumps({"rows": rows, "gain": gain.to_dict()}), encoding="utf-8")
    best = min(rows, key=lambda r: r["objective"])
    log(f"  {label}: best {best['objective']:.4f} at {best['n_trees']} trees "
        f"(folds {best['fold1']:.4f} / {best['fold2']:.4f})")
    return rows, gain


def _name(base):
    return "xgboost" if base == "persistence" else "xgboost_hybrid"


def select(sample_frac=None, seed=0, max_seconds=None, fresh=False, base="persistence"):
    started = time.perf_counter()
    timer = Timer()
    depths, min_child = GRID[base]
    tag = (f"pilot{sample_frac:g}_seed{seed}" if sample_frac else "full")
    tag_cache = tag if base == "persistence" else f"{tag}_{base}"
    cache = CACHE_DIR / tag_cache
    if fresh and cache.exists():
        for f in cache.glob("*.json"):
            f.unlink()
    cache.mkdir(parents=True, exist_ok=True)

    series = load_train()
    fit_t0s = hourly_runs(series, sample_frac, seed)
    sets = candidate_sets()
    names = tuple(dict.fromkeys([*(n for s in sets.values() for n in s),
                                 *tl.load_config().features]))
    t0s = fit_t0s.union(tl.training_runs(series))
    t = time.perf_counter()
    table = feature_table(t0s, names, CACHE_DIR / f"table_{tag}.pkl")
    timer.add("feature table", time.perf_counter() - t)
    folds = Folds(series, table, fit_t0s)
    log(f"{len(fit_t0s)} hourly fit runs ({tag}); validation: "
        + ", ".join(f"{len(f['val'])} runs" for f in folds.folds)
        + f"; sets: {', '.join(f'{k} ({len(v)})' for k, v in sets.items())}")

    def out_of_time():
        return max_seconds and time.perf_counter() - started > max_seconds

    rows = []
    base_cfg = Config(base=base)
    log(f"stage 1: candidate sets (base: {base})")
    gains = {}
    for name, feats in sets.items():
        if out_of_time():
            return _stop()
        r, gains[name] = _scored(folds, replace(base_cfg, features=feats), name, cache, timer)
        rows += r
    best = min(rows, key=lambda r: r["objective"])
    best_set = best["set"]
    log(f"stage 2: hyper-parameters on {best_set}")
    for depth in depths:
        for mcw in min_child:
            cfg = replace(base_cfg, features=sets[best_set], max_depth=depth,
                          min_child_weight=mcw)
            if out_of_time():
                return _stop()
            r, gains[(depth, mcw)] = _scored(folds, cfg, f"{best_set}|d{depth}|m{mcw:g}",
                                             cache, timer)
            rows += r
    stage2 = [r for r in rows if r["label"] == best_set or r["label"].startswith(f"{best_set}|")]
    best = min(stage2, key=lambda r: r["objective"])
    chosen = replace(base_cfg, features=sets[best_set], max_depth=best["max_depth"],
                     min_child_weight=best["min_child_weight"], n_trees=best["n_trees"])
    gain = gains[best_set] if best["label"] == best_set else gains[(best["max_depth"],
                                                                    best["min_child_weight"])]
    reg_gain = gain.reindex(list(chosen.features)).fillna(0.0).sort_values(ascending=False)
    cum = reg_gain.cumsum() / max(reg_gain.sum(), 1e-12)
    keep = set(cum.index[: int(np.searchsorted(cum.to_numpy(), PRUNE_KEEP)) + 1])
    weak = [f for f in chosen.features if f not in keep]
    if weak:
        log(f"stage 3: pruning to the {len(keep)} regressors with {PRUNE_KEEP:.0%} of the gain")
        if out_of_time():
            return _stop()
        pruned = replace(chosen, features=tuple(f for f in chosen.features if f not in weak))
        r, _ = _scored(folds, pruned, f"{best_set}|pruned", cache, timer)
        rows += r
        alt = min(r, key=lambda x: x["objective"])
        if alt["objective"] <= best["objective"]:
            chosen, best = replace(pruned, n_trees=alt["n_trees"]), alt

    elapsed = time.perf_counter() - started
    suffix = f"_{tag}" if sample_frac else ""
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EXPERIMENTS / f"{_name(base)}_selection{suffix}.csv",
                              index=False, float_format="%.6g")
    out = {**asdict(chosen), "selection": {
        "fit_runs": len(fit_t0s), "sample_frac": sample_frac, "seed": seed,
        "objective": "geometric mean over folds of per-run mean TIMSE / persistence",
        "folds": [[f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for a, b in FOLDS],
        "score": round(best["objective"], 5),
        "fold_ratios": [round(best["fold1"], 5), round(best["fold2"], 5)],
        "gain_share": {k: round(float(v), 4) for k, v in gain.sort_values(ascending=False)
                       .items()}}}
    path = config_path(base, suffix)
    path.write_text(json.dumps(out, indent=2, default=list) + "\n", encoding="utf-8")
    log(f"\nchosen: {chosen}\nscore {best['objective']:.4f}; written to {path}")
    report_time(timer, elapsed, sample_frac, sets, series, len(depths) * len(min_child))
    return chosen


def _stop():
    log("time limit reached; run the same command again to continue")
    return None


def report_time(timer, elapsed, frac, sets, series, grid):
    log(f"\nTiming: {elapsed:.0f} s in all, peak memory {timer.peak:.0f} MB")
    for key in timer.total:
        log(f"  {key}: {timer.total[key]:.1f} s total, {timer.mean(key):.1f} s each "
            f"(x{timer.count[key]})")
    if frac:
        configs = len(sets) + grid + 1
        fit = timer.mean("fit") / frac  # fitting time grows with rows
        est = (timer.total["feature table"] / frac + configs * 2 * fit
               + configs * 2 * timer.mean("score"))
        runs = len(hourly_runs(series))
        cols = max(len(v) for v in sets.values()) + 4
        rows = runs * Config().leads_per_run
        data_mb = rows * cols * (4 + 1) / 1e6 + runs * cols * 4 / 1e6  # rows + bins + table
        log(f"  estimate for the full selection ({configs} configs x 2 folds): "
            f"~{est / 60:.0f} min (one fit on all runs ~{fit / 60:.1f} min)")
        log(f"  memory for the largest set on all {runs} runs: ~{data_mb:.0f} MB of data "
            f"on top of the process (pilot peak {timer.peak:.0f} MB)")


# ---------------------------------------------------------------------- taming thin rules
MIN_LEAF_DAYS = 30  # a leaf reached on fewer distinct training days rests on one episode
FLARE_INPUTS = ("m_flares_count_120h", "xray_peak_max_120h")


def leaf_support(model, values, positions, regressors, t0s):
    """For every tree leaf: the distinct training days whose runs reach it.

    Rows use lead 36; the hybrid's trees don't split on the lead, so every lead
    reaches the same leaves.
    """
    anchor = model.anchor(values, positions, regressors)[:, [35]]
    rows, _ = model.rows(values, positions, regressors, leads=np.array([36]), anchor=anchor)
    leaves = model.trees.apply(rows)
    days = pd.DatetimeIndex(t0s).floor("D").asi8
    out = []
    for t in range(leaves.shape[1]):
        for leaf in np.unique(leaves[:, t]):
            sel = leaves[:, t] == leaf
            out.append({"tree": t, "leaf": int(leaf), "runs": int(sel.sum()),
                        "days": len(np.unique(days[sel])),
                        "value": float(model.trees.value[leaf])})
    return pd.DataFrame(out)


def tame():
    """Variants of the chosen hybrid that remove rules resting on one flare episode.

    Each variant is scored on the training folds (as in selection) and refitted on all
    training years to measure the support of every leaf. The folds can't test the
    flare-count rule out of sample (fold 1's fit never sees >= 34 flares; fold 2's
    validation never reaches the rule's threshold), so leaf support is the second
    criterion: the chosen variant has no leaf under MIN_LEAF_DAYS days.
    """
    from models.linear_model import LinearModel

    shipped = load_config(config_path("linear"))
    series = load_train()
    fit_t0s = hourly_runs(series)
    sets = candidate_sets()
    names = tuple(dict.fromkeys([*(n for v in sets.values() for n in v),
                                 *tl.load_config().features]))
    table = feature_table(fit_t0s.union(tl.training_runs(series)), names,
                          CACHE_DIR / "table_full.pkl")
    folds = Folds(series, table, fit_t0s)
    cache = CACHE_DIR / "tame"
    cache.mkdir(parents=True, exist_ok=True)
    timer = Timer()

    def drop(cfg, gone):
        return replace(cfg, features=tuple(f for f in cfg.features if f not in gone))

    variants = {"shipped": shipped}
    for mcw in (1000.0, 3000.0, 5000.0, 10000.0, 20000.0, 40000.0):
        variants[f"min_child_weight {mcw:g}"] = replace(shipped, min_child_weight=mcw)
    variants["min_child_weight 10000, depth 3"] = replace(shipped, min_child_weight=10000.0,
                                                          max_depth=3)
    for mcw in (100.0, 1000.0, 3000.0):
        variants[f"no flare count, mcw {mcw:g}"] = replace(
            drop(shipped, FLARE_INPUTS[:1]), min_child_weight=mcw)
    variants["no flare inputs, mcw 3000"] = replace(drop(shipped, FLARE_INPUTS),
                                                    min_child_weight=3000.0)
    values = series.to_numpy(dtype=float)
    positions = series.index.get_indexer(fit_t0s)
    rows = []
    for label, cfg in variants.items():
        scored, _ = _scored(folds, cfg, f"tame {label}", cache, timer)
        best = min(scored, key=lambda r: r["objective"])
        model = XGBoostModel(replace(cfg, n_trees=best["n_trees"]), LinearModel.load())
        x = table.loc[fit_t0s, list(model.input_features)].to_numpy(dtype=np.float32)
        model.fit(values, positions, x)
        support = leaf_support(model, values, positions, x, fit_t0s)
        thin = support[support["days"] < MIN_LEAF_DAYS]
        anchor = model.anchor(values, positions, x)[:, [35]]
        r36, _ = model.rows(values, positions, x, leads=np.array([36]), anchor=anchor)
        leaves = model.trees.apply(r36)
        is_thin = np.isin(leaves, thin["leaf"].to_numpy())
        thin_corr = np.where(is_thin, model.trees.value[leaves], 0.0).sum(axis=1)
        cols = model.config.columns
        inner = model.trees.left >= 0
        split_on = {cols[f] for f in model.trees.feature[inner]}
        rows.append({"variant": label, "n_features": len(cfg.features),
                     "min_child_weight": cfg.min_child_weight, "n_trees": best["n_trees"],
                     "objective": best["objective"], "fold1": best["fold1"],
                     "fold2": best["fold2"], "min_leaf_days": int(support["days"].min()),
                     "thin_leaves": int(len(thin)),
                     "max_thin_correction": float(np.abs(thin_corr).max()),
                     "runs_thin_over_20": int((np.abs(thin_corr) > 20).sum()),
                     "splits_on_flares": ", ".join(sorted(split_on & set(FLARE_INPUTS)))})
        log(f"  {label}: {best['objective']:.4f} at {best['n_trees']} trees; smallest leaf "
            f"{rows[-1]['min_leaf_days']} days; thin leaves move a run by up to "
            f"{rows[-1]['max_thin_correction']:.0f} km/s")
    out = pd.DataFrame(rows)
    out.to_csv(EXPERIMENTS / "xgboost_hybrid_taming.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200):
        log("\n" + out.to_string(index=False, float_format="{:.4f}".format))
    return out


# ---------------------------------------------------------------------- train / evaluate
def config_path(base="persistence", suffix=""):
    stem = "xgboost_model_config" if base == "persistence" else "xgboost_hybrid_config"
    return HERE / f"{stem}{suffix}.json"


def load_config(path=CONFIG_PATH):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d.pop("selection", None)
    d["features"] = tuple(d["features"])
    d["lags_days"] = tuple(d["lags_days"])
    return Config(**d)


def train(config=None, out=ASSET, base="persistence"):
    from models.linear_model import LinearModel

    config = config or load_config(config_path(base))
    series = load_train()
    t0s = hourly_runs(series)
    model = XGBoostModel(config, LinearModel.load() if config.base == "linear" else None)
    names = model.input_features
    table = feature_table(t0s, names, CACHE_DIR / "table_train.pkl")
    x = table[list(names)].to_numpy(dtype=np.float32) if names else None
    model.fit(series.to_numpy(dtype=float), series.index.get_indexer(t0s), x)
    model.fit_info.update({"start": f"{DATA_START:%Y-%m-%dT%H:%MZ}",
                           "end": f"{TRAIN_END:%Y-%m-%dT%H:%MZ}"})
    model.booster = None
    model.save(out)
    log(f"fitted {config} on {model.fit_info}; wrote {out} "
        f"({out.stat().st_size / 1e6:.1f} MB)")
    return model


def evaluate(model=None):
    """Score the trained model on the eval year once: against the linear model, the
    final baseline and persistence, overall, by regime and by lead."""
    from models.final_baseline import FinalBaseline
    from models.linear_model import LinearModel
    from training import solar_wind_events as ev
    from training.datasets import EVAL_END, load_hourly
    from training.evaluate import forecast_times, run_timse, score

    model = model or XGBoostModel.load()
    linear = LinearModel.load()
    series = load_hourly()
    values = series.to_numpy(dtype=float)
    t0s = forecast_times()
    pos = series.index.get_indexer(t0s)
    names = tuple(dict.fromkeys(model.input_features + linear.config.features))
    table = feature_table(t0s, names)
    actual = targets(values, pos)
    forecasts = {
        "xgboost": model.forecast(values, pos, table[list(model.input_features)].to_numpy()
                                  if model.input_features else None),
        "linear_model": linear.forecast(values, pos,
                                        table[list(linear.config.features)].to_numpy()),
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
        xg, lin, base = (runs.loc[sel, c] for c in ("xgboost", "linear_model", "final_baseline"))
        by_regime[r] = {"runs": int(sel.sum()),
                        "xgb / baseline": float((xg / base).median()),
                        "xgb beats baseline": float((xg < base).mean()),
                        "xgb / linear": float((xg / lin).median()),
                        "xgb beats linear": float((xg < lin).mean()),
                        "linear / baseline": float((lin / base).median())}
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
    parser.add_argument("command", choices=["select", "train", "evaluate", "tame"])
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="pilot: fraction of hourly fit runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--fresh", action="store_true", help="clear this run's cache first")
    parser.add_argument("--base", choices=["persistence", "linear"], default="persistence",
                        help="what the trees correct (linear = the hybrid)")
    args = parser.parse_args(argv)
    if args.command == "select":
        select(args.sample_frac, args.seed, args.max_seconds, args.fresh, args.base)
    elif args.command == "train":
        train(base=args.base)
    elif args.command == "tame":
        tame()
    else:
        evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
