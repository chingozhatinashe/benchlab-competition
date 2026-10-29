"""New regressors added to the submitted linear model, official metric, on the yearly folds.

    python -m training.new_features --sample 500      # timed pilot (cached separately)
    python -m training.new_features [--max-seconds N]  # every training run; resumable
    python -m training.new_features --summary

Candidates (docs/model_review.md, item 4; none is in the linear selection pool):
    recurrence  Bt and log temperature one rotation earlier: the 72 h mean starting 26, 27 or
                28 days before t0 (windows.recurrence_candidates)
    cycle       the 81-day F10.7 mean, and its trend: the 81- or 27-day mean minus the same
                mean 27 days earlier, per day (windows.cycle_candidates)
    protons     GOES >=10 MeV flux: log 6-hour mean, 24- and 72-hour maxima
                (windows.proton_candidates; history: training.pull_protons, missing
                2020-03 .. 2020-10)

Each candidate is added on its own to the submitted config (training/linear_model_config.json:
four regressors, official loss weights, blend 0.25), which is refitted per fold. The best
candidate of each group (by the geometric mean over folds) is then also tried together.
A missing value of a new candidate is filled with the median of the fold's fit rows, so every
variant fits on the same runs (the live model imputes training medians the same way).

Score: official S_final (training.competition_score) against the official truth
(training.pull_ace_1h), relative to persistence; summaries are geometric means over folds.
Decision rule, set before running: keep a candidate only if it is more than 1% better than
the submitted config and wins at least 4 of the 6 folds. The eval year is not used.
Results: src/experiments/new_features.csv (cache: src/data/new_features_cache/).
"""

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build, windows
from training import competition_score as cs
from training import linear_model as tl
from training.datasets import load_train

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "new_features_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
GROUPS = {"recurrence": windows.recurrence_candidates, "cycle": windows.cycle_candidates,
          "protons": windows.proton_candidates}


def log(msg):
    print(msg, flush=True)


def group_specs():
    return {g: [s.name for s in f()] for g, f in GROUPS.items()}


def new_table(t0s):
    """The new candidates at each t0, from the pulled history (training.pull_features)."""
    from training import pull_features as pf
    from training import pull_protons

    inputs = build.Inputs(hourly=pf.load_hourly(), shocks=pf.load_shocks(),
                          coronal_holes=None, flares=None, f107=pf.load_f107(),
                          protons=pull_protons.load())
    specs = [s for f in GROUPS.values() for s in f()]
    return windows.window_table(inputs, t0s, specs)


class Folds(tl.Folds):
    """tl.Folds whose new columns are filled with the fit rows' median, per fold."""

    def __init__(self, *args, new=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.new = set(new)

    def objective(self, config):
        config = tl.with_fixes(config, self.fixes)
        feats = config.input_features
        fill = [i for i, n in enumerate(feats) if n in self.new]
        ratios, scores = [], []
        for f in self.folds:
            x_fit, x_val = self.x(f["fit"], feats), self.x(f["val"], feats)
            if fill:
                med = np.nanmedian(x_fit[:, fill], axis=0)
                for x in (x_fit, x_val):
                    x[:, fill] = np.where(np.isnan(x[:, fill]), med, x[:, fill])
            model = tl.LinearModel(config).fit(self.values, self.positions[f["fit"]], x_fit,
                                               end=f["start"])
            forecast = model.forecast(self.values, self.positions[f["val"]], x_val)
            s = cs.score(forecast, f["obs"], f["pers"])["s_final"]
            scores.append(s)
            ratios.append(s / f["ref"])
        return float(np.exp(np.mean(np.log(ratios)))), ratios, scores


def variants(shipped, groups, best=None):
    """(name, config) pairs: the submitted config, each candidate added, then the combination."""
    out = [("shipped", shipped)]
    out += [(f"+ {n}", replace(shipped, features=(*shipped.features, n)))
            for names in groups.values() for n in names]
    if best:
        out.append((" + ".join(["shipped", *best]),
                    replace(shipped, features=(*shipped.features, *best))))
    return out


def run(sample=None, seed=0, max_seconds=None):
    started = time.perf_counter()
    tracemalloc.start()
    from training import pull_ace_1h

    tag = f"pilot{sample}_seed{seed}" if sample else "full"
    cache = CACHE_DIR / tag
    cache.mkdir(parents=True, exist_ok=True)
    series = load_train()
    t0s = tl.training_runs(series, sample, seed)
    t = time.perf_counter()
    table = tl.candidate_table(t0s).join(new_table(t0s))
    t_table = time.perf_counter() - t
    groups = group_specs()
    new = [n for names in groups.values() for n in names]
    log(f"{len(t0s)} runs ({tag}); tables {t_table:.0f} s; new candidates missing: " + ", ".join(
        f"{n} {table[n].isna().mean():.1%}" for n in new))
    folds = Folds(series, table, series.index.get_indexer(t0s), truth=pull_ace_1h.load(),
                  new=new)
    shipped = tl.load_config()
    rows, times = {}, []

    def score(name, cfg):
        path = cache / f"{name.replace(' ', '').replace('+', '_')}.json"
        if path.exists():
            rows[name] = json.loads(path.read_text(encoding="utf-8"))
            return True
        if max_seconds and time.perf_counter() - started > max_seconds:
            return False
        t = time.perf_counter()
        obj, ratios, scores = folds.objective(cfg)
        times.append(time.perf_counter() - t)
        rows[name] = {"variant": name, "objective": obj, "features": list(cfg.features),
                      **{f"ratio {a:%Y}": r for (a, _), r in zip(tl.FOLDS, ratios, strict=True)},
                      **{f"S {a:%Y}": s for (a, _), s in zip(tl.FOLDS, scores, strict=True)}}
        path.write_text(json.dumps(rows[name]), encoding="utf-8")
        log(f"  {name}: {obj:.4f} ({times[-1]:.0f} s)")
        return True

    for name, cfg in variants(shipped, groups):
        if not score(name, cfg):
            log("time limit reached; run the same command again to continue")
            return None
    best = [min(names, key=lambda n: rows[f"+ {n}"]["objective"]) for names in groups.values()]
    combo = variants(shipped, groups, best)[-1]
    if not score(*combo):
        log("time limit reached; run the same command again to continue")
        return None

    df = summarize(list(rows.values()), sample)
    if times:
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        each = float(np.mean(times))
        log(f"\ntiming ({len(t0s)} runs): {time.perf_counter() - started:.0f} s in all, "
            f"{each:.1f} s per variant, peak traced memory {peak:.0f} MB")
        if sample:
            n_full = len(tl.training_runs(series))
            log(f"  estimate for {n_full} runs, {len(rows)} variants: ~"
                f"{(len(rows) * each * n_full / len(t0s) + t_table * n_full / len(t0s)) / 60:.0f}"
                " min (fits scale about linearly with runs)")
    return df


def summarize(rows, sample=None):
    df = pd.DataFrame(rows).set_index("variant")
    ref = df.loc["shipped"]
    years = [c.split()[1] for c in df.columns if c.startswith("ratio ")]
    out = {}
    for name, r in df.iterrows():
        ratio = np.array([r[f"ratio {y}"] / ref[f"ratio {y}"] for y in years])
        gain = float(np.exp(np.log(ratio).mean()) - 1)
        won = int((ratio < 1).sum())
        out[name] = {"S_final": float(np.mean([r[f"S {y}"] for y in years])),
                     "/ persistence": r["objective"], "vs shipped": gain, "folds won": won,
                     "keep": bool(name != "shipped" and gain < -KEEP_GAIN and won >= KEEP_FOLDS)}
    table = pd.DataFrame(out).T.sort_values("/ persistence")
    if not sample:
        df.reset_index().to_csv(EXPERIMENTS / "new_features.csv", index=False,
                                float_format="%.6g")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(years)} folds (official truth; S_final in km/s):\n" + table.to_string())
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((CACHE_DIR / "full").glob("*.json"))]
        summarize(rows)
    else:
        run(args.sample, args.seed, args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
