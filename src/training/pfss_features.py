"""PFSS features added to the submitted linear model, official metric, on the yearly folds.

    python -m training.pfss_features --sample 500      # timed pilot (cached separately)
    python -m training.pfss_features [--max-seconds N]  # every training run; resumable
    python -m training.pfss_features --summary

Inputs: the PFSS rings of the GONG synoptic maps (training.pull_gong rings; one map per 12 h,
the newest posted by t0 - 10 min and observed within 72 h), read at the Carrington longitude
that faced Earth L hours before t0 (feature_engineering.windows, PFSS signals).

Variants, each added to the submitted config (training/linear_model_config.json), refitted per
fold:
    single     each ring signal lead-aligned at 72 h ("pfss_<var>_lead_72h": lead k reads the
               longitude that faced Earth about 72 h before t0 + k), and each map scalar
    delay      the two best ring signals at 48, 96 and 120 h
    together   the best two ring signals together at their best delays, and with the best
               scalar
A missing value is filled with the median of the fold's fit rows (training.new_features.Folds),
so every variant fits on the same runs.

Score: official S_final against the official truth, relative to persistence; geometric means
over the six yearly folds. Decision rule, set before running: keep only if more than 1% better
than the submitted config and winning at least 4 of the 6 folds. The eval year is not used.
Results: src/experiments/pfss_features.csv (cache: src/data/pfss_features_cache/).
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
from training import linear_model as tl
from training.datasets import load_train
from training.new_features import Folds

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "pfss_features_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
DELAY = 72
DELAYS = (48, 96, 120)


def log(msg):
    print(msg, flush=True)


def pfss_table(t0s):
    from training import pull_gong

    inputs = build.Inputs(pfss=pull_gong.load_ring_table())
    return windows.window_table(inputs, t0s, windows.pfss_candidates())


def lead(signal, delay=DELAY):
    return f"{signal}_lead_{delay}h"


def scalar(signal):
    return f"{signal}_at_0h"


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
    new = pfss_table(t0s)
    table = tl.candidate_table(t0s).join(new)
    t_table = time.perf_counter() - t
    by_year = new[f"{windows.PFSS_RING[0]}_at_0h"].notna().groupby(new.index.year).mean()
    log(f"{len(t0s)} runs ({tag}); tables {t_table:.0f} s; runs with a PFSS ring by year: "
        + ", ".join(f"{y} {v:.0%}" for y, v in by_year.items()))
    folds = Folds(series, table, series.index.get_indexer(t0s), truth=pull_ace_1h.load(),
                  new=list(new.columns))
    shipped = tl.load_config()
    rows, times = {}, []

    def score(name, features):
        path = cache / f"{name.replace(' ', '').replace('+', '_')}.json"
        if name in rows:
            return True
        if path.exists():
            rows[name] = json.loads(path.read_text(encoding="utf-8"))
            return True
        if max_seconds and time.perf_counter() - started > max_seconds:
            return False
        cfg = replace(shipped, features=(*shipped.features, *features))
        t = time.perf_counter()
        obj, ratios, scores = folds.objective(cfg)
        times.append(time.perf_counter() - t)
        rows[name] = {"variant": name, "objective": obj, "features": list(cfg.features),
                      **{f"ratio {a:%Y}": r for (a, _), r in zip(tl.FOLDS, ratios, strict=True)},
                      **{f"S {a:%Y}": s for (a, _), s in zip(tl.FOLDS, scores, strict=True)}}
        path.write_text(json.dumps(rows[name]), encoding="utf-8")
        log(f"  {name}: {obj:.4f} ({times[-1]:.0f} s)")
        return True

    def stop():
        log("time limit reached; run the same command again to continue")

    if not score("shipped", ()):
        return stop()
    singles = {s: lead(s) for s in windows.PFSS_RING}
    scalars = {s: scalar(s) for s in windows.PFSS_SCALARS}
    for f in (*singles.values(), *scalars.values()):
        if not score(f"+ {f}", (f,)):
            return stop()
    obj = {n: rows[f"+ {f}"]["objective"] for n, f in singles.items()}
    best2 = sorted(obj, key=obj.get)[:2]
    best_delay = {}
    for s in best2:
        for d in DELAYS:
            if not score(f"+ {lead(s, d)}", (lead(s, d),)):
                return stop()
        tried = {d: rows[f"+ {lead(s, d)}"]["objective"] for d in DELAYS}
        tried[DELAY] = obj[s]
        best_delay[s] = min(tried, key=tried.get)
    pair = tuple(lead(s, best_delay[s]) for s in best2)
    if not score(" + ".join(pair), pair):
        return stop()
    best_scalar = min(scalars.values(), key=lambda f: rows[f"+ {f}"]["objective"])
    trio = (*pair, best_scalar)
    if not score(" + ".join(trio), trio):
        return stop()

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
        df.reset_index().to_csv(EXPERIMENTS / "pfss_features.csv", index=False,
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
