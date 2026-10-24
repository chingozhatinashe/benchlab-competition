"""Choose the linear model's regressor cap (Config.cap_quantile) on the yearly folds.

    python -m training.regressor_cap --pilot --folds 2024   # one fold, timed, not cached
    python -m training.regressor_cap [--max-seconds 240]    # every fold; resumable
    python -m training.regressor_cap --summary

Each regressor is clipped to its training [q, 1 - q] quantiles in fitting and
forecasting (models.linear_model). For each q in QUANTILES and each fold
(training.folds.YEARLY), the shipped linear config with that cap is refitted, and
the shipped hybrid's trees are refitted on top of it (as training.compare_history).
q = 0 is the shipped model. Score: per-run mean TIMSE / persistence; summaries are
geometric means over folds. Results: src/experiments/regressor_cap.csv. Motivation:
docs/model_review.md (the 61 nT Bt run of 2026-01-20).
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "regressor_cap_cache"
QUANTILES = (0.0, 0.001, 0.005, 0.01, 0.025)


def log(msg):
    print(msg, flush=True)


def score_fold(data, start, end, timer):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    out, bounds = {}, {}
    for q in QUANTILES:
        t = time.perf_counter()
        cfg = replace(data.linear_cfg, cap_quantile=q)
        lin = LinearModel(cfg).fit(data.values, data.series.index.get_indexer(fit6),
                                   data.x(fit6, cfg.features), end=start_pos)
        out[f"linear q={q:g}"] = data.ratio(fold, lin.forecast(
            data.values, fold["pos"], data.x(fold["t0s"], cfg.features)))
        hyb = XGBoostModel(data.hybrid_cfg, lin)
        hyb.fit(data.values, data.series.index.get_indexer(fit_h),
                data.x(fit_h, hyb.input_features), end=start_pos)
        out[f"hybrid q={q:g}"] = data.ratio(fold, hyb.forecast(
            data.values, fold["pos"], data.x(fold["t0s"], hyb.input_features)))
        if lin.x_low is not None:
            bounds[f"{q:g}"] = {f: [round(float(lo), 3), round(float(hi), 3)] for f, lo, hi
                                in zip(cfg.features, lin.x_low, lin.x_high, strict=True)}
        timer[f"q={q:g}"] = timer.get(f"q={q:g}", 0.0) + time.perf_counter() - t
    out["bounds"] = bounds
    return out


def summarize(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (float, str))}
                       for r in rows]).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "regressor_cap.csv", index=False, float_format="%.6g")
    cols = [c for c in df.columns if c != "seconds"]
    geo = {c: float(np.exp(np.log(df[c]).mean())) for c in cols}
    table = pd.DataFrame({
        q: {"linear": geo[f"linear q={q:g}"], "hybrid": geo[f"hybrid q={q:g}"],
            "linear vs q=0": geo[f"linear q={q:g}"] / geo["linear q=0"] - 1,
            "hybrid vs q=0": geo[f"hybrid q={q:g}"] / geo["hybrid q=0"] - 1,
            "linear folds won": int((df[f"linear q={q:g}"] < df["linear q=0"]).sum()),
            "hybrid folds won": int((df[f"hybrid q={q:g}"] < df["hybrid q=0"]).sum())}
        for q in QUANTILES}).T
    table.index.name = "q"
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df.drop(columns="seconds", errors="ignore").T.to_string())
        log(f"\nover {len(df)} folds:\n{table.to_string()}")
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = ch.Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        if path.exists() and not pilot:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        r = {"fold": f"{start:%Y-%m}", **score_fold(data, start, end, timer)}
        r["seconds"] = time.perf_counter() - t
        log(f"  fold {r['fold']} ({r['seconds']:.0f} s): " + ", ".join(
            f"{k} {v:.4f}" for k, v in r.items() if isinstance(v, float) and k != "seconds"))
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB")
            log(f"  bounds at q=0.005: {r['bounds'].get('0.005')}")
            log(f"  six folds -> ~{6 * r['seconds'] / 60:.0f} min")
            return r
        path.write_text(json.dumps(r), encoding="utf-8")
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
