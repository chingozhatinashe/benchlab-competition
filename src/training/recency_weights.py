"""Recency sample weights for the linear model and the hybrid, chosen on the yearly folds.

    python -m training.recency_weights --pilot --folds 2024   # one fold, timed, not cached
    python -m training.recency_weights [--max-seconds 240]    # every fold; resumable
    python -m training.recency_weights --summary

The input-speed relationships change with the solar-cycle phase (Collin et al. 2025), and
the older years didn't help at solar maximum (docs/evaluation.md). Recency weights let the
fit lean on the recent years without dropping the old ones: each training run is weighted
by 0.5 ** (age / half_life), age = years from the run to the end of the training data (the
fold start), on top of the TIMSE weights, normalised to mean 1. half_life = inf is the
model as shipped.

Per fold (training.folds.YEARLY), the shipped linear config (with its data fixes) and the
shipped hybrid's trees on top are refitted on the data before the fold start, both with the
same half-life. The hybrid is also scored on the unweighted linear model ("hybrid trees
only"), to see where the effect comes from. Score: per-run mean TIMSE / persistence;
summaries are geometric means over folds. Results: src/experiments/recency_weights.csv.
Motivation: docs/model_review.md, item 2e.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "recency_weights_cache"
HALF_LIVES = (math.inf, 8.0, 4.0, 2.0, 1.0)  # years
YEAR = pd.Timedelta(days=365.25)


def log(msg):
    print(msg, flush=True)


def label(h):
    return "inf" if math.isinf(h) else f"{h:g}"


def recency_weights(t0s, end, half_life):
    """0.5 ** (age / half_life) per run, age in years before `end`, normalised to mean 1."""
    if math.isinf(half_life):
        return np.ones(len(t0s))
    age = ((end - t0s) / YEAR).to_numpy(dtype=float)
    w = 0.5 ** (age / half_life)
    return w / w.mean()


def effective_years(w, span_years):
    """Kish effective sample size of the weights, in years of data."""
    return float(span_years * w.sum() ** 2 / (len(w) * (w**2).sum()))


def score_fold(data, start, end, timer):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos6, pos_h = data.series.index.get_indexer(fit6), data.series.index.get_indexer(fit_h)
    x6 = data.x(fit6, data.linear_cfg.input_features)
    span = (start - DATA_START) / YEAR
    out, base_lin = {}, None
    for h in HALF_LIVES:
        t = time.perf_counter()
        w6 = recency_weights(fit6, start, h)
        wh = recency_weights(fit_h, start, h)
        lin = LinearModel(data.linear_cfg).fit(data.values, pos6, x6, end=start_pos,
                                               run_weights=w6)
        base_lin = base_lin or lin
        out[f"linear h={label(h)}"] = data.ratio(fold, lin.forecast(
            data.values, fold["pos"], data.x(fold["t0s"], data.linear_cfg.input_features)))
        for name, anchor in (("hybrid", lin), ("hybrid trees only", base_lin)):
            if name == "hybrid trees only" and math.isinf(h):
                continue
            hyb = XGBoostModel(data.hybrid_cfg, anchor)
            hyb.fit(data.values, pos_h, data.x(fit_h, hyb.input_features), end=start_pos,
                    run_weights=wh)
            out[f"{name} h={label(h)}"] = data.ratio(fold, hyb.forecast(
                data.values, fold["pos"], data.x(fold["t0s"], hyb.input_features)))
        out[f"effective years h={label(h)}"] = effective_years(w6, span)
        timer[f"h={label(h)}"] = timer.get(f"h={label(h)}", 0.0) + time.perf_counter() - t
    return out


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "recency_weights.csv", index=False,
                            float_format="%.6g")
    scores = [c for c in df.columns if c.startswith(("linear", "hybrid"))]
    geo = {c: float(np.exp(np.log(df[c]).mean())) for c in scores}
    table = {}
    for h in HALF_LIVES:
        r = {}
        for m, ref in (("linear", "linear"), ("hybrid", "hybrid"),
                       ("hybrid trees only", "hybrid")):
            c, c0 = f"{m} h={label(h)}", f"{ref} h=inf"
            if c not in geo:
                continue
            r[m] = geo[c]
            r[f"{m} vs inf"] = geo[c] / geo[c0] - 1
            r[f"{m} folds won"] = int((df[c] < df[c0]).sum())
        r["effective years (mean)"] = float(df[f"effective years h={label(h)}"].mean())
        table[label(h)] = r
    table = pd.DataFrame(table)
    table.columns.name = "half-life (years)"
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df[scores].T.to_string())
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
            f"{k} {v:.4f}" for k, v in r.items() if k.startswith(("linear", "hybrid"))))
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB")
            log("  effective years: " + ", ".join(
                f"{k[16:]} {v:.1f}" for k, v in r.items() if k.startswith("effective")))
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
