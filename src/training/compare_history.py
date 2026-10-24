"""Does the longer history help? Refit the existing models on two histories, same folds.

    python -m training.compare_history --folds 2024 --pilot   # one fold, timed
    python -m training.compare_history [--max-seconds 480]    # every fold; resumable

Models, with their shipped settings (no re-selection):
    final_baseline  training.final_baseline config
    linear          training.linear_model config with the data fixes
    hybrid          the XGBoost hybrid config, anchored on the fold's linear model

Histories (training.folds.HISTORY_STARTS): fit on hours from 2016-10 or from
2021-10, up to each fold's start. Inputs before the history start stay visible
(as live, where the recent past is always available); only fitting is cut.

Folds: training.folds.YEARLY. "since 2021" needs a year of fitting, so it is scored
on the folds starting 2022-10 onwards; both histories are compared on those
common folds, and "since 2016" also on all six. Score per fold: per-run mean
TIMSE relative to persistence on the fold's 6-hourly runs (the selection
objective); summaries are geometric means over folds. Results:
src/experiments/history_comparison.csv.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import last_observed, targets
from models.final_baseline import FinalBaseline, run_timse_weights
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import final_baseline as tf
from training import linear_model as tl
from training import xgboost_model as tx
from training.datasets import load_train
from training.evaluate import MIN_RUN_POINTS
from training.folds import HISTORY_STARTS, YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "history_cache"
MIN_FIT = pd.Timedelta(days=360)


def log(msg):
    print(msg, flush=True)


class Data:
    """The speed series, run positions and regressor tables, built once."""

    def __init__(self):
        self.series = load_train()
        self.values = self.series.to_numpy(dtype=float)
        self.runs6 = tl.training_runs(self.series)
        self.hourly = tx.hourly_runs(self.series)
        self.linear_cfg = tl.with_fixes(tl.load_config())
        self.hybrid_cfg = tx.load_config(tx.config_path("linear"))
        names = tuple(dict.fromkeys(self.hybrid_cfg.features + self.linear_cfg.input_features))
        self.table = tx.feature_table(self.hourly, names, CACHE_DIR / "table_hourly.pkl")

    def x(self, t0s, names):
        return self.table.loc[t0s, list(names)].to_numpy(dtype=float) if names else None

    def fold(self, start, end):
        val = self.runs6[(self.runs6 >= start) & (self.runs6 < end)]
        pos = self.series.index.get_indexer(val)
        y = targets(self.values, pos)
        q = run_timse_weights(y)
        last = last_observed(self.values, pos)
        scored = (np.isfinite(y).sum(axis=1) >= MIN_RUN_POINTS) & np.isfinite(last)
        ref = tl.run_scores(np.repeat(last[:, None], y.shape[1], 1), y, q)
        return {"t0s": val, "pos": pos, "y": y, "q": q, "scored": scored,
                "ref": np.nanmean(ref[scored])}

    def ratio(self, fold, forecast):
        s = tl.run_scores(forecast, fold["y"], fold["q"])
        return float(np.nanmean(s[fold["scored"]]) / fold["ref"])


def score_fold(data, history, start, end, timer):
    """Ratios to persistence for each model, fitted on [history, start)."""
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    hist_pos = data.series.index.get_loc(history)
    out = {}

    t = time.perf_counter()
    base = FinalBaseline(tf.load_config()).fit(data.series.iloc[hist_pos:start_pos])
    out["final_baseline"] = data.ratio(fold, base.forecast(data.values, fold["pos"]))
    timer["final_baseline"] = timer.get("final_baseline", 0) + time.perf_counter() - t

    t = time.perf_counter()
    fit6 = data.runs6[(data.runs6 >= history) & (data.runs6 < start)]
    lin = LinearModel(data.linear_cfg).fit(
        data.values, data.series.index.get_indexer(fit6),
        data.x(fit6, data.linear_cfg.input_features), end=start_pos)
    xv = data.x(fold["t0s"], data.linear_cfg.input_features)
    out["linear"] = data.ratio(fold, lin.forecast(data.values, fold["pos"], xv))
    timer["linear"] = timer.get("linear", 0) + time.perf_counter() - t

    t = time.perf_counter()
    hyb = XGBoostModel(data.hybrid_cfg, lin)
    fit_h = data.hourly[(data.hourly >= history) & (data.hourly < start)]
    hyb.fit(data.values, data.series.index.get_indexer(fit_h),
            data.x(fit_h, hyb.input_features), end=start_pos)
    xh = data.x(fold["t0s"], hyb.input_features)
    out["hybrid"] = data.ratio(fold, hyb.forecast(data.values, fold["pos"], xh))
    timer["hybrid"] = timer.get("hybrid", 0) + time.perf_counter() - t
    return out


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    folds = [f for f in YEARLY if years is None or f[0].year in years]
    timer, rows = {}, []
    for start, end in folds:
        for label, history in HISTORY_STARTS.items():
            if start - history < MIN_FIT:
                continue
            path = CACHE_DIR / f"{start:%Y}_{label.replace(' ', '')}.json"
            if path.exists() and not pilot:
                r = json.loads(path.read_text(encoding="utf-8"))
            else:
                if max_seconds and time.perf_counter() - started > max_seconds:
                    log("time limit reached; run the same command again to continue")
                    return None
                t = time.perf_counter()
                r = score_fold(data, history, start, end, timer)
                r["seconds"] = time.perf_counter() - t
                if not pilot:
                    path.write_text(json.dumps(r), encoding="utf-8")
                log(f"  fold {start:%Y-%m}, {label}: " + ", ".join(
                    f"{k} {v:.4f}" for k, v in r.items() if k != "seconds")
                    + f" ({r['seconds']:.0f} s)")
            rows.append({"fold": f"{start:%Y-%m}", "history": label, **r})
    df = pd.DataFrame(rows)
    if pilot:
        n = sum(1 for s, _ in YEARLY for h in HISTORY_STARTS.values() if s - h >= MIN_FIT)
        per = df["seconds"].mean()
        log(f"\npilot: {per:.0f} s per (fold, history) on average "
            + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
            + f"; the full comparison has {n} -> ~{n * per / 60:.0f} min")
        return df
    df.to_csv(EXPERIMENTS / "history_comparison.csv", index=False, float_format="%.6g")
    summarize(df)
    return df


def summarize(df):
    models = ["final_baseline", "linear", "hybrid"]
    common = sorted(set(df[df["history"] == "since 2021"]["fold"]))

    def geo(sub):
        return {m: float(np.exp(np.log(sub[m]).mean())) for m in models}

    rows = {}
    for label in HISTORY_STARTS:
        sub = df[(df["history"] == label) & df["fold"].isin(common)]
        rows[(f"common folds ({', '.join(common)})", label)] = geo(sub)
    rows[("all six folds", "since 2016")] = geo(df[df["history"] == "since 2016"])
    old = [f"{s:%Y-%m}" for s, _ in YEARLY[-2:]]
    for label in HISTORY_STARTS:
        sub = df[(df["history"] == label) & df["fold"].isin(old)]
        rows[(f"old two folds ({', '.join(old)})", label)] = geo(sub)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df.drop(columns="seconds").to_string(index=False))
        log("\ngeometric mean of per-run mean TIMSE / persistence (lower is better):\n"
            + pd.DataFrame(rows).T.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="fold start years to run (default all)")
    parser.add_argument("--pilot", action="store_true", help="time without caching")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    run(args.folds, args.pilot, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
