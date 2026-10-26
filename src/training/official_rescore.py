"""Re-score the shipped models with the official metric, on the yearly folds.

    python -m training.official_rescore --pilot --folds 2024   # one fold, timed, not cached
    python -m training.official_rescore [--max-seconds 480]    # every fold; resumable
    python -m training.official_rescore --summary

The models were chosen on per-run TIMSE / persistence (docs/timse.md). On 2026-10-25 the
organisers released the official metric (docs/timse_implementation.pdf,
training.competition_score): per lead, RMSE and a percentile TIMSE combined as
S_tau = sqrt(RMSE_tau * sqrt(TIMSE_tau)), averaged over the 72 leads, against SWPC's hourly
ACE file with status != 0 as missing (training.pull_ace_1h). This study asks whether the
choices made on the old metric still hold. Nothing is re-tuned here.

Per fold (training.folds.YEARLY, fitting on 2016-10 .. fold start, as compare_history):
    models      the final baseline, the linear model and the XGBoost hybrid, fitted exactly
                as in training.compare_history (their inputs are unchanged: the 1-minute
                training series, as live)
    baselines   the spec's: persistence (the latest valid hour available at t0, i.e.
                labelled <= t0 - 1 h; also "persistence (t0)", labelled <= t0, the spec's
                literal reading), 27-day recurrence (persistence where missing),
                climatology (the truth's mean before the fold start; provisional: the
                organisers haven't released v_clim)
    truths      "official"     the spec's truth (status 0)
                "any status"   the same file, every non-fill speed (sensitivity)
                "training"     the 1-minute series the models were trained on (sensitivity)
Scores: S_final, mean RMSE, mean TRMSE, bias, S_49:72 per fold and truth, and the old
per-run TIMSE ratio for the models. The six-fold summary is the geometric mean of
S_final / persistence's (the official truth), the rule used for model choices:
more than 1% better and winning at least 4 of 6 folds.
Results: src/experiments/official_rescore.csv (+ lead curves in official_rescore_curves.json).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.final_baseline import FinalBaseline
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import competition_score as cs
from training import final_baseline as tf
from training import pull_ace_1h
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "official_rescore_cache"
MODELS = ("final baseline", "linear", "hybrid")
BASELINES = ("persistence", "persistence (t0)", "recurrence 27 d", "climatology")
TRUTHS = ("official", "any status", "training")
STATS = ("s_final", "rmse", "trmse", "bias", "s_49_72")


def log(msg):
    print(msg, flush=True)


def truths(data):
    return {"official": pull_ace_1h.load(),
            "any status": pull_ace_1h.load(column="speed_any_status"),
            "training": data.series}


def model_forecasts(data, fold, start, timer):
    """The three models' (runs x 72) forecasts, fitted on [DATA_START, start)."""
    start_pos = data.series.index.get_loc(start)
    out = {}
    t = time.perf_counter()
    base = FinalBaseline(tf.load_config()).fit(data.series.iloc[:start_pos])
    out["final baseline"] = base.forecast(data.values, fold["pos"])
    timer["final baseline"] = timer.get("final baseline", 0) + time.perf_counter() - t
    t = time.perf_counter()
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    lin = LinearModel(data.linear_cfg).fit(
        data.values, data.series.index.get_indexer(fit6),
        data.x(fit6, data.linear_cfg.input_features), end=start_pos)
    out["linear"] = lin.forecast(data.values, fold["pos"],
                                 data.x(fold["t0s"], data.linear_cfg.input_features))
    timer["linear"] = timer.get("linear", 0) + time.perf_counter() - t
    t = time.perf_counter()
    hyb = XGBoostModel(data.hybrid_cfg, lin)
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    hyb.fit(data.values, data.series.index.get_indexer(fit_h),
            data.x(fit_h, hyb.input_features), end=start_pos)
    out["hybrid"] = hyb.forecast(data.values, fold["pos"], data.x(fold["t0s"], hyb.input_features))
    timer["hybrid"] = timer.get("hybrid", 0) + time.perf_counter() - t
    return out


def baseline_forecasts(truth, t0s, start):
    n = len(t0s)
    clim = float(truth[(truth.index >= DATA_START) & (truth.index < start)].mean())
    return {
        "persistence": np.repeat(cs.persistence(truth, t0s)[:, None], cs.HORIZON, 1),
        "persistence (t0)": np.repeat(cs.persistence(truth, t0s, pd.Timedelta(0))[:, None],
                                      cs.HORIZON, 1),
        "recurrence 27 d": cs.recurrence(truth, t0s),  # NaN -> persistence in score()
        "climatology": np.full((n, cs.HORIZON), clim),
    }


def score_fold(data, truth_series, start, end, timer):
    fold = data.fold(start, end)
    t0s = fold["t0s"]
    models = model_forecasts(data, fold, start, timer)
    row, curves = {"fold": f"{start:%Y-%m}", "runs": len(t0s)}, {}
    for name, f in models.items():  # the old metric, for comparison
        row[f"{name} | old ratio"] = data.ratio(fold, f)
    for tname, truth in truth_series.items():
        obs = cs.verifying(truth, t0s)
        pers = cs.persistence(truth, t0s)
        forecasts = {**models, **baseline_forecasts(truth, t0s, start)}
        row[f"{tname} | valid pairs"] = float(np.isfinite(obs).mean())
        for name, f in forecasts.items():
            s = cs.score(f, obs, pers)
            for k in STATS:
                row[f"{tname} | {name} | {k}"] = s[k]
            if tname == "official":
                curves[name] = {k: s["curves"][k] for k in ("s", "rmse", "trmse", "bias")}
    return row, curves


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "official_rescore.csv", index=False,
                            float_format="%.6g")
    names = (*MODELS, *BASELINES)
    out = {}
    for tname in TRUTHS:
        ref = df[f"{tname} | persistence | s_final"]
        for name in names:
            s = df[f"{tname} | {name} | s_final"]
            rel = s / ref
            out[(tname, name)] = {
                "S_final": float(s.mean()),
                "/ persistence": float(np.exp(np.log(rel).mean())),
                "RMSE": float(df[f"{tname} | {name} | rmse"].mean()),
                "TRMSE": float(df[f"{tname} | {name} | trmse"].mean()),
                "bias": float(df[f"{tname} | {name} | bias"].mean()),
                "S_49:72": float(df[f"{tname} | {name} | s_49_72"].mean())}
    table = pd.DataFrame(out).T
    pair = {}
    for a, b in (("hybrid", "linear"), ("hybrid", "final baseline"), ("linear", "final baseline")):
        sa, sb = df[f"official | {a} | s_final"], df[f"official | {b} | s_final"]
        pair[f"{a} vs {b}"] = {"official change": float(np.exp(np.log(sa / sb).mean()) - 1),
                               "folds won": int((sa < sb).sum()),
                               "old-metric change": float(np.exp(np.log(
                                   df[f"{a} | old ratio"] / df[f"{b} | old ratio"]).mean()) - 1)}
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(df)} folds (S in km/s; '/ persistence' is the geometric mean):\n"
            + table.to_string())
        log("\npairs on the official truth:\n" + pd.DataFrame(pair).T.to_string())
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = ch.Data()
    truth_series = truths(data)
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows, all_curves = {}, [], {}
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        if path.exists() and not pilot:
            cached = json.loads(path.read_text(encoding="utf-8"))
            rows.append(cached["row"])
            all_curves[cached["row"]["fold"]] = cached["curves"]
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        row, curves = score_fold(data, truth_series, start, end, timer)
        log(f"  fold {row['fold']} ({time.perf_counter() - t:.0f} s): " + ", ".join(
            f"{n} {row[f'official | {n} | s_final']:.1f}" for n in (*MODELS, *BASELINES))
            + " km/s")
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB; six folds -> ~"
                f"{6 * (time.perf_counter() - t) / 60:.0f} min")
            with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
                log(pd.Series(row).to_string())
            return row
        path.write_text(json.dumps({"row": row, "curves": curves}), encoding="utf-8")
        rows.append(row)
        all_curves[row["fold"]] = curves
    (EXPERIMENTS / "official_rescore_curves.json").write_text(
        json.dumps(all_curves), encoding="utf-8")
    return summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summarize([json.loads(p.read_text(encoding="utf-8"))["row"]
                   for p in sorted(CACHE_DIR.glob("*.json"))])
    else:
        run(args.folds, args.pilot, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
