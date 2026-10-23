"""Does regime switching help on its own? Fold study of models.regime_switching.

    python -m training.regime_switching --pilot --folds 2024   # one fold, timed, not cached
    python -m training.regime_switching [--max-seconds 240]    # every fold; resumable
    python -m training.regime_switching --summary

Per fold (training.folds.YEARLY), for K = 1..5 HMM states (level emissions):
    1. fit the gate's HMM on the in-situ hours before the fold start;
    2. filter the whole in-situ series (each t0 uses the row labelled t0 - 1h);
    3. fit the experts on the 6-hourly runs before the start (targets cut there);
    4. score the fold's 6-hourly runs, soft and hard gating, for two expert input sets:
       "speed" (last speed + recurrence means) and "regs" (+ the linear model's 4
       selected regressors, as inputs only).
K = 1 is the control: one global regression with the same inputs. Also scored:
"hmm alone", the 4-state Markov-switching AR HMM's own multi-step forecast.

Context (not refitted here): the final baseline from src/experiments/history_comparison.csv
(history since 2016) and the shipped linear model and hybrid from
src/experiments/regime_hmm.csv, all on the same folds.

Score: per-run mean TIMSE relative to persistence; summaries are geometric means over
folds. Results: src/experiments/regime_switching.csv; analysis
src/experiments/regime_switching.ipynb; write-up docs/regime_switching.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import last_observed
from models.regime_hmm import Config as HMMConfig
from models.regime_hmm import RegimeHMM, observations
from models.regime_switching import Config, RegimeSwitchingModel
from training import compare_history as ch
from training import pull_features as pf
from training.datasets import DATA_START, TRAIN_END
from training.folds import YEARLY
from training.regime_hmm import INSITU_START

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "regime_switching_cache"
STATES = (1, 2, 3, 4, 5)


def log(msg):
    print(msg, flush=True)


class Data(ch.Data):
    def __init__(self):
        super().__init__()
        hourly = pf.load_hourly()
        self.insitu = hourly[(hourly.index >= INSITU_START) & (hourly.index < TRAIN_END)]
        self.input_sets = {"speed": (), "regs": tuple(self.linear_cfg.features)}


def context():
    """Fold scores of the existing models on the same folds (not refitted here)."""
    out = {}
    path = EXPERIMENTS / "history_comparison.csv"
    if path.exists():
        h = pd.read_csv(path)
        h = h[h["history"] == "since 2016"].set_index("fold")
        out["final baseline"] = h["final_baseline"]
    path = EXPERIMENTS / "regime_hmm.csv"
    if path.exists():
        r = pd.read_csv(path).set_index("fold")
        out["linear"], out["hybrid"] = r["linear"], r["hybrid"]
    return pd.DataFrame(out)


def score_fold(data, start, end, timer):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_pos = data.series.index.get_indexer(fit6)
    before = data.insitu[data.insitu.index < start]

    def clock(name, t):
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t

    out = {"states": {}}
    for k in STATES:
        t = time.perf_counter()
        base = RegimeSwitchingModel(Config(n_states=k)).fit_hmm(before)
        g_fit, g_val = base.gate(data.insitu, fit6), base.gate(data.insitu, fold["t0s"])
        clock("hmm fits", t)
        out["states"][str(k)] = {
            "summary": base.hmm.summary().round(3).to_dict(),
            "val share": g_val.mean(axis=0).round(4).tolist()}
        t = time.perf_counter()
        for name, feats in data.input_sets.items():
            model = RegimeSwitchingModel(Config(n_states=k, features=feats), base.hmm)
            model.fit(data.values, fit_pos, g_fit, data.x(fit6, feats), end=start_pos)
            xv = data.x(fold["t0s"], feats)
            out[f"K={k} soft {name}"] = data.ratio(
                fold, model.forecast(data.values, fold["pos"], g_val, xv))
            if k > 1:
                out[f"K={k} hard {name}"] = data.ratio(
                    fold, model.forecast(data.values, fold["pos"], g_val, xv, hard=True))
        clock("experts", t)
    t = time.perf_counter()
    hmm = RegimeHMM(HMMConfig(n_states=4, autoregressive=True)).fit(observations(before))
    probs = hmm.filter(observations(data.insitu))
    row = data.insitu.index.get_indexer(fold["t0s"] - pd.Timedelta(hours=1))
    p = np.where((row >= 0)[:, None], probs[np.maximum(row, 0)], hmm.start[None, :])
    last = last_observed(data.values, fold["pos"])
    f = np.clip(hmm.mean_forecast(p, last), 200, 1500)
    out["hmm alone"] = data.ratio(fold, f)
    clock("hmm alone", t)
    return out


def scores_frame(rows):
    return pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (float, str))}
                         for r in rows]).set_index("fold")


def summarize(rows):
    df = scores_frame(rows)
    ctx = context()
    df = df.join(ctx, how="left")
    df.reset_index().to_csv(EXPERIMENTS / "regime_switching.csv", index=False,
                            float_format="%.6g")
    cols = [c for c in df.columns if c != "seconds"]
    geo = pd.Series({c: float(np.exp(np.log(df[c].dropna()).mean())) for c in cols})
    rows_out = {}
    for c in cols:
        control = f"K=1 soft {c.split()[-1]}" if c.startswith("K=") else None
        rows_out[c] = {
            "fold score": geo[c],
            "vs K=1 (same inputs)": geo[c] / geo[control] - 1 if control else np.nan,
            "folds won vs K=1": int((df[c] < df[control]).sum()) if control else np.nan,
        }
    summary = pd.DataFrame(rows_out).T.sort_values("fold score")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df.drop(columns="seconds", errors="ignore").T.to_string())
        log(f"\nover {len(df)} folds:\n{summary.to_string()}")
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = Data()
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
