"""Do HMM regime features help? Step 1 of the HMM study (models.regime_hmm).

    python -m training.regime_hmm --pilot --folds 2024    # one fold, timed, not cached
    python -m training.regime_hmm [--max-seconds 240]     # every fold; resumable
    python -m training.regime_hmm --summary               # scores from the cached folds

Per fold (training.folds.YEARLY), for each HMM variant (VARIANTS):
    1. fit the HMM on the in-situ hours before the fold start (speed, log T ratio, log Bt);
    2. run its causal filter over the whole series: features at each t0 use the row
       labelled t0 - 1h only (hmm_p_<state>, hmm_state_hours);
    3. score, against the shipped models refitted on the fold (as training.compare_history):
           hybrid + hmm    the shipped hybrid with the HMM features added to its trees
           linear + hmm    the shipped linear model with the HMM features added
           both + hmm      the hybrid with HMM features, anchored on linear + hmm

Score: per-run mean TIMSE relative to persistence on the fold's 6-hourly runs;
summaries are geometric means over folds. Results: src/experiments/regime_hmm.csv.
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
from models.regime_hmm import Config as HMMConfig
from models.regime_hmm import RegimeHMM, observations
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import pull_features as pf
from training.datasets import DATA_START, TRAIN_END
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "regime_hmm_cache"
VARIANTS = {"ar4": HMMConfig(autoregressive=True), "level4": HMMConfig(autoregressive=False)}
INSITU_START = pd.Timestamp("2016-09-14T00:00Z")  # first in-situ hour pulled


def log(msg):
    print(msg, flush=True)


class Data(ch.Data):
    """compare_history.Data plus the in-situ hourly frame the HMM reads."""

    def __init__(self):
        super().__init__()
        hourly = pf.load_hourly()
        self.insitu = hourly[(hourly.index >= INSITU_START) & (hourly.index < TRAIN_END)]

    def frame(self, hmm_features):
        return self.table.join(hmm_features)


def fit_hmm(data, config, start):
    rows = data.insitu[data.insitu.index < start]
    model = RegimeHMM(config).fit(observations(rows))
    return model, model.features(data.insitu, data.hourly)


def _x(frame, t0s, names):
    return frame.loc[t0s, list(names)].to_numpy(dtype=float) if names else None


def score_fold(data, start, end, timer):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos6, pos_h = data.series.index.get_indexer(fit6), data.series.index.get_indexer(fit_h)

    def clock(name, t):
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t

    def linear(cfg, frame):
        m = LinearModel(cfg).fit(data.values, pos6, _x(frame, fit6, cfg.features), end=start_pos)
        f = m.forecast(data.values, fold["pos"], _x(frame, fold["t0s"], cfg.features))
        return m, data.ratio(fold, f)

    def hybrid(cfg, lin, frame):
        m = XGBoostModel(cfg, lin)
        m.fit(data.values, pos_h, _x(frame, fit_h, m.input_features), end=start_pos)
        f = m.forecast(data.values, fold["pos"], _x(frame, fold["t0s"], m.input_features))
        return data.ratio(fold, f)

    out = {}
    t = time.perf_counter()
    lin, out["linear"] = linear(data.linear_cfg, data.table)
    out["hybrid"] = hybrid(data.hybrid_cfg, lin, data.table)
    clock("reference models", t)
    for name, config in VARIANTS.items():
        t = time.perf_counter()
        hmm, feats = fit_hmm(data, config, start)
        clock(f"hmm {name}", t)
        out[f"{name} states"] = hmm.summary().round(3).to_dict()
        frame = data.frame(feats)
        extra = tuple(feats.columns)
        t = time.perf_counter()
        hyb_cfg = replace(data.hybrid_cfg, features=data.hybrid_cfg.features + extra)
        out[f"hybrid + {name}"] = hybrid(hyb_cfg, lin, frame)
        lin_cfg = replace(data.linear_cfg, features=data.linear_cfg.input_features + extra)
        lin_h, out[f"linear + {name}"] = linear(lin_cfg, frame)
        out[f"both + {name}"] = hybrid(hyb_cfg, lin_h, frame)
        clock(f"models {name}", t)
    return out


def scores_table(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("states")}
                       for r in rows])
    return df


def summarize(rows):
    df = scores_table(rows)
    df.to_csv(EXPERIMENTS / "regime_hmm.csv", index=False, float_format="%.6g")
    cols = [c for c in df.columns if c not in ("fold", "seconds")]
    geo = {c: float(np.exp(np.log(df[c]).mean())) for c in cols}
    won = {c: int((df[c] < df["hybrid"]).sum()) for c in cols}
    summary = pd.DataFrame({"fold score": geo,
                            "vs hybrid": {c: geo[c] / geo["hybrid"] - 1 for c in cols},
                            "folds won vs hybrid": won}).sort_values("fold score")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df.drop(columns="seconds", errors="ignore")
            .set_index("fold").T.to_string())
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
            f"{k} {v:.4f}" for k, v in r.items()
            if isinstance(v, float) and k != "seconds"))
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB")
            log(f"  six folds -> ~{6 * r['seconds'] / 60:.0f} min (earlier folds fit on less)")
            for name in VARIANTS:
                log(f"\n  {name} states:\n" + pd.DataFrame(r[f'{name} states']).to_string())
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
