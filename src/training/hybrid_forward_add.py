"""Forward-add test for the hybrid's regressors on the yearly folds.

    python -m training.hybrid_forward_add --pilot --folds 2024   # one fold, timed, not cached
    python -m training.hybrid_forward_add [--max-seconds 240]    # every fold; resumable
    python -m training.hybrid_forward_add --summary

The hybrid's 30 regressors (set C, training.xgboost_model.candidate_sets) keep one window
per family, chosen by the linear model's single-feature ranking, plus extra coronal-hole
lags. Every one of the 24 families is in, but a second window of a family, useful to the
trees only through thresholds or interactions, could have been left out
(docs/model_review.md, section 2).

For each family with unused windows, the unused window most different from the one in
use (`contrast`) is added to the shipped 30, one family at a time, and the trees are
refitted on the shipped linear model (unchanged), with the shipped tree settings. "+ all"
adds all of them at once. Keep rule (set before running): more than 0.5% better than the
shipped hybrid on the six-fold score, winning at least 4 of 6 folds.

Score: per-run mean TIMSE / persistence, geometric mean over folds. Results:
src/experiments/hybrid_forward_add.csv.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import windows
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import xgboost_model as tx
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "hybrid_forward_add_cache"
TABLE = Path(__file__).resolve().parents[1] / "data" / "history_cache" / "table_hourly_all.pkl"
KEEP_GAIN = 0.005
KEEP_FOLDS = 4


def log(msg):
    print(msg, flush=True)


def contrast(used, unused):
    """The unused window furthest (in log hours) from the windows in use."""
    def dist(name):
        h = windows.WindowSpec.parse(name).hours
        return min(abs(np.log(max(h, 1)) - np.log(max(windows.WindowSpec.parse(u).hours, 1)))
                   for u in used)
    return max(unused, key=dist)


def additions(hybrid_features):
    """family -> the window to add, for every family with unused windows (not ch_cm_at)."""
    used, every = defaultdict(list), defaultdict(list)
    for f in hybrid_features:
        used[windows.WindowSpec.parse(f).family].append(f)
    for s in windows.candidates():
        every[s.family].append(s.name)
    out = {}
    for fam, names in every.items():
        unused = [n for n in names if n not in hybrid_features]
        if unused and fam != "ch_cm_at":  # coronal-hole timing is training.lead_aligned_ch
            out[fam] = contrast(used[fam], unused)
    return out


def variants(hybrid_features):
    adds = additions(hybrid_features)
    out = {"shipped": ()}
    out.update({f"+ {name}": (name,) for name in adds.values()})
    out["+ all"] = tuple(adds.values())
    return out


def score_fold(data, table, start, end, timer, names):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos_h = data.series.index.get_indexer(fit_h)

    def x(t0s, cols):
        return table.loc[t0s, list(cols)].to_numpy(dtype=float)

    lin = LinearModel(data.linear_cfg).fit(
        data.values, data.series.index.get_indexer(fit6), x(fit6, data.linear_cfg.input_features),
        end=start_pos)
    out = {}
    for name, extra in variants(data.hybrid_cfg.features).items():
        if name not in names:
            continue
        t = time.perf_counter()
        cfg = replace(data.hybrid_cfg, features=(*data.hybrid_cfg.features, *extra))
        hyb = XGBoostModel(cfg, lin)
        hyb.fit(data.values, pos_h, x(fit_h, hyb.input_features), end=start_pos)
        out[f"hybrid {name}"] = data.ratio(fold, hyb.forecast(
            data.values, fold["pos"], x(fold["t0s"], hyb.input_features)))
        timer["hybrid"] = timer.get("hybrid", 0.0) + time.perf_counter() - t
    return out


def load_table(data):
    names = tuple(dict.fromkeys((*data.hybrid_cfg.features, *data.linear_cfg.input_features,
                                 *(n for e in variants(data.hybrid_cfg.features).values()
                                   for n in e))))
    return tx.feature_table(data.hourly, names, TABLE)


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "hybrid_forward_add.csv", index=False,
                            float_format="%.6g")
    ref = "hybrid shipped"
    geo_ref = float(np.exp(np.log(df[ref]).mean()))
    table = {}
    for c in [c for c in df.columns if c.startswith("hybrid")]:
        geo = float(np.exp(np.log(df[c]).mean()))
        won = int((df[c] < df[ref]).sum())
        table[c] = {"score": geo, "vs shipped": geo / geo_ref - 1, "folds won": won,
                    "keep": bool(geo / geo_ref - 1 < -KEEP_GAIN and won >= KEEP_FOLDS)}
    table = pd.DataFrame(table).T.sort_values("score")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df.drop(columns="seconds", errors="ignore").T.to_string())
        log(f"\nover {len(df)} folds:\n{table.to_string()}")
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = ch.Data()
    table = load_table(data)
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    names = tuple(variants(data.hybrid_cfg.features))
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        cached = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = tuple(n for n in names if f"hybrid {n}" not in cached)
        if not todo:
            rows.append(cached)
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        new = {}
        for n in todo:  # one variant at a time, saved as it goes
            new.update(score_fold(data, table, start, end, timer, (n,)))
            if not pilot:
                path.write_text(json.dumps({"fold": f"{start:%Y-%m}", **cached, **new}),
                                encoding="utf-8")
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
        r = {"fold": f"{start:%Y-%m}", **cached, **new}
        log(f"  fold {r['fold']} ({time.perf_counter() - t:.0f} s): " + ", ".join(
            f"{k} {v:.4f}" for k, v in new.items()))
        if pilot:
            from training.xgboost_model import peak_mb

            log(f"  time: hybrid {timer['hybrid']:.0f} s for {len(todo)} variants; peak "
                f"memory {peak_mb():.0f} MB; six folds -> ~"
                f"{6 * (time.perf_counter() - t) / 60:.0f} min")
            return r
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
