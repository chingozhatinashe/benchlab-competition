"""Lead-aligned coronal-hole features for the linear model and the hybrid, on the yearly folds.

    python -m training.lead_aligned_ch --pilot --folds 2024   # one fold, timed, not cached
    python -m training.lead_aligned_ch [--max-seconds 240]    # every fold; resumable
    python -m training.lead_aligned_ch --summary

The shipped linear model uses the central-meridian coronal-hole strip of one image,
24 h before t0 (ch_cm_at_24h), for every lead. Wind from a hole at the central meridian
reaches L1 about 3-5 days later, so the image that matters for the hour t0 + k is the one
about D hours before it: lag D - k before t0. Collin et al. (2025) time their coronal-hole
features against the target this way (4-7 days before it).

Lead-aligned feature at delay D: for lead k, the strip at lag L_k = D - k, rounded up to
the 6-hour image grid (so live runs need at most one image per grid step) and at least 0.
The 24-hour window variant averages the strips at lags L_k ... L_k + 24 (delays D to D + 24).

No model change is needed: the linear model fits each lead on its own, so a model whose
lead k uses the column at L_k equals, lead by lead, a model fitted with that one column.
`Stitched` fits one LinearModel per distinct L_k and takes lead k from the one with L_k.
It has the LinearModel interface the hybrid needs (config.features, forecast), so the
hybrid's trees can be refitted on top of it.

Variants (all on the shipped config, with its data fixes and the regressor cap):
    shipped        ch_cm_at_24h for every lead
    at D           ch_cm_at_24h replaced by the lead-aligned strip at delay D
    win D          ... by the 24-hour window mean at delays D .. D + 24
    + at D         the lead-aligned strip added to ch_cm_at_24h
The hybrid is refitted on the best linear variant chosen on the folds (`--hybrid`).

Score: per-run mean TIMSE / persistence; summaries are geometric means over folds.
Results: src/experiments/lead_aligned_ch.csv. Motivation: docs/model_review.md, item 3.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import HORIZON, LEADS
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import xgboost_model as tx
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "lead_aligned_ch_cache"
TABLE = Path(__file__).resolve().parents[1] / "data" / "history_cache" / "table_hourly_ch.pkl"
GRID = 6  # hours between pulled images
MAX_LAG = 120  # largest lag in the pulled history (feature_engineering.windows.CH_LAG_HOURS)
FIXED = "ch_cm_at_24h"
WINDOW = 24
DELAYS = (60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120)
WIN_DELAYS = tuple(d for d in DELAYS if math.ceil((d - 1) / GRID) * GRID + WINDOW <= MAX_LAG)


def log(msg):
    print(msg, flush=True)


def lead_lags(delay):
    """L_k for each lead k: delay - k rounded up to the image grid, at least 0.

    For delay < 73 h the far leads would need an image after t0 - 1 h; they take
    the latest one (lag 0).
    """
    return np.maximum(0, np.ceil((delay - LEADS) / GRID).astype(int) * GRID)


def column(lag, window=0):
    return f"ch_cm_at_{lag}h" if not window else f"ch_cm_win{window}at_{lag}h"


def lag_columns():
    return [f"ch_cm_at_{lag}h" for lag in range(0, MAX_LAG + 1, GRID)]


def add_windows(table):
    """The 24-hour window means of the strip, as extra columns."""
    for lag in range(0, MAX_LAG - WINDOW + 1, GRID):
        cols = [f"ch_cm_at_{lag + h}h" for h in range(0, WINDOW + 1, GRID)]
        table[column(lag, WINDOW)] = table[cols].mean(axis=1, skipna=True).where(
            table[cols].notna().sum(axis=1) >= 3)
    return table


class Stitched:
    """Lead k from a LinearModel fitted with the coronal-hole column at lag L_k."""

    def __init__(self, base_cfg, base_features, lags, window=0):
        self.lags = np.asarray(lags)
        self.window = window
        self.base_features = tuple(base_features)
        self.ch_cols = {int(lag): column(int(lag), window) for lag in np.unique(self.lags)}
        feats = (*self.base_features, *dict.fromkeys(self.ch_cols.values()))
        self.config = replace(base_cfg, features=feats, optional=())  # what callers pass in
        self.models = {}
        self._base_cfg = base_cfg

    def _sub(self, lag):
        f = tuple(dict.fromkeys((*self.base_features, self.ch_cols[lag])))
        opt = tuple(c for c in f if c.startswith("ch_cm"))
        return replace(self._base_cfg, features=f, optional=opt)

    def _x(self, x, lag):
        idx = [self.config.features.index(c) for c in self._sub(lag).features]
        return np.asarray(x, dtype=float)[:, idx]

    def fit(self, values, positions, x, end=None):
        for lag in self.ch_cols:
            self.models[lag] = LinearModel(self._sub(lag)).fit(values, positions,
                                                               self._x(x, lag), end=end)
        return self

    def forecast(self, values, positions, x):
        out = np.empty((len(positions), HORIZON))
        for lag, model in self.models.items():
            leads = self.lags == lag
            out[:, leads] = model.forecast(values, positions, self._x(x, lag))[:, leads]
        return out


def variants(base, with_windows=False):
    """name -> (base features, lags or None, window)."""
    without = tuple(f for f in base if f != FIXED)
    out = {"shipped": (base, None, 0)}
    for d in DELAYS:
        out[f"at {d}"] = (without, lead_lags(d), 0)
    if with_windows:  # worse on the 2024 pilot fold (+1% to +6%); left out of the full run
        for d in WIN_DELAYS:
            out[f"win {d}"] = (without, lead_lags(d), WINDOW)
    for d in DELAYS:
        out[f"+ at {d}"] = (base, lead_lags(d), 0)
    return out


def load_table(data):
    names = tuple(dict.fromkeys((*data.hybrid_cfg.features, *data.linear_cfg.features,
                                 *lag_columns())))
    table = tx.feature_table(data.hourly, names, TABLE)
    return add_windows(table.astype(float))


def make_model(data, spec):
    feats, lags, window = spec
    if lags is None:
        return LinearModel(replace(data.linear_cfg, features=feats))
    return Stitched(data.linear_cfg, feats, lags, window)


def score_fold(data, table, start, end, timer, linear_names, hybrid_names=()):
    """Scores of the named linear variants, and of the hybrid refitted on the named ones."""
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos6, pos_h = data.series.index.get_indexer(fit6), data.series.index.get_indexer(fit_h)

    def x(t0s, names):
        return table.loc[t0s, list(names)].to_numpy(dtype=float)

    out = {}
    for name, spec in variants(data.linear_cfg.features).items():
        if name not in linear_names and name not in hybrid_names:
            continue
        t = time.perf_counter()
        lin = make_model(data, spec)
        lin.fit(data.values, pos6, x(fit6, lin.config.features), end=start_pos)
        out[f"linear {name}"] = data.ratio(fold, lin.forecast(
            data.values, fold["pos"], x(fold["t0s"], lin.config.features)))
        timer["linear"] = timer.get("linear", 0.0) + time.perf_counter() - t
        if name in hybrid_names:
            t = time.perf_counter()
            hyb = XGBoostModel(data.hybrid_cfg, lin)
            hyb.fit(data.values, pos_h, x(fit_h, hyb.input_features), end=start_pos)
            out[f"hybrid {name}"] = data.ratio(fold, hyb.forecast(
                data.values, fold["pos"], x(fold["t0s"], hyb.input_features)))
            timer["hybrid"] = timer.get("hybrid", 0.0) + time.perf_counter() - t
    return out


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "lead_aligned_ch.csv", index=False,
                            float_format="%.6g")
    table = {}
    for c in [c for c in df.columns if c.startswith(("linear", "hybrid"))]:
        model, name = c.split(" ", 1)
        ref = f"{model} shipped"
        ok = df[list(dict.fromkeys([c, ref]))].dropna()
        geo = float(np.exp(np.log(ok[c]).mean()))
        table[c] = {"score": geo,
                    "vs shipped": geo / float(np.exp(np.log(ok[ref]).mean())) - 1,
                    "folds won": int((ok[c] < ok[ref]).sum()), "folds": len(ok)}
    table = pd.DataFrame(table).T
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df.drop(columns="seconds", errors="ignore").T.to_string())
        log(f"\nover the folds:\n{table.to_string()}")
    return df


def run(years=None, pilot=False, max_seconds=None, hybrid=()):
    started = time.perf_counter()
    data = ch.Data()
    table = load_table(data)
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, []
    names = tuple(variants(data.linear_cfg.features))
    hybrid = ("shipped", *hybrid) if hybrid else ()
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        cached = ({} if pilot or not path.exists()
                  else json.loads(path.read_text(encoding="utf-8")))
        lin_todo = tuple(n for n in names if f"linear {n}" not in cached)
        hyb_todo = tuple(n for n in hybrid if f"hybrid {n}" not in cached)
        if not lin_todo and not hyb_todo:
            rows.append(cached)
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        new = score_fold(data, table, start, end, timer, lin_todo, hyb_todo)
        r = {"fold": f"{start:%Y-%m}", **cached,
             **{k: v for k, v in new.items() if k not in cached}}
        r["seconds"] = cached.get("seconds", 0.0) + time.perf_counter() - t
        log(f"  fold {r['fold']} ({time.perf_counter() - t:.0f} s): " + ", ".join(
            f"{k} {v:.4f}" for k, v in new.items()))
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
    parser.add_argument("--hybrid", nargs="+", default=(),
                        help='linear variants to refit the hybrid on, e.g. "at 96"')
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summarize([json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(CACHE_DIR.glob("*.json"))])
    else:
        run(args.folds, args.pilot, args.max_seconds, tuple(args.hybrid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
