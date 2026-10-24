"""Coronal-hole latitude grid, B0 and darkness, lead-aligned, on the yearly folds.

    python -m training.coronal_grid images                      # metrics from the cached images
    python -m training.coronal_grid folds --pilot --folds 2024  # one fold, timed, not cached
    python -m training.coronal_grid folds [--max-seconds 240]   # every fold; resumable
    python -m training.coronal_grid folds --summary

Step 1 (`images`) recomputes, from the cached AIA 193 browse images (src/data/aia193/,
pulled by training.pull_features), the central-meridian strip (|longitude| <= 10 deg) with
Earth's B0 tilt taken into account, split by latitude, plus a darkness-weighted area
(feature_engineering.coronal_holes.grid_metrics):

    ch_cm_b0      |lat| <= 60 (as ch_cm, B0-corrected)
    ch_cm_low     |lat| <= 30
    ch_cm_earth   |lat - B0| <= 15 (around the latitude Earth sees)
    ch_cm_toward  the hemisphere tilted towards Earth, 10 < |lat| <= 60
    ch_cm_away    the other hemisphere
    ch_cm_depth   area weighted by how far below the threshold each hole pixel is

Output: src/data/features/coronal_grid.csv, one row per image cut-off.

Step 2 (`folds`) scores them on the six yearly folds, lead-aligned at D = 72 h like the
coronal-hole strip in training.lead_aligned_ch (whose best variant, "at 72", is the
reference here), in the linear model:

    ref            the lead-aligned ch_cm strip (lead_aligned_ch "at 72")
    <name>         ch_cm replaced by the lead-aligned <name>
    + <name>       <name> added to the lead-aligned ch_cm
    + b0           B0 at t0 added (seasonal Earth latitude)
    + toward/away  both hemispheres added
Score: per-run mean TIMSE / persistence, geometric mean over folds. Results:
src/experiments/coronal_grid.csv. Motivation: docs/model_review.md, item 3.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import coronal_holes, windows
from training import compare_history as ch
from training import lead_aligned_ch as la
from training import pull_features as pf
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
GRID_CSV = pf.OUT / "coronal_grid.csv"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "coronal_grid_cache"
DELAY = 72
NAMES = coronal_holes.GRID_NAMES


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------- step 1: images
def _one(args):
    cutoff, taken = args
    if pd.isna(taken):
        return {"cutoff": cutoff, **dict.fromkeys(NAMES, np.nan), "b0": np.nan}
    taken = pd.Timestamp(taken)
    path = pf.AIA_DIR / f"{taken:%Y}" / f"{taken:%Y%m%d_%H%M%S}_512_0193.jpg"
    if not path.exists():
        return {"cutoff": cutoff, **dict.fromkeys(NAMES, np.nan), "b0": np.nan}
    return {"cutoff": cutoff, **coronal_holes.grid_metrics(path.read_bytes(), taken)}


def compute_images(workers=4, max_seconds=None):
    raw = pd.read_csv(pf.OUT / "coronal_holes.csv")
    raw = raw.drop_duplicates("cutoff", keep="last")
    done = set(pd.read_csv(GRID_CSV)["cutoff"]) if GRID_CSV.exists() else set()
    todo = [(c, t) for c, t in zip(raw["cutoff"], raw["taken"], strict=True) if c not in done]
    log(f"images: {len(done)} done, {len(todo)} to go")
    started = time.perf_counter()
    chunk = 500
    with ProcessPoolExecutor(workers) as pool:
        for s in range(0, len(todo), chunk):
            rows = list(pool.map(_one, todo[s : s + chunk], chunksize=25))
            pd.DataFrame(rows).to_csv(GRID_CSV, mode="a", header=not GRID_CSV.exists(),
                                      index=False)
            log(f"  {s + len(rows)}/{len(todo)} ({time.perf_counter() - started:.0f} s)")
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run again to continue")
                return False
    return True


def load_grid():
    df = pd.read_csv(GRID_CSV)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("cutoff"), utc=True), name="cutoff")
    return df[~df.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------- step 2: folds
def grid_table(t0s):
    """The grid metrics at every lag on the image grid, for each t0 (same rules as ch_cm)."""
    grid = load_grid()
    cols = {}
    for name in NAMES:
        for lag in range(0, la.MAX_LAG + 1, la.GRID):
            cols[f"{name}_at_{lag}h"] = windows._ch_at(grid[name], t0s, lag,
                                                       windows.CH_FALLBACK_HOURS)
    out = pd.DataFrame(cols, index=t0s)
    out["b0_at_0h"] = coronal_holes.b0_angle(t0s)
    return out


class Aligned(la.Stitched):
    """Stitched with several lead-aligned signals: lead k uses each one's column at L_k."""

    def __init__(self, base_cfg, base_features, lags, signals):
        self.signals = tuple(signals)
        super().__init__(base_cfg, base_features, lags)
        self.ch_cols = {lag: tuple(f"{s}_at_{lag}h" for s in self.signals) for lag in self.ch_cols}
        cols = [c for lag in sorted(self.ch_cols) for c in self.ch_cols[lag]]
        self.config = replace(base_cfg, features=(*self.base_features, *cols), optional=())

    def _sub(self, lag):
        f = (*self.base_features, *self.ch_cols[lag])
        opt = tuple(c for c in f if c.startswith("ch_cm"))
        return replace(self._base_cfg, features=f, optional=opt)


def variants(base):
    """name -> (base features, signals)."""
    without = tuple(f for f in base if f != la.FIXED)
    out = {"ref": (without, ("ch_cm",))}
    for name in NAMES:
        out[name] = (without, (name,))
        if name != "ch_cm_b0":
            out[f"+ {name}"] = (without, ("ch_cm", name))
    out["+ b0"] = ((*without, "b0_at_0h"), ("ch_cm",))
    out["+ toward/away"] = (without, ("ch_cm", "ch_cm_toward", "ch_cm_away"))
    return out


def score_fold(data, table, start, end, timer, names):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    pos6 = data.series.index.get_indexer(fit6)

    def x(t0s, cols):
        return table.loc[t0s, list(cols)].to_numpy(dtype=float)

    lags = la.lead_lags(DELAY)
    out = {}
    for name, (feats, signals) in variants(data.linear_cfg.features).items():
        if name not in names:
            continue
        t = time.perf_counter()
        model = Aligned(data.linear_cfg, feats, lags, signals)
        model.fit(data.values, pos6, x(fit6, model.config.features), end=start_pos)
        out[f"linear {name}"] = data.ratio(fold, model.forecast(
            data.values, fold["pos"], x(fold["t0s"], model.config.features)))
        timer["linear"] = timer.get("linear", 0.0) + time.perf_counter() - t
    return out


def load_table(data):
    """Regressors, ch_cm at every lag and the grid metrics, at the 6-hourly runs."""
    t0s = data.runs6
    return pd.concat([la.load_table(data).reindex(t0s), grid_table(t0s)], axis=1)


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "coronal_grid.csv", index=False, float_format="%.6g")
    ref = "linear ref"
    table = {}
    for c in [c for c in df.columns if c.startswith("linear")]:
        geo = float(np.exp(np.log(df[c]).mean()))
        table[c] = {"score": geo,
                    "vs ref": geo / float(np.exp(np.log(df[ref]).mean())) - 1,
                    "folds won": int((df[c] < df[ref]).sum())}
    table = pd.DataFrame(table).T
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df.drop(columns="seconds", errors="ignore").T.to_string())
        log(f"\nover {len(df)} folds:\n{table.to_string()}")
    return df


def run_folds(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = ch.Data()
    table = load_table(data)
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    names = tuple(variants(data.linear_cfg.features))
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        cached = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = tuple(n for n in names if f"linear {n}" not in cached)
        if not todo:
            rows.append(cached)
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        new = score_fold(data, table, start, end, timer, todo)
        r = {"fold": f"{start:%Y-%m}", **cached, **new}
        r["seconds"] = cached.get("seconds", 0.0) + time.perf_counter() - t
        log(f"  fold {r['fold']} ({time.perf_counter() - t:.0f} s): " + ", ".join(
            f"{k} {v:.4f}" for k, v in new.items()))
        if pilot:
            log(f"  time: linear {timer['linear']:.0f} s; six folds -> "
                f"~{6 * r['seconds'] / 60:.0f} min")
            return r
        path.write_text(json.dumps(r), encoding="utf-8")
        rows.append(r)
    return summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["images", "folds"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.step == "images":
        compute_images(args.workers, args.max_seconds)
    elif args.summary:
        summarize([json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(CACHE_DIR.glob("*.json"))])
    else:
        run_folds(args.folds, args.pilot, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
