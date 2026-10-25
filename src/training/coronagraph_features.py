"""Coronagraph (SOHO/LASCO) features in the hybrid and the linear model, on the yearly folds.

    python -m training.coronagraph_features --pilot --folds 2024   # one fold, timed, not cached
    python -m training.coronagraph_features [--max-seconds 240]    # every fold; resumable
    python -m training.coronagraph_features --summary

Path (i) of item 4 (docs/model_review.md): the coronagraph measurements go straight into
the speed models, with no CME model. The history comes from training.pull_lasco_images
(sector-ring cells of every C2 image each hour and every C3 image each 2 h);
feature_engineering.coronagraph turns them into brightening signals, and
feature_engineering.windows into per-t0 window maxima ("c2max_max_24h": the largest
C2 brightening in any sector in the last 24 h), by the live availability rule.

Variants (the trees refitted with the shipped settings on the shipped linear model):
    shipped              the shipped hybrid
    hybrid + <set>       the set added to the trees' 30 regressors
    linear + <feature>   one coronagraph term in the linear model (optional: a run
                         without it uses the submodel)
Score: per-run mean TIMSE / persistence, geometric mean over folds; also by regime
(training.solar_wind_events: CME runs against the rest), relative to the shipped
model on the same runs. Decision rule (set before running): more than 1% better on
the six-fold score, winning at least 4 of 6 folds, with the CME runs better and the
other runs not worse by more than 0.5%. Results: src/experiments/coronagraph_features.csv.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build, windows
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import linear_model as tl
from training import pull_lasco_images as pl
from training import regime_models as rm
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "coronagraph_features_cache"
TABLE = CACHE_DIR / "table_hourly.pkl"
C2_MAX = ("c2max_max_6h", "c2max_max_24h", "c2max_max_72h")
C3_MAX = ("c3max_max_6h", "c3max_max_24h", "c3max_max_72h")
HYBRID_SETS = {
    "c2 max": C2_MAX,
    "c2 max + mean": (*C2_MAX, "c2mean_max_24h", "c2mean_max_72h"),
    "c2 + c3 max": (*C2_MAX, *C3_MAX),
    "c2 max + share": (*C2_MAX, "c2share_max_24h", "c2share_max_72h", "c3share_max_72h"),
    "all c2": tuple(s.name for s in windows.coronagraph_candidates()
                    if s.signal.startswith("c2")),
}
LINEAR_TERMS = ("c2max_max_24h", "c2max_max_72h")
_REFERENCE = {}  # fold start -> (shipped hybrid forecast, ratio), reused across variants


def log(msg):
    print(msg, flush=True)


def load_signals():
    """The coronagraph signals of the pulled history (feature_engineering.coronagraph)."""
    from feature_engineering import coronagraph as cg

    cells = pl.load_cells()
    tables = {i: g.set_index("time")[cg.cell_columns()].dropna()
              for i, g in cells.groupby("instrument")}
    return cg.signal_frame(tables)


def coronagraph_table(t0s):
    if TABLE.exists():
        table = pd.read_pickle(TABLE)
        if table.index.equals(t0s):
            return table
    specs = windows.coronagraph_candidates()
    table = windows.window_table(build.Inputs(coronagraph=load_signals()), t0s, specs)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.astype(np.float32).to_pickle(TABLE)
    return table


def variant_names():
    return ("shipped", *(f"hybrid + {k}" for k in HYBRID_SETS),
            *(f"linear + {f}" for f in LINEAR_TERMS))


def route_ratios(data, fold, forecast, ref_forecast):
    """Mean TIMSE relative to the reference forecast, on CME runs and on the others."""
    s = tl.run_scores(forecast, fold["y"], fold["q"])
    r = tl.run_scores(ref_forecast, fold["y"], fold["q"])
    route = data.labels["route"].reindex(fold["t0s"]).to_numpy()
    out = {}
    for name, mask in (("cme", route == "CME"), ("other", route != "CME")):
        m = mask & fold["scored"]
        out[name] = float(s[m].mean() / r[m].mean()) if m.any() else np.nan
    return out


def score_fold(data, table, start, end, timer, names):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos6, pos_h = data.series.index.get_indexer(fit6), data.series.index.get_indexer(fit_h)

    def x(t0s, cols):
        base = [c for c in cols if c not in table.columns]
        both = pd.concat([data.table.loc[t0s, base], table.loc[t0s, [c for c in cols
                                                                     if c in table.columns]]],
                         axis=1)
        return both[list(cols)].to_numpy(dtype=float)

    cfg = data.linear_cfg
    lin = LinearModel(cfg).fit(data.values, pos6, x(fit6, cfg.input_features), end=start_pos)
    out = {}
    for name in variant_names():
        if name not in names:
            continue
        t = time.perf_counter()
        if name.startswith("linear + "):
            feat = name.removeprefix("linear + ")
            lcfg = replace(cfg, features=(*cfg.features, feat), optional=(*cfg.optional, feat))
            model = LinearModel(lcfg).fit(data.values, pos6, x(fit6, lcfg.input_features),
                                          end=start_pos)
            f = model.forecast(data.values, fold["pos"], x(fold["t0s"], lcfg.input_features))
            ref_lin = lin.forecast(data.values, fold["pos"], x(fold["t0s"], cfg.input_features))
            out[name] = data.ratio(fold, f)
            out["linear shipped"] = data.ratio(fold, ref_lin)
            for k, v in route_ratios(data, fold, f, ref_lin).items():
                out[f"{name} {k}"] = v
        else:
            def hybrid(extra):
                hcfg = replace(data.hybrid_cfg, features=(*data.hybrid_cfg.features, *extra))
                hyb = XGBoostModel(hcfg, lin)
                hyb.fit(data.values, pos_h, x(fit_h, hyb.input_features), end=start_pos)
                return hyb.forecast(data.values, fold["pos"], x(fold["t0s"], hyb.input_features))

            if start not in _REFERENCE:
                f0 = hybrid(())
                _REFERENCE[start] = (f0, data.ratio(fold, f0))
            ref, out["hybrid shipped"] = _REFERENCE[start]
            f = hybrid(HYBRID_SETS[name.removeprefix("hybrid + ")])
            out[name] = data.ratio(fold, f)
            for k, v in route_ratios(data, fold, f, ref).items():
                out[f"{name} {k}"] = v
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return out


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "coronagraph_features.csv", index=False,
                            float_format="%.6g")
    table = {}
    for name in variant_names()[1:]:
        if name not in df:
            continue
        ref = "hybrid shipped" if name.startswith("hybrid") else "linear shipped"
        geo = float(np.exp(np.log(df[name]).mean()))
        geo_ref = float(np.exp(np.log(df[ref]).mean()))
        table[name] = {"score": geo, "vs shipped": geo / geo_ref - 1,
                       "folds won": int((df[name] < df[ref]).sum()),
                       "CME runs vs shipped": float(np.exp(np.log(df[f"{name} cme"]).mean())) - 1,
                       "other runs vs shipped":
                           float(np.exp(np.log(df[f"{name} other"]).mean())) - 1}
    table = pd.DataFrame(table).T
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df[[c for c in df.columns if not c.endswith((" cme", " other"))
                                   and c != "seconds"]].T.to_string())
        log(f"\nover {len(df)} folds:\n{table.to_string()}")
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = rm.Data()
    table = coronagraph_table(data.hourly)
    log(f"data ready in {time.perf_counter() - started:.0f} s; coronagraph coverage "
        + ", ".join(f"{c} {table[c].notna().mean():.2f}" for c in C2_MAX[:1] + C3_MAX[:1]))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    names = variant_names()
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        cached = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = tuple(n for n in names[1:] if n not in cached)
        if not todo:
            rows.append(cached)
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        new = {}
        for n in todo:  # one variant at a time (with the shipped reference), saved as it goes
            new.update(score_fold(data, table, start, end, timer, (n,)))
            if not pilot:
                path.write_text(json.dumps({"fold": f"{start:%Y-%m}", **cached, **new}),
                                encoding="utf-8")
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
        r = {"fold": f"{start:%Y-%m}", **cached, **new}
        log(f"  fold {r['fold']} ({time.perf_counter() - t:.0f} s): " + ", ".join(
            f"{k} {v:.4f}" for k, v in new.items() if not k.endswith((" cme", " other"))))
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB; six folds -> ~"
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
