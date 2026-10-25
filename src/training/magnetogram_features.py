"""Magnetogram (SDO/HMI) features in the hybrid and the linear model, on the yearly folds.

    python -m training.magnetogram_features --pilot --folds 2024   # one fold, timed, not cached
    python -m training.magnetogram_features [--max-seconds 240]    # every fold; resumable
    python -m training.magnetogram_features --summary

Step 3, path (i) of item 4 (docs/model_review.md): the photospheric field goes straight
into the speed models, with no model of the corona. The history comes from
training.pull_hmi_images (a 10 x 10 deg latitude/longitude grid of every 6-hourly HMI
quick-look magnetogram); feature_engineering.magnetogram.signals turns each grid into
generic numbers, mainly over the central-meridian strip, where the wind reaching Earth
3-5 days later comes from:
    mg_cm_signed     net radial field (polarity) of the strip
    mg_cm_unipolar   its size: a unipolar strip is open field (a coronal hole, fast wind)
    mg_cm_unsigned   total field: active regions and their slow wind
    mg_cm_strong     share of strong-field pixels
    mg_cm_imbalance  |positive - negative| / (positive + negative) of the strong pixels
    mg_disk_unsigned total field over |longitude| <= 60 deg
    mg_polar         north minus south polar-cap field (the solar-cycle dipole)
They are read like the AIA strips (feature_engineering.windows: "mg_cm_unipolar_at_24h",
the image 24 h before the latest one by t0 - 1 h).

Variants (the trees refitted with the shipped settings on the shipped linear model):
    shipped              the shipped hybrid / linear model
    hybrid + <set>       the set added to the trees' 30 regressors
    linear + <signal>    the signal lead-aligned at 72 h (as ch_cm_lead_72h: lead k uses
                         the image about 72 h before t0 + k). Not optional: the linear
                         model has one submodel without all its optional features, so a
                         missing magnetogram would also drop the coronal-hole term; it is
                         filled with the training median instead (the HMI and AIA browse
                         archives have the same gaps)
Score: per-run mean TIMSE / persistence, geometric mean over folds; also on CME runs and
the rest (training.coronagraph_features.route_ratios). Decision rule (set before running):
more than 1% better on the six-fold score, winning at least 4 of 6 folds.
Results: src/experiments/magnetogram_features.csv.
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
from training import coronagraph_features as cf
from training import pull_hmi_images as ph
from training import regime_models as rm
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "magnetogram_features_cache"
DELAY = 72  # the coronal-hole strip's lead alignment (docs/lead_aligned_ch.md), not re-tuned


def at(signal, *lags):
    return tuple(f"{signal}_at_{h}h" for h in lags)


CM = (*at("mg_cm_unipolar", 24, 72), *at("mg_cm_unsigned", 24, 72),
      *at("mg_cm_strong", 24, 72))
HYBRID_SETS = {
    "unipolar": at("mg_cm_unipolar", 0, 24, 48, 72),
    "cm": CM,
    "cm + polarity": (*CM, *at("mg_cm_signed", 24, 72)),
    "cm + disk + polar": (*CM, *at("mg_cm_imbalance", 24, 72), *at("mg_disk_unsigned", 0),
                          *at("mg_polar", 0)),
}
LINEAR_TERMS = tuple(f"{s}_lead_{DELAY}h" for s in
                     ("mg_cm_unipolar", "mg_cm_unsigned", "mg_cm_strong", "mg_cm_imbalance"))
_REFERENCE = {}  # fold start -> (shipped hybrid forecast, ratio)


def log(msg):
    print(msg, flush=True)


def variant_names():
    return ("shipped", *(f"hybrid + {k}" for k in HYBRID_SETS),
            *(f"linear + {f}" for f in LINEAR_TERMS))


def table_columns(linear_cfg):
    cols = [c for s in HYBRID_SETS.values() for c in s]
    for f in LINEAR_TERMS:
        cfg = replace(linear_cfg, features=(*linear_cfg.features, f))
        cols += [c for c in cfg.input_features if c.startswith("mg_")]
    return list(dict.fromkeys(cols))


def magnetogram_table(t0s, columns):
    """The magnetogram columns at each t0, rebuilt every run (seconds; never stale while
    the history pull is still adding months)."""
    specs = [windows.WindowSpec.parse(c) for c in columns]
    table = windows.window_table(build.Inputs(magnetogram=ph.load_signals()), t0s, specs)
    return table.astype(np.float32)


def score_fold(data, table, start, end, timer, names):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos6, pos_h = data.series.index.get_indexer(fit6), data.series.index.get_indexer(fit_h)

    def x(t0s, cols):
        base = data.table.loc[t0s, [c for c in cols if c not in table.columns]]
        both = pd.concat([base, table.loc[t0s, [c for c in cols if c in table.columns]]], axis=1)
        return both[list(cols)].to_numpy(dtype=float)

    cfg = data.linear_cfg
    lin = LinearModel(cfg).fit(data.values, pos6, x(fit6, cfg.input_features), end=start_pos)
    ref_lin = lin.forecast(data.values, fold["pos"], x(fold["t0s"], cfg.input_features))

    def hybrid(extra):
        hcfg = replace(data.hybrid_cfg, features=(*data.hybrid_cfg.features, *extra))
        hyb = XGBoostModel(hcfg, lin)
        hyb.fit(data.values, pos_h, x(fit_h, hyb.input_features), end=start_pos)
        return hyb.forecast(data.values, fold["pos"], x(fold["t0s"], hyb.input_features))

    out = {"linear shipped": data.ratio(fold, ref_lin)}
    for name in names:
        t = time.perf_counter()
        if name.startswith("linear + "):
            feat = name.removeprefix("linear + ")
            lcfg = replace(cfg, features=(*cfg.features, feat))  # missing: training median
            model = LinearModel(lcfg).fit(data.values, pos6, x(fit6, lcfg.input_features),
                                          end=start_pos)
            f = model.forecast(data.values, fold["pos"], x(fold["t0s"], lcfg.input_features))
            ref = ref_lin
        else:
            if start not in _REFERENCE:
                f0 = hybrid(())
                _REFERENCE[start] = (f0, data.ratio(fold, f0))
            ref, out["hybrid shipped"] = _REFERENCE[start]
            f = hybrid(HYBRID_SETS[name.removeprefix("hybrid + ")])
        out[name] = data.ratio(fold, f)
        for k, v in cf.route_ratios(data, fold, f, ref).items():
            out[f"{name} {k}"] = v
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return out


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "magnetogram_features.csv", index=False,
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
        log("\nper fold:\n" + df[[c for c in df.columns if not c.endswith((" cme", " other"))]]
            .T.to_string())
        log(f"\nover {len(df)} folds:\n{table.to_string()}")
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = rm.Data()
    table = magnetogram_table(data.hourly, table_columns(data.linear_cfg))
    log(f"data ready in {time.perf_counter() - started:.0f} s; magnetogram coverage "
        + ", ".join(f"{c} {table[c].notna().mean():.2f}" for c in HYBRID_SETS["unipolar"][:2]))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    names = variant_names()[1:]
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        cached = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = tuple(n for n in names if n not in cached)
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
