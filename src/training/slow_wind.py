"""Slow-wind state bias and high-speed decay terms for the linear model, on the yearly folds.

    python -m training.slow_wind --pilot --folds 2024   # one fold, timed, not cached
    python -m training.slow_wind [--max-seconds N]      # every fold; resumable
    python -m training.slow_wind --summary

The performance study (docs/model_diagnostics.md) found:
    - runs that stay slow are 1.235 of persistence, and slow hours are over-forecast by
      41 km/s (60% of the squared error), more so in recent folds;
    - high forecasts relax too slowly after fast wind (residual -64 km/s at a last speed of
      700-750 km/s);
    - the current-sheet distance and X-ray terms hurt slow and intermediate runs.
The model already has one intercept per lead and a deliberate upward lean (the retune's
blend 0.25), so a global shift is not tried; the terms here depend on the state
(models.linear_model Config.terms: computed from the last speed and the regressors, so
they need nothing extra live).

Variants, each added to the submitted config (refitted per fold, data fixes):
    (i a) state bias      hinge:last<450; both hinges (also >550); a slow indicator
    (i c) quiet wind      the current-sheet or X-ray term gated to slow runs (its own slope
                          when last < 450); X-ray dropped; a tighter regressor cap (5%)
    (ii)  decay           hinge:last>550; speed trend (speed_slope_6h / _24h, the change of
                          the in-situ speed over the last 6 or 24 h); trend x last speed
    blend                 the reference at blend 0.1, 0.15, 0.2, 0.35 (the best variant is
                          re-checked with --variants after the run)
Score: official S_final / persistence, geometric mean over the six yearly folds, plus the
run regimes (training.model_diagnostics) and the slow runs' mean residual. Decision rule,
set before running: keep if > 1% better than the reference and winning >= 4 of 6 folds,
without the slow runs getting worse; a gain >= 0.3% on >= 4 folds goes forward to the
combination step. The eval year is not used.
Results: src/experiments/slow_wind.csv (cache: src/data/slow_wind_cache/).
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
from training import competition_score as cs
from training import model_diagnostics as dg
from training import pull_features as pf
from training import xgboost_model as tx
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "slow_wind_cache"
KEEP_GAIN, KEEP_FOLDS, NEAR_GAIN = 0.01, 4, 0.003
SLOPES = ("speed_slope_6h", "speed_slope_24h")
DHCS, XRAY = "pfss_d_hcs_lead_72h", "xray_peak_max_72h"
VARIANTS = {
    "reference": {},
    "hinge slow": {"terms": ("hinge:last<450",)},
    "hinges": {"terms": ("hinge:last<450", "hinge:last>550")},
    "slow indicator": {"terms": ("gate:const:last<450",)},
    "d_hcs gated slow": {"terms": (f"gate:{DHCS}:last<450",)},
    "xray gated slow": {"terms": (f"gate:{XRAY}:last<450",)},
    "both gated slow": {"terms": (f"gate:{DHCS}:last<450", f"gate:{XRAY}:last<450")},
    "drop xray": {"drop": (XRAY,)},
    "cap 5%": {"cap_quantile": 0.05},
    "hinge fast": {"terms": ("hinge:last>550",)},
    "speed slope 6h": {"add": ("speed_slope_6h",)},
    "speed slope 24h": {"add": ("speed_slope_24h",)},
    "slope 24h x last": {"add": ("speed_slope_24h",), "terms": ("int:last*speed_slope_24h",)},
    "blend 0.1": {"blend": 0.1},
    "blend 0.15": {"blend": 0.15},
    "blend 0.2": {"blend": 0.2},
    "blend 0.35": {"blend": 0.35},
}


def log(msg):
    print(msg, flush=True)


def variant_config(cfg, spec):
    features = tuple(f for f in cfg.features if f not in spec.get("drop", ()))
    features += tuple(spec.get("add", ()))
    out = replace(cfg, features=features,
                  optional=tuple(f for f in cfg.optional if f in features),
                  terms=tuple(spec.get("terms", ())))
    for key in ("cap_quantile", "blend"):
        if key in spec:
            out = replace(out, **{key: spec[key]})
    return out


class Data(dg.Data):
    """model_diagnostics.Data plus the speed-trend columns."""

    def __init__(self):
        super().__init__()
        inputs = build.Inputs(hourly=pf.load_hourly())
        slopes = windows.window_table(inputs, self.runs6,
                                      [windows.WindowSpec.parse(n) for n in SLOPES])
        self.table = self.table.join(slopes.astype(np.float32))


def score_fold(data, start, end, names, timer, specs=None):
    specs = specs or VARIANTS
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[data.runs6 < start]
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    regime = data.run_regimes(val)
    table = data.table.copy()
    table[data.pfss] = table[data.pfss].fillna(table.loc[fit6, data.pfss].median())
    pos_fit, pos_val = data.series.index.get_indexer(fit6), data.series.index.get_indexer(val)
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"]}
    for name in names:
        t = time.perf_counter()
        cfg = variant_config(data.cfg, specs[name])
        cols = list(cfg.input_features)
        lin = LinearModel(cfg).fit(data.values, pos_fit,
                                   table.loc[fit6, cols].to_numpy(dtype=float), end=start_pos)
        f = lin.forecast(data.values, pos_val, table.loc[val, cols].to_numpy(dtype=float))
        row = dg.regime_scores(f, obs, pers, regime)
        res = obs - f
        for r in dg.RUN_REGIMES:
            row[f"residual {r}"] = float(np.nanmean(res[regime == r]))
        row["residual all"] = float(np.nanmean(res))
        out[name] = row
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return out


def run(years=None, pilot=False, max_seconds=None, names=None):
    names = list(names or VARIANTS)
    if "reference" not in names:
        names.insert(0, "reference")
    started = time.perf_counter()
    data = Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, {}
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        year = f"{start:%Y}"
        path = CACHE_DIR / f"{year}.json"
        row = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = [n for n in names if n not in row]
        if todo:
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
            t = time.perf_counter()
            row.update(score_fold(data, start, end, todo, timer))
            if not pilot:
                path.write_text(json.dumps(row), encoding="utf-8")
            log(f"  {year}: {len(todo)} variants in {time.perf_counter() - t:.0f} s")
        rows[year] = row
    if pilot:
        log("  time: " + ", ".join(f"{k} {v:.1f} s" for k, v in timer.items())
            + f"; peak memory {tx.peak_mb():.0f} MB; six folds -> ~"
            f"{6 * sum(timer.values()) / 60:.0f} min after the data")
    return summarize(rows, pilot)


def summarize(rows, pilot=False):
    years = sorted(rows)
    names = [n for n in rows[years[0]] if isinstance(rows[years[0]][n], dict)]
    out, long = {}, []
    for name in names:
        if not all(name in rows[y] for y in years):
            continue
        rel = np.array([rows[y][name]["all"] / rows[y]["reference"]["all"] for y in years])
        slow = np.array([rows[y][name]["slow wind"] / rows[y]["reference"]["slow wind"]
                         for y in years])
        gain = float(np.exp(np.log(rel).mean()) - 1)
        won = int((rel < 1).sum())
        out[name] = {
            "/ persistence": float(np.exp(np.mean([np.log(rows[y][name]["all"]
                                                          / rows[y]["persistence"])
                                                   for y in years]))),
            "vs reference": gain, "folds won": won,
            "slow runs vs reference": float(np.exp(np.log(slow).mean()) - 1),
            **{f"{r} vs reference": float(np.exp(np.mean(
                [np.log(rows[y][name][r] / rows[y]["reference"][r]) for y in years])) - 1)
               for r in dg.RUN_REGIMES if r != "slow wind"},
            "slow residual": float(np.mean([rows[y][name]["residual slow wind"] for y in years])),
        }
        slow_ok = out[name]["slow runs vs reference"] <= 0
        out[name]["keep"] = bool(gain < -KEEP_GAIN and won >= KEEP_FOLDS and slow_ok)
        out[name]["forward"] = bool(gain <= -NEAR_GAIN and won >= KEEP_FOLDS)
        long += [{"fold": y, "variant": name, "persistence": rows[y]["persistence"],
                  **rows[y][name]} for y in years]
    table = pd.DataFrame(out).T.sort_values("vs reference")
    if not pilot:
        pd.DataFrame(long).to_csv(EXPERIMENTS / "slow_wind.csv", index=False,
                                  float_format="%.6g")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 20):
        log(f"\nover {len(years)} fold(s):\n{table.to_string()}")
        log("by fold (vs reference):\n" + pd.DataFrame(
            {y: {n: rows[y][n]["all"] / rows[y]["reference"]["all"] for n in table.index}
             for y in years}).round(4).to_string())
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--variants", nargs="*", help="only these (default: all)")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summarize({p.stem: json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(CACHE_DIR.glob("*.json"))})
    else:
        run(args.folds, args.pilot, args.max_seconds, args.variants)


if __name__ == "__main__":
    sys.exit(main())
