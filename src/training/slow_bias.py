"""Is the slow-wind over-forecast what the official metric rewards? On the yearly folds.

    python -m training.slow_bias --pilot --folds 2024   # one fold, timed, not cached
    python -m training.slow_bias                        # every fold; resumable
    python -m training.slow_bias --summary

Step 5 of docs/next_steps.md. The submitted linear model over-forecasts runs that stay slow
by about 40 km/s (docs/model_diagnostics.md), and every attempt to remove it failed
(docs/slow_wind.md, docs/post_calibration.md). The question: would removing it help the
official score at all, or does the metric's tail-weighted TIMSE reward it?

Per fold, the submitted config is refitted on the fold's training runs; its forecasts of
those runs (in-sample) and of the validation runs are shifted down by delta km/s (every
lead) in a chosen set of runs:
    oracle      the runs whose regime is "slow wind" (training.model_diagnostics, from the
                truth: no arrival, no CME, staying slow)
    last<450    a realistic indicator: the last observed hourly speed below 450 km/s
    all runs    a global shift (a control: the retune's blend already set the global lean)
delta is (i) a grid 0..80 km/s scored on the validation runs (hindsight: an upper bound),
and (ii) chosen on the fold's training runs (minimising S_final there) and applied to the
validation runs (honest). Also per lead: the fitted mean residual of the set's training runs.
Reported: S_final, its RMSE and TRMSE parts, and S within each run regime.

Reading, fixed before running:
    - if even the oracle's fitted shift doesn't make S_final > 1% better on >= 4 of 6 folds,
      the over-forecast is close to what the metric rewards: stop chasing it;
    - if the oracle passes but last<450 doesn't, the problem is detecting slow runs;
    - if last<450 passes the rule too, it is a candidate model change (a separate study).
The eval year is not used. Results: src/experiments/slow_bias.csv (cache:
src/data/slow_bias_cache/).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.linear_model import LinearModel
from training import competition_score as cs
from training import model_diagnostics as dg
from training import xgboost_model as tx
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "slow_bias_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
SLOW_LAST = 450.0
GRID = tuple(range(0, 81, 5))
SETS = ("oracle", "last<450", "all runs")


def log(msg):
    print(msg, flush=True)


def last_speed(data, t0s):
    """The last observed hourly speed before t0 (the hour labelled t0 - 1 h, else the latest
    within 6 h; NaN if none)."""
    s = data.series.dropna()
    v = s.asof(pd.DatetimeIndex(t0s) - pd.Timedelta(hours=1))
    stamp = pd.Series(s.index, index=s.index).asof(pd.DatetimeIndex(t0s) - pd.Timedelta(hours=1))
    old = (pd.DatetimeIndex(t0s) - pd.DatetimeIndex(stamp)) > pd.Timedelta(hours=7)
    return np.where(old, np.nan, v.to_numpy(float))


def masks(data, t0s, regime):
    last = last_speed(data, t0s)
    return {"oracle": np.asarray(regime) == "slow wind",
            "last<450": np.nan_to_num(last, nan=np.inf) < SLOW_LAST,
            "all runs": np.ones(len(t0s), dtype=bool)}


def parts(f, obs, pers):
    s = cs.score(f, obs, pers)
    return s["s_final"], s["rmse"], s["trmse"]


def best_delta(f, obs, pers, m):
    """The grid shift minimising S_final of the (training) runs, and that S_final."""
    scores = []
    for d in GRID:
        g = f.copy()
        g[m] -= d
        scores.append(cs.score(g, obs, pers)["s_final"])
    i = int(np.argmin(scores))
    return GRID[i], scores[i]


def score_fold(data, start, end, timer):
    t = time.perf_counter()
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[data.runs6 < start]
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    table = data.table.copy()
    table[data.pfss] = table[data.pfss].fillna(table.loc[fit6, data.pfss].median())
    cols = list(data.cfg.input_features)
    pos_fit, pos_val = data.series.index.get_indexer(fit6), data.series.index.get_indexer(val)
    lin = LinearModel(data.cfg).fit(data.values, pos_fit, table.loc[fit6, cols].to_numpy(float),
                                    end=start_pos)
    f_fit = lin.forecast(data.values, pos_fit, table.loc[fit6, cols].to_numpy(float))
    f_val = lin.forecast(data.values, pos_val, table.loc[val, cols].to_numpy(float))
    obs_fit, pers_fit = cs.verifying(data.truth, fit6), cs.persistence(data.truth, fit6)
    obs_val, pers_val = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    reg_fit, reg_val = data.run_regimes(fit6), data.run_regimes(val)
    m_fit, m_val = masks(data, fit6, reg_fit), masks(data, val, reg_val)
    timer["fit"] = timer.get("fit", 0.0) + time.perf_counter() - t

    t = time.perf_counter()
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers_val[:, None], cs.HORIZON, 1),
                                                     obs_val, pers_val)["s_final"]}
    s, r, tr = parts(f_val, obs_val, pers_val)
    out["reference"] = {"all": s, "rmse": r, "trmse": tr,
                        **dg.regime_scores(f_val, obs_val, pers_val, reg_val)}
    res = obs_val - f_val
    out["slow residual"] = float(np.nanmean(res[reg_val == "slow wind"]))
    for name in SETS:
        mv, mf = m_val[name], m_fit[name]
        out[f"{name} share"] = float(mv.mean())
        out[f"{name} precision"] = float((reg_val[mv] == "slow wind").mean()) if mv.any() else 0.0
        out[f"{name} recall"] = float(mv[reg_val == "slow wind"].mean())
        for d in GRID:  # hindsight grid on the validation runs
            g = f_val.copy()
            g[mv] -= d
            s, r, tr = parts(g, obs_val, pers_val)
            out[f"{name} grid {d}"] = {"all": s, "rmse": r, "trmse": tr,
                                      "slow wind": dg.regime_scores(g, obs_val, pers_val,
                                                                    reg_val)["slow wind"]}
        d_fit, _ = best_delta(f_fit, obs_fit, pers_fit, mf)  # honest: chosen on training
        g = f_val.copy()
        g[mv] -= d_fit
        s, r, tr = parts(g, obs_val, pers_val)
        out[f"{name} fitted"] = {"delta": d_fit, "all": s, "rmse": r, "trmse": tr,
                                 **dg.regime_scores(g, obs_val, pers_val, reg_val)}
        lead_bias = np.nanmean((f_fit - obs_fit)[mf], axis=0)  # per lead, on training
        g = f_val.copy()
        g[mv] -= np.nan_to_num(lead_bias)
        s, r, tr = parts(g, obs_val, pers_val)
        out[f"{name} lead bias"] = {"mean delta": float(np.nanmean(lead_bias)), "all": s,
                                    "rmse": r, "trmse": tr,
                                    **dg.regime_scores(g, obs_val, pers_val, reg_val)}
    timer["score"] = timer.get("score", 0.0) + time.perf_counter() - t
    return out


def run(years=None, pilot=False):
    started = time.perf_counter()
    data = dg.Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, {}
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        year = f"{start:%Y}"
        path = CACHE_DIR / f"{year}.json"
        if path.exists() and not pilot:
            rows[year] = json.loads(path.read_text("utf-8"))
            continue
        t = time.perf_counter()
        rows[year] = score_fold(data, start, end, timer)
        if not pilot:
            path.write_text(json.dumps(rows[year]), encoding="utf-8")
        log(f"  {year} in {time.perf_counter() - t:.0f} s")
    if pilot:
        log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
            + f"; peak memory {tx.peak_mb():.0f} MB; six folds -> ~"
            f"{6 * sum(timer.values()) / 60:.0f} min after the data")
    return summarize(rows, pilot)


def gmean_rel(rows, years, key, field="all"):
    rel = np.array([rows[y][key][field] / rows[y]["reference"][field] for y in years])
    return float(np.exp(np.log(rel).mean()) - 1), int((rel < 1).sum())


def summarize(rows, pilot=False):
    years = sorted(rows)
    out, long = {}, []
    for name in SETS:
        for kind in ("fitted", "lead bias"):
            key = f"{name} {kind}"
            gain, won = gmean_rel(rows, years, key)
            out[key] = {"vs reference": gain, "folds won": won,
                        "rmse": gmean_rel(rows, years, key, "rmse")[0],
                        "trmse": gmean_rel(rows, years, key, "trmse")[0],
                        "slow wind": gmean_rel(rows, years, key, "slow wind")[0],
                        **{r: gmean_rel(rows, years, key, r)[0] for r in dg.RUN_REGIMES
                           if r != "slow wind"},
                        "delta": float(np.mean([rows[y][key].get(
                            "delta", rows[y][key].get("mean delta")) for y in years])),
                        "share": float(np.mean([rows[y][f"{name} share"] for y in years])),
                        "precision": float(np.mean([rows[y][f"{name} precision"]
                                                    for y in years])),
                        "recall": float(np.mean([rows[y][f"{name} recall"] for y in years]))}
            out[key]["passes"] = bool(gain < -KEEP_GAIN and won >= min(KEEP_FOLDS, len(years)))
        for d in GRID:
            key = f"{name} grid {d}"
            gain, won = gmean_rel(rows, years, key)
            long.append({"set": name, "delta": d, "vs reference": gain, "folds won": won,
                         "rmse": gmean_rel(rows, years, key, "rmse")[0],
                         "trmse": gmean_rel(rows, years, key, "trmse")[0],
                         "slow wind": gmean_rel(rows, years, key, "slow wind")[0]})
    table = pd.DataFrame(out).T
    grid = pd.DataFrame(long)
    if not pilot:
        flat = [{"fold": y, "variant": k, **({"value": v} if not isinstance(v, dict) else v)}
                for y in years for k, v in rows[y].items()]
        pd.DataFrame(flat).to_csv(EXPERIMENTS / "slow_bias.csv", index=False,
                                  float_format="%.6g", lineterminator="\n")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 20):
        log(f"\nover {len(years)} fold(s); slow-run residual (obs - forecast): "
            + ", ".join(f"{y} {rows[y]['slow residual']:.1f}" for y in years) + " km/s")
        log(f"\nshift chosen on the training runs:\n{table.to_string()}")
        log("\nhindsight grid (vs reference, geometric mean over folds):\n"
            + grid.pivot(index="delta", columns="set", values="vs reference").to_string())
        log("\n... its RMSE and TRMSE parts, oracle:\n"
            + grid[grid.set == "oracle"].set_index("delta")[["rmse", "trmse", "slow wind"]]
            .to_string())
    return table, grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summarize({p.stem: json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(CACHE_DIR.glob("*.json"))})
    else:
        run(args.folds, args.pilot)


if __name__ == "__main__":
    sys.exit(main())
