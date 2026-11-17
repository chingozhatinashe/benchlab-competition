"""The near-misses combined in the submitted linear model, on the yearly folds (pre-registered).

    python -m training.near_miss --pilot --folds 2024   # one fold, timed, not cached
    python -m training.near_miss [--max-seconds N]      # every fold; resumable
    python -m training.near_miss --summary

Step 1 of docs/next_steps.md. Each near-miss was at least 0.3% better on at least 4 of 6
folds alone, short of the 1% bar:
    cap 5%       Config.cap_quantile 0.05 instead of 0.01        -0.83%, 5 of 6 (slow_wind.md)
    density 1h   density_mean_1h instead of density_mean_24h     -0.31%, 4 of 6
                                                                 (official_reselection.md)
    wsa          wsa_speed_lead_0h, offline-calibrated per fold  -0.75%, 4 of 6 (wsa.md)
    bayes        structural_bayes_parr / _mean_lead_0h from the  -0.41%, 4 of 6
                 latent fold fits (no refit)                     (bayes_features.md)
Candidates, fixed before running (nested, cheapest at runtime first):
    (a) cap 5% + density 1h       (b) (a) + wsa       (c) (b) + bayes
The single near-misses are scored too, as a check that this setup reproduces them; they are
not candidates. Rule, fixed before running: keep a candidate if it is > 1% better than the
reference (geometric mean over the six folds), wins >= 4 of 6, and its slow-wind runs are
not more than 1% worse; the smallest passing set wins. The parts were chosen on these folds,
so the summary also reports, for each fold, the candidate the rule picks on the other five.
The eval year is not used. The WSA calibration comes from training.wsa_model's cache and the
Bayes draws from training.bayes_stream's (run those first).
Results: src/experiments/near_miss.csv (cache: src/data/near_miss_cache/).
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
from models import wsa
from models.bayes_model import signal_name
from models.linear_model import LinearModel
from training import bayes_features as bf
from training import competition_score as cs
from training import model_diagnostics as dg
from training import pull_features as pf
from training import wsa_model as tw
from training import xgboost_model as tx
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "near_miss_cache"
KEEP_GAIN, KEEP_FOLDS, SLOW_LIMIT = 0.01, 4, 0.01
REFERENCE_CAP = 0.005  # the submitted config before this study (shipped (a) on 2026-11-17)
DENSITY_OLD, DENSITY_NEW = "density_mean_24h", "density_mean_1h"
WSA = "wsa_speed_lead_0h"
BAYES = tuple(f"{signal_name(s)}_lead_0h" for s in ("parr", "mean"))
PARTS = {
    "cap 5%": {"cap_quantile": 0.05},
    "density 1h": {"swap": (DENSITY_OLD, DENSITY_NEW)},
    "wsa": {"add": (WSA,)},
    "bayes": {"add": BAYES},
}
CANDIDATES = {
    "(a) cap + density": ("cap 5%", "density 1h"),
    "(b) (a) + wsa": ("cap 5%", "density 1h", "wsa"),
    "(c) (b) + bayes": ("cap 5%", "density 1h", "wsa", "bayes"),
}
VARIANTS = {"reference": (), **{p: (p,) for p in PARTS}, **CANDIDATES}


def log(msg):
    print(msg, flush=True)


def reference_config(cfg):
    """The submitted config as it was before this study: density 24 h, cap 0.5%."""
    features = tuple(DENSITY_OLD if f == DENSITY_NEW else f for f in cfg.features)
    return replace(cfg, features=features, cap_quantile=REFERENCE_CAP)


def variant_config(cfg, parts):
    features = list(cfg.features)
    for p in parts:
        spec = PARTS[p]
        if "swap" in spec:
            old, new = spec["swap"]
            features[features.index(old)] = new
        features += list(spec.get("add", ()))
        if "cap_quantile" in spec:
            cfg = replace(cfg, cap_quantile=spec["cap_quantile"])
    return replace(cfg, features=tuple(features),
                   optional=tuple(f for f in cfg.optional if f in features))


class Data(dg.Data):
    """model_diagnostics.Data with the reference config (reference_config), both density
    windows, and the WSA study's ring data."""

    def __init__(self):
        super().__init__()
        self.cfg = reference_config(self.cfg)
        todo = [c for c in (DENSITY_OLD, DENSITY_NEW) if c not in self.table]
        if todo:
            inputs = build.Inputs(hourly=pf.load_hourly())
            extra = windows.window_table(inputs, self.runs6,
                                         [windows.WindowSpec.parse(c) for c in todo])
            self.table = self.table.join(extra.astype(np.float32))
        self.wsa = tw.Data()
        if not self.wsa.t0s.equals(self.runs6):
            raise RuntimeError("the WSA study's runs differ from the training runs")


def wsa_frame(data, year, t0s):
    """The WSA lag columns of `t0s`, with the fold's cached offline calibration."""
    path = tw.CACHE_DIR / "full" / f"{year}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run training.wsa_model first")
    params = wsa.Params(**json.loads(path.read_text("utf-8"))["params"])
    rows = data.runs6.get_indexer(t0s)
    return pd.DataFrame(tw.wsa_columns(data.wsa, rows, params), index=t0s)


def score_fold(data, start, end, names, timer):
    year = f"{start:%Y}"
    t = time.perf_counter()
    parts = bf.fold_parts(data, start, end)  # the Bayes features of the latent fold fit
    ff, val = parts["ff"], parts["val"]
    fit_tab = ff["table"].loc[ff["fit6"]].join(parts["f_fit"]).join(
        wsa_frame(data, year, ff["fit6"]))
    val_tab = ff["table"].loc[val].join(parts["f_val"]).join(wsa_frame(data, year, val))
    wsa_cols = [c for c in fit_tab if c.startswith(tw.SIGNAL)]
    med = fit_tab[wsa_cols].median()  # a run without a ring: the fit runs' median
    fit_tab[wsa_cols] = fit_tab[wsa_cols].fillna(med)
    val_tab[wsa_cols] = val_tab[wsa_cols].fillna(med)
    timer["features"] = timer.get("features", 0.0) + time.perf_counter() - t
    pos_val = data.series.index.get_indexer(val)
    obs, pers, regime = parts["obs"], parts["pers"], parts["regime"]
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"]}
    for name in names:
        t = time.perf_counter()
        cfg = variant_config(data.cfg, VARIANTS[name])
        cols = list(cfg.input_features)
        lin = LinearModel(cfg).fit(data.values, ff["pos_fit"], fit_tab[cols].to_numpy(float),
                                   end=ff["start_pos"])
        f = lin.forecast(data.values, pos_val, val_tab[cols].to_numpy(float))
        row = dg.regime_scores(f, obs, pers, regime)
        row["residual slow wind"] = float(np.nanmean((obs - f)[regime == "slow wind"]))
        out[name] = row
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return out


def run(years=None, pilot=False, max_seconds=None):
    names = list(VARIANTS)
    started = time.perf_counter()
    bf.FIT = "latent"
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
            f"{6 * sum(timer.values()) / 60:.0f} min after the data "
            f"({time.perf_counter() - started:.0f} s in all)")
    return summarize(rows, pilot)


def judge(rows, years, name):
    """(gain, folds won, slow runs vs reference, keep) of `name` over `years`."""
    rel = np.array([rows[y][name]["all"] / rows[y]["reference"]["all"] for y in years])
    slow = np.array([rows[y][name]["slow wind"] / rows[y]["reference"]["slow wind"]
                     for y in years])
    gain = float(np.exp(np.log(rel).mean()) - 1)
    won = int((rel < 1).sum())
    slow_g = float(np.exp(np.log(slow).mean()) - 1)
    need = KEEP_FOLDS if len(years) == 6 else int(np.ceil(KEEP_FOLDS * len(years) / 6))
    return gain, won, slow_g, bool(gain < -KEEP_GAIN and won >= need and slow_g <= SLOW_LIMIT)


def pick(rows, years):
    """The rule's choice: the first (smallest) candidate that passes, else None."""
    return next((c for c in CANDIDATES if judge(rows, years, c)[3]), None)


def summarize(rows, pilot=False):
    years = sorted(rows)
    names = [n for n in VARIANTS if all(n in rows[y] for y in years)]
    out, long = {}, []
    for name in names:
        gain, won, slow_g, keep = judge(rows, years, name)
        out[name] = {
            "/ persistence": float(np.exp(np.mean([np.log(rows[y][name]["all"]
                                                          / rows[y]["persistence"])
                                                   for y in years]))),
            "vs reference": gain, "folds won": won, "slow runs vs reference": slow_g,
            **{f"{r} vs reference": float(np.exp(np.mean(
                [np.log(rows[y][name][r] / rows[y]["reference"][r]) for y in years])) - 1)
               for r in dg.RUN_REGIMES if r != "slow wind"},
            "slow residual": float(np.mean([rows[y][name]["residual slow wind"]
                                            for y in years])),
            "candidate": name in CANDIDATES, "keep": keep and name in CANDIDATES,
        }
        long += [{"fold": y, "variant": name, "persistence": rows[y]["persistence"],
                  **rows[y][name]} for y in years]
    table = pd.DataFrame(out).T
    if not pilot:
        pd.DataFrame(long).to_csv(EXPERIMENTS / "near_miss.csv", index=False,
                                  float_format="%.6g", lineterminator="\n")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 20):
        log(f"\nover {len(years)} fold(s):\n{table.to_string()}")
        log("by fold (vs reference):\n" + pd.DataFrame(
            {y: {n: rows[y][n]["all"] / rows[y]["reference"]["all"] for n in table.index}
             for y in years}).round(4).to_string())
    if len(years) == 6:
        log(f"\nthe rule picks: {pick(rows, years)}")
        for y in years:
            log(f"  without {y}: {pick(rows, [z for z in years if z != y])}")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summarize({p.stem: json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(CACHE_DIR.glob("*.json"))})
    else:
        run(args.folds, args.pilot, args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
