"""The hybrid (linear + XGBoost) with PFSS and WSA features, official metric, on the yearly folds.

    python -m training.hybrid_pfss --pilot --folds 2024   # one fold, timed, not cached
    python -m training.hybrid_pfss [--max-seconds N]      # every fold; resumable
    python -m training.hybrid_pfss --summary

The linear model did not use the expansion factor or the coronal-hole boundary distance
(docs/pfss.md) and gained only 0.75% from the WSA speed (docs/wsa.md). Bailey et al. 2021
found those inputs useful in gradient-boosted trees; this checks whether the hybrid's trees
can use them.

Per fold (fit on the runs before the fold start, as training.official_retune.fit_pair):
    anchor   the submitted linear config (with pfss_d_hcs_lead_72h), fitted on the 6-hourly
             runs; its PFSS columns' gaps take the fit runs' median (training.pfss_features)
    trees    the hybrid's regressors (training/xgboost_model_linear_config.json), official
             weights at blend BLEND (the retune's best for the hybrid), fitted on every hour;
             new columns are left missing where there is no map (the trees handle it)
Variants:
    linear shipped        the reference: must reproduce 0.7794 of persistence
    hybrid                the trees' current regressors on the new anchor
    hybrid + pfss         plus the ring signals at PFSS_LAGS and the three map scalars
    hybrid + wsa          plus the WSA/HUX speed at WSA_LAGS (that fold's offline calibration,
                          from src/experiments/wsa_model.csv: fitted on the fold's training
                          years only)
    hybrid + pfss + wsa   both
Ring columns at lag L read the newest map at t0 at the Carrington longitude that faced Earth
at t0 - L; negative lags are longitudes that will face Earth after t0 (the map was posted
before t0).
Score: official S_final against the official truth, relative to persistence; geometric mean
over the six folds. Decision rule, set before running: keep a hybrid variant only if it is
more than 1% better than the submitted linear model and wins at least 4 of 6 folds. The eval
year is not used. Results: src/experiments/hybrid_pfss.csv (cache: src/data/hybrid_pfss_cache/).
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build, pfss, windows
from models import wsa
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import competition_score as cs
from training import linear_model as tl
from training import xgboost_model as tx
from training.datasets import load_train
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "hybrid_pfss_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
BLEND = 0.1
PFSS_TREE = ("d_hcs", "log_fs", "theta_b", "foot_lat", "br_ss")
PFSS_LAGS = (-24, 0, 24, 48, 72, 96, 120)
WSA_LAGS = (-72, -48, -24, 0, 24, 48)
WSA_SIGNAL = "wsa_speed"
VARIANTS = ("linear shipped", "hybrid", "hybrid + pfss", "hybrid + wsa", "hybrid + pfss + wsa")
HOUR = pd.Timedelta(hours=1)


def log(msg):
    print(msg, flush=True)


def pfss_columns():
    return ([f"pfss_{v}_at_{lag}h" for v in PFSS_TREE for lag in PFSS_LAGS]
            + [f"{s}_at_0h" for s in windows.PFSS_SCALARS])


def wsa_columns(lags=WSA_LAGS):
    return [f"{WSA_SIGNAL}_at_{lag}h" for lag in lags]


def extra_columns(name):
    return {"linear shipped": (), "hybrid": (), "hybrid + pfss": tuple(pfss_columns()),
            "hybrid + wsa": tuple(wsa_columns()),
            "hybrid + pfss + wsa": (*pfss_columns(), *wsa_columns())}[name]


def ring_rows(rings, t0s):
    """Row of the newest ring at each t0 (as feature_engineering.windows._pfss), or -1."""
    i = rings.t0.searchsorted(t0s, side="right") - 1
    ok = i >= 0
    ok[ok] = (t0s[ok] - rings.t0[i[ok]]) < windows.CH_GRID
    return np.where(ok, i, -1)


def wsa_table(rings, t0s, params, lags=WSA_LAGS):
    """WSA/HUX speed at 1 AU of the newest ring at each t0, at Earth's longitude at t0 - L.

    HUX runs once per ring used; rows without a usable ring (or a ring more than half
    missing) are NaN.
    """
    t0s = pd.DatetimeIndex(t0s)
    out = pd.DataFrame(np.nan, index=t0s, columns=wsa_columns(lags), dtype=np.float32)
    i = ring_rows(rings, t0s)
    used = np.unique(i[i >= 0])
    if not len(used):
        return out
    log_fs = rings.ring[used, pfss.RING_VARS.index("log_fs")]
    theta_b = rings.ring[used, pfss.RING_VARS.index("theta_b")]
    good = np.isfinite(log_fs).mean(axis=1) > 0.5
    v1 = np.full((len(used), wsa.N_LON), np.nan)
    if good.any():
        v1[good] = wsa.hux(wsa.source_speed(log_fs[good], theta_b[good], params))
    has = i >= 0
    row = np.searchsorted(used, i[has])
    for lag, col in zip(lags, out.columns, strict=True):
        lon = (pfss.earth_longitude(t0s[has] - lag * HOUR) + params.shift) % wsa.N_LON
        lo = np.floor(lon).astype(int) % wsa.N_LON
        hi = (lo + 1) % wsa.N_LON
        w = lon - np.floor(lon)
        out.loc[t0s[has], col] = ((1 - w) * v1[row, lo] + w * v1[row, hi]).astype(np.float32)
    return out


def wsa_params(year):
    """The fold's offline WSA calibration (training.wsa_model, fit years only)."""
    d = pd.read_csv(EXPERIMENTS / "wsa_model.csv").set_index("fold")
    return wsa.Params(**{k: float(d.loc[year, f"param {k}"]) for k in wsa.Params().to_dict()})


class Data:
    """Speed series, 6-hourly and hourly runs, and one table for both, built once."""

    def __init__(self):
        from training import pull_ace_1h, pull_gong

        self.series = load_train()
        self.values = self.series.to_numpy(dtype=float)
        self.runs6 = tl.training_runs(self.series)
        self.hourly = tx.hourly_runs(self.series)
        self.truth = pull_ace_1h.load()
        self.linear_cfg = tl.with_fixes(tl.load_config())
        self.hybrid_cfg = replace(tx.load_config(tx.config_path("linear")),
                                  weights="official", blend=BLEND)
        lin = self.linear_cfg.input_features
        self.anchor_pfss = [n for n in lin if n.startswith("pfss_")]
        base = tuple(dict.fromkeys(n for n in (*self.hybrid_cfg.features, *lin)
                                   if not n.startswith("pfss_")))
        table = tx.feature_table(self.hourly, base, ch.CACHE_DIR / "table_hourly.pkl")
        self.rings = pull_gong.load_ring_table()
        names = list(dict.fromkeys([*self.anchor_pfss, *pfss_columns()]))
        ring_table = windows.window_table(build.Inputs(pfss=self.rings), self.hourly,
                                          [windows.WindowSpec.parse(n) for n in names])
        self.table = table[list(base)].join(ring_table.astype(np.float32))

    def index(self, t0s):
        return self.series.index.get_indexer(t0s)


def score_fold(data, start, end, names, timer, cache=None):
    """{variant: S_final} and persistence's S_final for the fold's 6-hourly runs."""
    cache = cache if cache is not None else {}
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[data.runs6 < start]
    fit_h = data.hourly[data.hourly < start]
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    out = {"persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1), obs,
                                   pers)["s_final"], "runs": len(val)}
    t = time.perf_counter()
    table = data.table.join(wsa_table(data.rings, data.hourly, wsa_params(start.year)))
    med = table.loc[fit6, data.anchor_pfss].median()
    table[data.anchor_pfss] = table[data.anchor_pfss].fillna(med)
    timer["tables"] = timer.get("tables", 0.0) + time.perf_counter() - t

    def x(t0s, cols):
        return table.loc[t0s, list(cols)].to_numpy(dtype=float)

    t = time.perf_counter()
    cfg = data.linear_cfg
    lin = LinearModel(cfg).fit(data.values, data.index(fit6), x(fit6, cfg.input_features),
                               end=start_pos)
    out["linear shipped"] = cs.score(lin.forecast(data.values, data.index(val),
                                                  x(val, cfg.input_features)), obs, pers)["s_final"]
    timer["linear shipped"] = timer.get("linear shipped", 0.0) + time.perf_counter() - t
    for name in names:
        if name == "linear shipped" or name in cache:
            continue
        t = time.perf_counter()
        hcfg = replace(data.hybrid_cfg,
                       features=(*data.hybrid_cfg.features, *extra_columns(name)))
        hyb = XGBoostModel(hcfg, lin)
        hyb.fit(data.values, data.index(fit_h), x(fit_h, hyb.input_features), end=start_pos)
        f = hyb.forecast(data.values, data.index(val), x(val, hyb.input_features))
        out[name] = cs.score(f, obs, pers)["s_final"]
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return out


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s; {len(data.hourly)} hourly fit "
        f"times, {len(data.table.columns)} columns")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, {}
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        year = f"{start:%Y}"
        path = CACHE_DIR / f"{year}.json"
        cached = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        row = dict(cached)
        for name in VARIANTS:  # one variant at a time, saved as it goes
            if name in row:
                continue
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
            t = time.perf_counter()
            row.update(score_fold(data, start, end, (name,), timer, row))
            if not pilot:
                path.write_text(json.dumps(row), encoding="utf-8")
            log(f"  {year} {name}: {row[name] / row['persistence']:.4f} "
                f"({time.perf_counter() - t:.0f} s)")
        rows[year] = row
        if pilot:
            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {tx.peak_mb():.0f} MB; six folds -> ~"
                f"{6 * sum(timer.values()) / 60:.0f} min")
    return summarize(rows, pilot)


def summarize(rows, pilot=False):
    years = sorted(rows)
    ratio = pd.DataFrame({y: {n: rows[y][n] / rows[y]["persistence"] for n in VARIANTS}
                          for y in years})
    out = {}
    for name, r in ratio.iterrows():
        vs = (r / ratio.loc["linear shipped"]).to_numpy()
        vs_h = (r / ratio.loc["hybrid"]).to_numpy()
        gain = float(np.exp(np.log(vs).mean()) - 1)
        won = int((vs < 1).sum())
        out[name] = {"/ persistence": float(np.exp(np.log(r.to_numpy()).mean())),
                     "vs linear": gain, "folds won": won,
                     "vs hybrid": float(np.exp(np.log(vs_h).mean()) - 1),
                     "keep": bool(name.startswith("hybrid") and gain < -KEEP_GAIN
                                  and won >= KEEP_FOLDS)}
    table = pd.DataFrame(out).T.sort_values("/ persistence")
    if not pilot:
        pd.DataFrame([{"fold": y, **rows[y]} for y in years]).to_csv(
            EXPERIMENTS / "hybrid_pfss.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(years)} fold(s) (official S_final / persistence):\n"
            + table.to_string())
        log("by fold:\n" + ratio.round(4).to_string())
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
