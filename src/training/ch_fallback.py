"""Fallbacks for a missing coronal-hole strip, on the yearly folds (official metric).

    python -m training.ch_fallback --pilot --folds 2024   # one fold, timed, not cached
    python -m training.ch_fallback [--max-seconds N]      # every fold; resumable
    python -m training.ch_fallback --summary

The submitted linear model reads the coronal-hole strip of the AIA image about 72 h before
each forecast hour (ch_cm_lead_72h, 13 lagged columns). When every lag is missing, the run
goes to a submodel without the strip (Config.optional). The SDO browse archive has holes
(0 images in 2024-12, 2025-10 and 2025-11; part of 2025-01, 2025-09, 2025-12 and 2026-03),
so 9-16% of the 2024-25 runs and 20% of the eval year's use the submodel; live, the strip
was present in every staging cycle since 2026-10-26. This asks which fallback is better
than the submodel, for the history and for a future outage.

Variants (each fills the strip's missing lagged values, in fit and forecast, before the
model sees them; anything still missing goes to the submodel as before):
    reference      as submitted: images up to 12 h older (windows.CH_FALLBACK_HOURS)
    older 48 h     images up to 48 h older than the lag
    rotation       the strip one solar rotation (27 d = 648 h) earlier, same lag
    pfss proxy     the strip estimated from the PFSS ring at the same Earth-facing longitude
                   (ridge on the ring signals and the open area, fitted on the fit runs)
    older + rotation, older + rotation + proxy   in that order
Score: official S_final / persistence, geometric mean over folds; also on the "affected"
runs (those the reference forecasts with the submodel) and per run regime. Decision rule,
set before running: the affected runs exist in 3 folds only (2022, 2023, 2024), so 4-of-6
cannot be met; a variant goes forward to the combination step (docs/model_diagnostics.md
follow-ups) if it improves the affected runs by more than 2% in every fold that has them
and doesn't make the six-fold score worse. The eval year is not used.
Results: src/experiments/ch_fallback.csv (cache: src/data/ch_fallback_cache/).
"""

import argparse
import json
import sys
import time
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
CACHE_DIR = HERE.parent / "data" / "ch_fallback_cache"
OLDER = (6, 12, 18, 24, 30, 36, 42, 48)
ROTATION = 648
PROXY_VARS = ("log_fs", "theta_b", "foot_lat", "foot_pol", "br_ss", "d_hcs")
PROXY_RIDGE = 1.0
VARIANTS = ("reference", "older 48 h", "rotation", "pfss proxy", "older + rotation",
            "older + rotation + proxy")
GAIN, MIN_FOLDS_AFFECTED = 0.02, 3


def log(msg):
    print(msg, flush=True)


class Data(dg.Data):
    """model_diagnostics.Data plus the strip's fallback columns and the proxy inputs."""

    def __init__(self):
        super().__init__()
        from training import pull_gong

        self.ch_cols = [n for n in self.cfg.input_features if n.startswith("ch_cm_at_")]
        self.lags = [windows.WindowSpec.parse(n).hours for n in self.ch_cols]
        holes = build.Inputs(coronal_holes=pf.load_coronal_holes())
        specs = [windows.WindowSpec.parse(n) for n in self.ch_cols]
        self.older = windows.window_table(holes, self.runs6, specs, ch_fallback=OLDER)
        rot = [windows.WindowSpec("ch_cm", "at", h + ROTATION) for h in self.lags]
        self.rotation = windows.window_table(holes, self.runs6, rot).set_axis(
            self.ch_cols, axis=1)
        rings = build.Inputs(pfss=pull_gong.load_ring_table())
        pspecs = [windows.WindowSpec(f"pfss_{v}", "at", h) for h in self.lags
                  for v in PROXY_VARS]
        pspecs.append(windows.WindowSpec("pfss_open_area", "at", 0))
        self.proxy_inputs = windows.window_table(rings, self.runs6, pspecs)

    def proxy_matrix(self, t0s, lag):
        cols = [f"pfss_{v}_at_{lag}h" for v in PROXY_VARS] + ["pfss_open_area_at_0h"]
        return self.proxy_inputs.loc[t0s, cols].to_numpy(dtype=float)


def fit_proxy(data, ch, fit6):
    """Ridge of the strip on the ring signals at the same lag, pooled over lags."""
    xs, ys = [], []
    for col, lag in zip(data.ch_cols, data.lags, strict=True):
        x = data.proxy_matrix(fit6, lag)
        y = ch.loc[fit6, col].to_numpy(dtype=float)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        xs.append(x[ok])
        ys.append(y[ok])
    x, y = np.vstack(xs), np.concatenate(ys)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    penalty = PROXY_RIDGE * len(z) * np.eye(z.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    resid = y - z @ coef
    return {"mu": mu, "sd": sd, "coef": coef, "r2": float(1 - resid.var() / y.var()),
            "rows": len(y)}


def apply_proxy(data, ch, proxy):
    out = ch.copy()
    for col, lag in zip(data.ch_cols, data.lags, strict=True):
        x = data.proxy_matrix(out.index, lag)
        z = np.column_stack([np.ones(len(x)), (x - proxy["mu"]) / proxy["sd"]])
        est = z @ proxy["coef"]
        gap = out[col].isna().to_numpy() & np.isfinite(est)
        out.loc[gap, col] = est[gap]
    return out


def strip(data, variant, fit6):
    """The strip's lagged columns (runs6 x 13) for a variant."""
    ch = data.table[data.ch_cols].astype(float)
    proxy = None
    if variant in ("older 48 h", "older + rotation", "older + rotation + proxy"):
        ch = ch.fillna(data.older)
    if variant in ("rotation", "older + rotation", "older + rotation + proxy"):
        ch = ch.fillna(data.rotation)
    if variant in ("pfss proxy", "older + rotation + proxy"):
        proxy = fit_proxy(data, ch, fit6)
        ch = apply_proxy(data, ch, proxy)
    return ch, proxy


def score_fold(data, start, end, names, timer):
    cfg = data.cfg
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[data.runs6 < start]
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    regime = data.run_regimes(val)
    base = data.table.copy()
    base[data.pfss] = base[data.pfss].fillna(base.loc[fit6, data.pfss].median())
    pos_fit, pos_val = data.series.index.get_indexer(fit6), data.series.index.get_indexer(val)
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"]}
    affected = None
    for name in names:
        t = time.perf_counter()
        table = base.copy()
        ch, proxy = strip(data, name, fit6)
        table[data.ch_cols] = ch
        x = table[list(cfg.input_features)].to_numpy(dtype=float)
        xf, xv = x[data.runs6.get_indexer(fit6)], x[data.runs6.get_indexer(val)]
        lin = LinearModel(cfg).fit(data.values, pos_fit, xf, end=start_pos)
        f = lin.forecast(data.values, pos_val, xv)
        sub = lin.uses_fallback(xv)
        if name == "reference":
            affected = sub
            out["affected runs"] = int(sub.sum())
        row = {"all": cs.score(f, obs, pers)["s_final"], "submodel runs": int(sub.sum()),
               "fit runs on the full model": lin.fit_info["runs"]}
        if affected is not None and affected.sum() >= 2:
            row["affected"] = cs.score(f[affected], obs[affected], pers[affected])["s_final"]
        for r in dg.RUN_REGIMES:
            m = regime == r
            row[r] = cs.score(f[m], obs[m], pers[m])["s_final"] if m.sum() >= 2 else np.nan
        if proxy is not None:
            row["proxy r2"] = proxy["r2"]
        out[name] = row
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return out


def run(years=None, pilot=False, max_seconds=None):
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
        if path.exists() and not pilot:
            rows[year] = json.loads(path.read_text("utf-8"))
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        row = score_fold(data, start, end, VARIANTS, timer)
        if not pilot:
            path.write_text(json.dumps(row), encoding="utf-8")
        rows[year] = row
        log(f"  {year}: {row['affected runs']} affected runs; "
            + ", ".join(f"{n} {row[n]['all'] / row['persistence']:.4f}" for n in VARIANTS)
            + f" ({time.perf_counter() - t:.0f} s)")
    if pilot:
        log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
            + f"; peak memory {tx.peak_mb():.0f} MB; six folds -> ~"
            f"{6 * sum(timer.values()) / 60:.0f} min after the data")
    return summarize(rows, pilot)


def summarize(rows, pilot=False):
    years = sorted(rows)
    out, long = {}, []
    for name in VARIANTS:
        rel = np.array([rows[y][name]["all"] / rows[y]["reference"]["all"] for y in years])
        aff = {y: rows[y][name]["affected"] / rows[y]["reference"]["affected"]
               for y in years if "affected" in rows[y]["reference"]}
        a = np.array(list(aff.values()))
        out[name] = {
            "/ persistence": float(np.exp(np.mean([np.log(rows[y][name]["all"]
                                                          / rows[y]["persistence"])
                                                   for y in years]))),
            "vs reference": float(np.exp(np.log(rel).mean()) - 1),
            "folds won": int((rel < 1).sum()),
            "affected vs reference": float(np.exp(np.log(a).mean()) - 1) if len(a) else np.nan,
            "affected folds won by 2%": int((a < 1 - GAIN).sum()),
            "submodel runs": int(sum(rows[y][name]["submodel runs"] for y in years)),
        }
        out[name]["forward"] = bool(name != "reference" and len(a) >= MIN_FOLDS_AFFECTED
                                    and (a < 1 - GAIN).all() and out[name]["vs reference"] <= 0)
        for y in years:
            long.append({"fold": y, "variant": name, "persistence": rows[y]["persistence"],
                         **{k: v for k, v in rows[y][name].items()}})
    table = pd.DataFrame(out).T
    if not pilot:
        pd.DataFrame(long).to_csv(EXPERIMENTS / "ch_fallback.csv", index=False,
                                  float_format="%.6g")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(years)} fold(s):\n{table.to_string()}")
        log("affected runs by fold (S_final / reference):\n" + pd.DataFrame(
            {y: {n: rows[y][n]["affected"] / rows[y]["reference"]["affected"]
                 for n in VARIANTS} for y in years if "affected" in rows[y]["reference"]})
            .round(4).to_string())
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
