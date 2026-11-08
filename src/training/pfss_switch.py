"""PFSS stream-arrival switch: persistence for slow wind, the submitted model otherwise.

    python -m training.pfss_switch --pilot --folds 2024   # one fold, timed, not cached
    python -m training.pfss_switch [--max-seconds N]      # every fold; resumable
    python -m training.pfss_switch --summary

Runs that stay slow are 1.235 of persistence with the submitted linear model
(docs/model_diagnostics.md), and no state term at t0 separates them from slow wind about to
be hit by a stream (docs/slow_wind.md). The PFSS map at t0 also covers the longitudes that
will face Earth after t0, so it might. A classifier gives P(the run stays slow) at t0, and
the forecast moves towards persistence when it is high.

Classifier: logistic regression (standardised, L2 with C = 1, not tuned), per fold, on the
fit runs whose 72 target hours end before the fold start; label: the run regime is "slow
wind" (max observed speed < 450 km/s; training.solar_wind_events). Feature sets:
    insitu   last speed, speed_slope_24h, bt_mean_1h, density_mean_24h, ch_cm_at_0h
    pfss     insitu plus pfss_{d_hcs,log_fs,theta_b,foot_pol,br_ss}_at_{-72..72 by 24}h and
             pfss_open_area_at_0h (negative lags: longitudes facing Earth after t0)
Missing values take the fit runs' median.
Slow-wind models (added after the pilot, user request 2026-11-08): official persistence, or
the submitted config fitted on the slow-wind fit runs only ("slow model").
Switches (f = the submitted model refitted per fold, p = the slow-wind model, P = P(slow)):
    hard tau          p where P > tau, else f (tau = 0.5 .. 0.9; also tau chosen nested)
    soft g            f + P^g (p - f), g = 1, 2
    shrunk nested     f + alpha P (p - f), alpha chosen nested
Nested: on the year before the fold, with the classifier and the linear model fitted on the
runs before that year. Reference rows: the submitted model and the oracle (p on the true
slow runs; the ceiling, -2.8%). Score: official S_final / persistence, geometric mean over
the six yearly folds, per run regime too. Decision rule, set before running: keep if > 1%
better than the submitted model and winning >= 4 of 6 folds. The eval year is not used.
Results: src/experiments/pfss_switch.csv (cache: src/data/pfss_switch_cache/).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from feature_engineering import build, windows
from feature_engineering.recurrence import last_observed
from training import competition_score as cs
from training import model_diagnostics as dg
from training import post_calibration as pc
from training import slow_wind as sw
from training import xgboost_model as tx
from training.folds import YEAR, YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "pfss_switch_cache"
RING = ("d_hcs", "log_fs", "theta_b", "foot_pol", "br_ss")
LAGS = (-72, -48, -24, 0, 24, 48, 72)
INSITU = ("last", "speed_slope_24h", "bt_mean_1h", "density_mean_24h", "ch_cm_at_0h")
PFSS = tuple(f"pfss_{v}_at_{lag}h" for v in RING for lag in LAGS) + ("pfss_open_area_at_0h",)
SETS = {"insitu": INSITU, "pfss": INSITU + PFSS}
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
HOUR = pd.Timedelta(hours=1)


def log(msg):
    print(msg, flush=True)


class Data(sw.Data):
    """slow_wind.Data (the submitted config's table + speed trends) plus the PFSS columns."""

    def __init__(self):
        super().__init__()
        from training import pull_gong

        rings = build.Inputs(pfss=pull_gong.load_ring_table())
        specs = [windows.WindowSpec.parse(n) for n in PFSS if n not in self.table]
        extra = windows.window_table(rings, self.runs6, specs).astype(np.float32)
        self.table = self.table.join(extra)
        self.table["last"] = last_observed(self.values, self.series.index.get_indexer(self.runs6))


def labelled(data, before):
    """Fit runs whose 72 target hours end before `before`, with their slow-wind label."""
    t0s = data.runs6[data.runs6 < before - (cs.HORIZON + 1) * HOUR]
    return t0s, (data.run_regimes(t0s) == "slow wind").astype(int)


def classifier(data, fit_end, names):
    t0s, y = labelled(data, fit_end)
    x = data.table.loc[t0s, list(names)].astype(float)
    med = x.median()
    mu, sd = x.fillna(med).mean(), x.fillna(med).std().replace(0, 1.0)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(((x.fillna(med) - mu) / sd).to_numpy(), y)

    def predict(t0s_new):
        z = (data.table.loc[t0s_new, list(names)].astype(float).fillna(med) - mu) / sd
        return clf.predict_proba(z.to_numpy())[:, 1]

    return predict, float(y.mean())


def as_forecast(pers, f):
    """Persistence as a (runs, 72) forecast; f where persistence is missing."""
    p = np.repeat(np.asarray(pers, dtype=float)[:, None], cs.HORIZON, 1)
    return np.where(np.isfinite(p), p, f)


def switch(kind, f, p, p_slow, param=None):
    """Move from f (the stream model) towards p (the slow-wind model's forecast)."""
    if kind == "hard":
        return np.where((p_slow > param)[:, None], p, f)
    if kind == "soft":
        return f + (p_slow ** param)[:, None] * (p - f)
    return f + param * p_slow[:, None] * (p - f)  # shrunk


def best_param(kind, grid, f, alt, obs, pers, p_slow):
    scores = [cs.score(switch(kind, f, alt, p_slow, g), obs, pers)["s_final"] for g in grid]
    return grid[int(np.argmin(scores))]


def with_fill(data, fit_end):
    table = data.table.copy()
    fit = data.runs6[data.runs6 < fit_end]
    table[data.pfss] = table[data.pfss].fillna(table.loc[fit, data.pfss].median())
    return table


def slow_model_forecasts(data, table, fit_end, t0s):
    """The submitted config fitted on the slow-wind fit runs only (labels before fit_end)."""
    from models.linear_model import LinearModel

    fit, y = labelled(data, fit_end)
    fit = fit[y.astype(bool)]
    cols = list(data.cfg.input_features)
    lin = LinearModel(data.cfg).fit(data.values, data.series.index.get_indexer(fit),
                                    table.loc[fit, cols].to_numpy(dtype=float),
                                    end=data.series.index.get_loc(fit_end))
    return lin.forecast(data.values, data.series.index.get_indexer(t0s),
                        table.loc[t0s, cols].to_numpy(dtype=float))


def score_fold(data, start, end, timer):
    t = time.perf_counter()
    inner_start = start - YEAR
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    inner = data.runs6[(data.runs6 >= inner_start) & (data.runs6 < start)]
    table, table_in = with_fill(data, start), with_fill(data, inner_start)
    ((f, _),) = pc.linear_forecasts(data, table, start, val)
    ((f_in, _),) = pc.linear_forecasts(data, table_in, inner_start, inner)
    s_val = slow_model_forecasts(data, table, start, val)
    s_in = slow_model_forecasts(data, table_in, inner_start, inner)
    timer["linear fits"] = timer.get("linear fits", 0.0) + time.perf_counter() - t
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    obs_in, pers_in = cs.verifying(data.truth, inner), cs.persistence(data.truth, inner)
    regime = data.run_regimes(val)
    slow = regime == "slow wind"
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"]}

    def row(g):
        r = dg.regime_scores(g, obs, pers, regime)
        res = obs - g
        for name in dg.RUN_REGIMES:
            r[f"residual {name}"] = float(np.nanmean(res[regime == name]))
        r["residual all"] = float(np.nanmean(res))
        return r

    alts = {"persistence": (as_forecast(pers, f), as_forecast(pers_in, f_in)),
            "slow model": (s_val, s_in)}
    out["reference"] = row(f)
    out["slow model alone"] = row(s_val)
    for alt_name, (alt, _) in alts.items():
        tag = "" if alt_name == "persistence" else " (slow model)"
        out["oracle" + tag] = row(switch("hard", f, alt, slow.astype(float), 0.5))
    t = time.perf_counter()
    for set_name, names in SETS.items():
        predict, base_rate = classifier(data, start, names)
        p_val = predict(val)
        predict_in, _ = classifier(data, inner_start, names)
        p_in = predict_in(inner)
        y = slow.astype(int)
        out[f"auc {set_name}"] = float(roc_auc_score(y, p_val))
        clim = log_loss(y, np.full(len(y), base_rate), labels=[0, 1])
        out[f"log-loss skill {set_name}"] = float(1 - log_loss(y, p_val, labels=[0, 1]) / clim)
        for name in ("stream arrival", "CME", "slow wind"):
            out[f"P>0.5 share {name} {set_name}"] = float((p_val[regime == name] > 0.5).mean())
        for alt_name, (alt, alt_in) in alts.items():
            tag = set_name if alt_name == "persistence" else f"{set_name}, slow model"
            for tau in TAUS:
                out[f"hard {tau} [{tag}]"] = row(switch("hard", f, alt, p_val, tau))
            tau = best_param("hard", TAUS + (1.01,), f_in, alt_in, obs_in, pers_in, p_in)
            out[f"hard nested [{tag}]"] = row(switch("hard", f, alt, p_val, tau))
            out[f"tau nested {tag}"] = tau
            for g in (1, 2):
                out[f"soft g{g} [{tag}]"] = row(switch("soft", f, alt, p_val, g))
            alpha = best_param("shrunk", ALPHAS, f_in, alt_in, obs_in, pers_in, p_in)
            out[f"shrunk nested [{tag}]"] = row(switch("shrunk", f, alt, p_val, alpha))
            out[f"alpha nested {tag}"] = alpha
    timer["classifiers + switches"] = (timer.get("classifiers + switches", 0.0)
                                       + time.perf_counter() - t)
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
        rows[year] = score_fold(data, start, end, timer)
        if not pilot:
            path.write_text(json.dumps(rows[year]), encoding="utf-8")
        r = rows[year]
        log(f"  {year}: AUC insitu {r['auc insitu']:.3f}, pfss {r['auc pfss']:.3f}; "
            f"log-loss skill {r['log-loss skill insitu']:.3f} / {r['log-loss skill pfss']:.3f}; "
            f"tau {r['tau nested insitu']} / {r['tau nested pfss']}, alpha "
            f"{r['alpha nested insitu']} / {r['alpha nested pfss']} "
            f"({time.perf_counter() - t:.0f} s)")
    if pilot:
        log("  time: " + ", ".join(f"{k} {v:.1f} s" for k, v in timer.items())
            + f"; peak memory {tx.peak_mb():.0f} MB; six folds -> ~"
            f"{6 * sum(timer.values()) / 60:.0f} min after the data")
    return summarize(rows, pilot)


def summarize(rows, pilot=False):
    table = sw.summarize(rows, pilot=True)  # same layout and keep rule
    years = sorted(rows)
    clf = pd.DataFrame({y: {k: v for k, v in rows[y].items()
                            if not isinstance(v, dict) and k not in ("runs", "persistence")}
                        for y in years})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        log("\nclassifier by fold:\n" + clf.to_string())
    if not pilot:
        names = [n for n in rows[years[0]] if isinstance(rows[years[0]][n], dict)]
        pd.DataFrame([{"fold": y, "variant": n, "persistence": rows[y]["persistence"],
                       **rows[y][n]} for n in names for y in years]).to_csv(
            EXPERIMENTS / "pfss_switch.csv", index=False, float_format="%.6g")
        clf.T.rename_axis("fold").to_csv(EXPERIMENTS / "pfss_switch_classifier.csv",
                                         float_format="%.6g")
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
