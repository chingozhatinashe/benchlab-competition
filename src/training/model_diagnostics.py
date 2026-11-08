"""Performance study of the submitted linear model ("pfss-dhcs"): regimes, residuals, importance.

    python -m training.model_diagnostics --pilot --folds 2024   # one fold, timed, not cached
    python -m training.model_diagnostics [--max-seconds N]      # six folds + eval year; resumable
    python -m training.model_diagnostics --summary              # the tables, from the cache

Diagnostic only: nothing is selected or changed from it.

Forecasts
    folds      out-of-fold, on the six yearly folds (training.folds.YEARLY): the submitted
               config refitted on the runs before each fold, as in training.pfss_features
               and training.hybrid_pfss (the anchor's PFSS gaps take the fit runs' median);
               reproduces 0.7794 of persistence
    eval year  the shipped asset (assets/linear_model.json) through the live pipeline
               (missing regressors -> training medians). Descriptive only: the eval year
               is never used to change the model.
    references persistence (official) and the final baseline (refitted per fold; the
               shipped asset on the eval year)
Regimes (training.solar_wind_events, thresholds fixed before looking at errors)
    per run    CME (with possible CME), stream arrival, fast stream (mean >= 550 km/s),
               intermediate, slow wind (max < 450 km/s), from the 72 target hours
    per hour   CME (CME-affected or possible-CME hour), else by the observed speed:
               slow < 450, intermediate 450-550, fast >= 550 km/s
Importance
    drop       per fold, refit without one regressor or recurrence term; change in S_final
    permute    per fold, shuffle one regressor across the fold's runs (all its lag columns
               together, missing values included), N_PERM repeats, no refit
    contrib    each design column's contribution coef * value (km/s: the model is a
               relative Gaussian on the speed itself), per run and lead
Truth: SWPC's hourly ACE file (training.pull_ace_1h); residual = observed - forecast.
Results: src/experiments/model_diagnostics_{regimes,residuals,importance}.csv; the figures
in src/experiments/model_diagnostics.ipynb (cache: src/data/model_diagnostics_cache/).
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build, windows
from feature_engineering.recurrence import last_observed
from models.final_baseline import FinalBaseline
from models.linear_model import LinearModel
from training import compare_history as ch
from training import competition_score as cs
from training import final_baseline as tf
from training import linear_model as tl
from training import solar_wind_events as ev
from training import xgboost_model as tx
from training.datasets import DATA_START, load_hourly, load_train
from training.evaluate import forecast_times
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "model_diagnostics_cache"
EVAL = "eval year"
RUN_REGIMES = tuple(ev.ROUTES)
HOUR_REGIMES = ("CME", "slow", "intermediate", "fast")
SLOW, FAST = 450.0, 550.0
RECURRENCE_DROPS = {"recurrence 13.5 d": (27.0,), "recurrence 27 d": (13.5,),
                    "both recurrence terms": ()}
N_PERM = 5
SEED = 0
MAX_SHIFT = 24
MIN_OVERLAP = 36


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------- pure helpers
def hour_regimes(obs, cme_hours):
    """Per target hour: 'CME' where flagged, else slow / intermediate / fast by the observed
    speed; '' where the observation is missing."""
    obs = np.asarray(obs, dtype=float)
    out = np.select([np.asarray(cme_hours, dtype=bool), ~np.isfinite(obs), obs < SLOW,
                     obs < FAST], ["CME", "", "slow", "intermediate"], "fast")
    return out.astype("<U12")


def best_shift(forecast, observed, max_shift=MAX_SHIFT, min_overlap=MIN_OVERLAP):
    """Per run, the shift s (h) minimising the RMSE of observed[k] against forecast[k - s].

    s > 0: the observed structure comes s hours after the forecast one (the forecast is
    early); s < 0: it is late. NaN where no shift has min_overlap valid pairs.
    """
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(observed, dtype=float)
    n, h = f.shape
    shifts = np.arange(-max_shift, max_shift + 1)
    err = np.full((n, len(shifts)), np.inf)
    for i, s in enumerate(shifts):
        if s >= 0:
            a, b = y[:, s:], f[:, : h - s]
        else:
            a, b = y[:, : h + s], f[:, -s:]
        d2 = (a - b) ** 2
        ok = np.isfinite(d2)
        cnt = ok.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mse = np.where(ok, d2, 0.0).sum(axis=1) / cnt
        err[:, i] = np.where(cnt >= min_overlap, mse, np.inf)
    best = shifts[np.argmin(err, axis=1)].astype(float)
    best[~np.isfinite(err.min(axis=1))] = np.nan
    return best


def regime_scores(f, obs, pers, regime):
    """Official S_final over all runs and within each run regime."""
    out = {"all": cs.score(f, obs, pers)["s_final"]}
    for r in RUN_REGIMES:
        m = np.asarray(regime) == r
        out[r] = cs.score(f[m], obs[m], pers[m])["s_final"] if m.sum() >= 2 else np.nan
    return out


def drop_configs(cfg):
    """{name: config} without one regressor, or without recurrence terms."""
    out = {}
    for f in cfg.features:
        out[f] = replace(cfg, features=tuple(g for g in cfg.features if g != f),
                         optional=tuple(g for g in cfg.optional if g != f))
    for name, lags in RECURRENCE_DROPS.items():
        out[name] = replace(cfg, lags_days=lags)
    return out


def column_names(cfg):
    """Design columns after the intercept: the speed terms, then the regressors."""
    return (["last speed"] + [f"recurrence {d:g} d" for d in cfg.lags_days]
            + list(cfg.features))


def contributions(model, values, positions, regressors):
    """(n, 72, columns) contribution coef * value of each non-intercept design column
    (km/s), as the model forecasts (leads filled, imputed, capped)."""
    raw = model.fill_leads(regressors)
    x, _ = model.design(values, positions, model.cap(model.impute(raw)))
    return (x * model.coef[None])[:, :, 1:].astype(np.float32)


# ---------------------------------------------------------------------- data
class Data:
    """Speed series, truth, run tables and event flags, built once."""

    def __init__(self):
        from training import pull_ace_1h, pull_gong

        self.series = load_train()
        self.values = self.series.to_numpy(dtype=float)
        self.full = load_hourly()  # to the end of the eval year
        self.full_values = self.full.to_numpy(dtype=float)
        self.runs6 = tl.training_runs(self.series)
        self.eval_t0s = forecast_times()
        self.truth = pull_ace_1h.load()
        self.cfg = tl.with_fixes(tl.load_config())
        names = self.cfg.input_features
        self.pfss = [n for n in names if n.startswith("pfss_")]
        base = [n for n in names if n not in self.pfss]
        hourly = tx.feature_table(tx.hourly_runs(self.series), base,
                                  ch.CACHE_DIR / "table_hourly.pkl")
        rings = build.Inputs(pfss=pull_gong.load_ring_table())
        specs = [windows.WindowSpec.parse(n) for n in self.pfss]
        self.table = hourly.loc[self.runs6, base].join(
            windows.window_table(rings, self.runs6, specs).astype(np.float32))
        self.eval_table = tx.feature_table(self.eval_t0s, base).join(
            windows.window_table(rings, self.eval_t0s, specs).astype(np.float32))
        self.cme, self.possible, self.arrivals = self.events()

    def events(self):
        path = CACHE_DIR / "events.pkl"
        if path.exists():
            d = pd.read_pickle(path)
            if d["cme"].index.equals(self.full.index):
                return d["cme"], d["possible"], d["arrivals"]
        t = time.perf_counter()
        cme, possible, arrivals = ev.hourly_events(DATA_START - pd.Timedelta(days=30),
                                                   self.full.index[-1] + pd.Timedelta(hours=1),
                                                   self.full.index)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"cme": cme, "possible": possible, "arrivals": arrivals}, path)
        log(f"events detected in {time.perf_counter() - t:.0f} s")
        return cme, possible, arrivals

    def run_regimes(self, t0s):
        pos = self.full.index.get_indexer(t0s)
        labels = ev.label_runs(pos, self.full_values, self.cme, self.possible, self.arrivals)
        return np.asarray(pd.Series(labels).map(ev.ROUTE_OF), dtype="<U16")

    def cme_hours(self, t0s):
        """(runs, 72) True where the target hour is CME-affected or a possible CME."""
        flag = (self.cme | self.possible).to_numpy()
        idx = self.full.index.get_indexer(t0s)[:, None] + np.arange(1, cs.HORIZON + 1)
        out = np.zeros(idx.shape, dtype=bool)
        inside = idx < len(flag)
        out[inside] = flag[idx[inside]]
        return out


def permutation_scores(model, values, pos, x, obs, pers, regime, rng):
    """{feature: {regime: mean S_final over N_PERM shuffles}} for each regressor."""
    cfg = model.config
    out = {}
    for f in cfg.features:
        cols = np.unique(cfg.lead_columns(f))
        rows = []
        for _ in range(N_PERM):
            xp = x.copy()
            xp[:, cols] = x[rng.permutation(len(x))][:, cols]
            rows.append(regime_scores(model.forecast(values, pos, xp), obs, pers, regime))
        out[f] = pd.DataFrame(rows).mean().to_dict()
    return out


def fold_results(data, start, end, timer, importance=True):
    """Forecasts, metadata and importance scores for one fold's 6-hourly runs."""
    cfg = data.cfg
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[data.runs6 < start]
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    table = data.table.copy()
    imputed = table.loc[val, data.pfss].isna().any(axis=1).to_numpy()
    table[data.pfss] = table[data.pfss].fillna(table.loc[fit6, data.pfss].median())

    def x(t0s, c):
        return table.loc[t0s, list(c)].to_numpy(dtype=float) if c else None

    pos_fit, pos_val = data.series.index.get_indexer(fit6), data.series.index.get_indexer(val)
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    regime = data.run_regimes(val)
    t = time.perf_counter()
    lin = LinearModel(cfg).fit(data.values, pos_fit, x(fit6, cfg.input_features), end=start_pos)
    xv = x(val, cfg.input_features)
    f_lin = lin.forecast(data.values, pos_val, xv)
    base = FinalBaseline(tf.load_config()).fit(data.series.iloc[:start_pos])
    f_base = base.forecast(data.values, pos_val)
    timer["forecasts"] = timer.get("forecasts", 0.0) + time.perf_counter() - t
    out = {"t0s": val, "f_lin": f_lin, "f_base": f_base, "obs": obs, "pers": pers,
           "last": last_observed(data.values, pos_val), "regime": regime,
           "submodel": lin.uses_fallback(xv), "pfss_imputed": imputed,
           "contrib": contributions(lin, data.values, pos_val, xv),
           "columns": column_names(cfg), "x": xv, "x_names": list(cfg.input_features)}
    if not importance:
        return out
    t = time.perf_counter()
    out["drop"] = {}
    for name, c2 in drop_configs(cfg).items():
        m = LinearModel(c2).fit(data.values, pos_fit, x(fit6, c2.input_features), end=start_pos)
        f = m.forecast(data.values, pos_val, x(val, c2.input_features))
        out["drop"][name] = regime_scores(f, obs, pers, regime)
    timer["drop refits"] = timer.get("drop refits", 0.0) + time.perf_counter() - t
    t = time.perf_counter()
    out["permute"] = permutation_scores(lin, data.values, pos_val, xv, obs, pers, regime,
                                        np.random.default_rng(SEED + start.year))
    timer["permutations"] = timer.get("permutations", 0.0) + time.perf_counter() - t
    out["reference"] = regime_scores(f_lin, obs, pers, regime)
    return out


def eval_results(data, timer):
    """The shipped assets on the eval year, through the live imputation."""
    t = time.perf_counter()
    t0s = data.eval_t0s
    pos = data.full.index.get_indexer(t0s)
    lin = LinearModel.load()
    xv = data.eval_table[list(lin.config.input_features)].to_numpy(dtype=float)
    obs, pers = cs.verifying(data.truth, t0s), cs.persistence(data.truth, t0s)
    regime = data.run_regimes(t0s)
    f_lin = lin.forecast(data.full_values, pos, xv)
    out = {"t0s": t0s, "f_lin": f_lin,
           "f_base": FinalBaseline.load().forecast(data.full_values, pos),
           "obs": obs, "pers": pers, "last": last_observed(data.full_values, pos),
           "regime": regime, "submodel": lin.uses_fallback(xv),
           "pfss_imputed": np.isnan(xv[:, [lin.config.input_features.index(c)
                                           for c in data.pfss]]).any(axis=1),
           "contrib": contributions(lin, data.full_values, pos, xv),
           "columns": column_names(lin.config), "x": xv,
           "x_names": list(lin.config.input_features),
           "reference": regime_scores(f_lin, obs, pers, regime)}
    out["permute"] = permutation_scores(lin, data.full_values, pos, xv, obs, pers, regime,
                                        np.random.default_rng(SEED))
    timer["eval year"] = timer.get("eval year", 0.0) + time.perf_counter() - t
    return out


# ---------------------------------------------------------------------- run
def periods(years=None):
    out = [(f"{s:%Y}", s, e) for s, e in YEARLY if years is None or s.year in years]
    if years is None:
        out.append((EVAL, None, None))
    return out


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s; {len(data.runs6)} training "
        f"runs, {len(data.eval_t0s)} eval runs")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer = {}
    for name, start, end in periods(years):
        path = CACHE_DIR / f"{name.replace(' ', '_')}.pkl"
        if path.exists() and not pilot:
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        res = eval_results(data, timer) if start is None else fold_results(data, start, end,
                                                                            timer)
        if not pilot:
            pd.to_pickle(res, path)
        ref = res["reference"]["all"]
        pers = cs.score(np.repeat(res["pers"][:, None], cs.HORIZON, 1), res["obs"],
                        res["pers"])["s_final"]
        log(f"  {name}: linear {ref / pers:.4f} of persistence ({len(res['t0s'])} runs, "
            f"{time.perf_counter() - t:.0f} s)")
    if pilot:
        per_fold = sum(timer.values())
        log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
            + f"; peak memory {tx.peak_mb():.0f} MB; six folds + eval year -> ~"
            f"{7 * per_fold / 60:.0f} min after the data")
        return None
    return summarize()


# ---------------------------------------------------------------------- summary
def load_results():
    """{period: results} for every cached period, in fold order."""
    out = {}
    for name, _, _ in periods():
        path = CACHE_DIR / f"{name.replace(' ', '_')}.pkl"
        if path.exists():
            out[name] = pd.read_pickle(path)
    return out


def pooled(results, folds_only=True):
    """The folds' runs concatenated (one dict of arrays)."""
    keys = [k for k in results if not (folds_only and k == EVAL)]
    out = {}
    for k in ("f_lin", "f_base", "obs", "pers", "last", "regime", "submodel", "pfss_imputed",
              "contrib"):
        out[k] = np.concatenate([results[p][k] for p in keys])
    out["t0s"] = pd.DatetimeIndex(np.concatenate([results[p]["t0s"] for p in keys]))
    out["period"] = np.concatenate([[p] * len(results[p]["t0s"]) for p in keys])
    out["columns"] = results[keys[0]]["columns"]
    return out


def cached_cme_hours(t0s):
    """(runs, 72) CME or possible-CME flags of the target hours, from the cached events."""
    d = pd.read_pickle(CACHE_DIR / "events.pkl")
    flag = (d["cme"] | d["possible"])
    idx = flag.index.get_indexer(pd.DatetimeIndex(t0s))[:, None] + np.arange(1, cs.HORIZON + 1)
    out = np.zeros(idx.shape, dtype=bool)
    inside = (idx >= 0) & (idx < len(flag))
    out[inside] = flag.to_numpy()[idx[inside]]
    return out


def regime_table(results):
    """Per period (each fold, the folds pooled, the eval year) and run regime."""
    sets = {p: r for p, r in results.items()}
    if any(p != EVAL for p in results):
        sets["folds pooled"] = pooled(results)
    rows = []
    for period, r in sets.items():
        reg = np.asarray(r["regime"])
        for regime in ("all", *RUN_REGIMES):
            m = np.ones(len(reg), bool) if regime == "all" else reg == regime
            if m.sum() < 2:
                continue
            obs, pers = r["obs"][m], r["pers"][m]
            p_fc = np.repeat(pers[:, None], cs.HORIZON, 1)
            s = {k: cs.score(f, obs, pers) for k, f in
                 (("linear", r["f_lin"][m]), ("final baseline", r["f_base"][m]),
                  ("persistence", p_fc))}
            run_rmse = {k: np.sqrt(np.nanmean((f - obs) ** 2, axis=1)) for k, f in
                        (("linear", r["f_lin"][m]), ("persistence", p_fc))}
            ok = np.isfinite(run_rmse["linear"]) & np.isfinite(run_rmse["persistence"])
            rows.append({
                "period": period, "regime": regime, "runs": int(m.sum()),
                "share": float(m.mean()),
                **{f"s_final {k}": v["s_final"] for k, v in s.items()},
                "linear / persistence": s["linear"]["s_final"] / s["persistence"]["s_final"],
                "baseline / persistence": (s["final baseline"]["s_final"]
                                           / s["persistence"]["s_final"]),
                "rmse linear": s["linear"]["rmse"], "rmse persistence": s["persistence"]["rmse"],
                "residual linear": -s["linear"]["bias"],  # observed - forecast
                "beats persistence (run rmse)": float(
                    (run_rmse["linear"][ok] < run_rmse["persistence"][ok]).mean()),
            })
    return pd.DataFrame(rows)


def residual_table(results, cme_hours):
    """By period (folds pooled, eval year), hourly regime and lead: n, bias, RMSE, MAE,
    and each regime's share of the squared error and of the hours."""
    rows = []
    sets = {"folds pooled": pooled(results)} if any(p != EVAL for p in results) else {}
    if EVAL in results:
        sets[EVAL] = results[EVAL]
    for period, r in sets.items():
        res = r["obs"] - r["f_lin"]
        hreg = hour_regimes(r["obs"], cme_hours(r["t0s"]))
        sq_total = np.nansum(res**2)
        n_total = np.isfinite(res).sum()
        for regime in ("all", *HOUR_REGIMES):
            m = np.isfinite(res) & ((hreg != "") if regime == "all" else hreg == regime)
            e = np.where(m, res, np.nan)
            with np.errstate(invalid="ignore"):
                n = m.sum(axis=0)
                rows += [{"period": period, "regime": regime, "lead": k + 1, "n": int(n[k]),
                          "residual": float(np.nanmean(e[:, k])) if n[k] else np.nan,
                          "rmse": float(np.sqrt(np.nanmean(e[:, k] ** 2))) if n[k] else np.nan,
                          "mae": float(np.nanmean(np.abs(e[:, k]))) if n[k] else np.nan}
                         for k in range(cs.HORIZON)]
                rows.append({"period": period, "regime": regime, "lead": 0, "n": int(m.sum()),
                             "residual": float(np.nanmean(e)), "rmse": float(np.sqrt(
                                 np.nanmean(e**2))), "mae": float(np.nanmean(np.abs(e))),
                             "share of squared error": float(np.nansum(e**2) / sq_total),
                             "share of hours": float(m.sum() / n_total)})
    return pd.DataFrame(rows)


def importance_table(results):
    """Change in S_final against the full model, per method, input and run regime: the
    geometric mean over the folds, the folds where it hurts, and the eval year."""
    rows = []
    folds = [p for p in results if p != EVAL]
    for method in ("drop", "permute"):
        names = list(results[folds[0]][method]) if folds else []
        for name in names:
            for regime in ("all", *RUN_REGIMES):
                rel = np.array([results[p][method][name][regime] / results[p]["reference"][regime]
                                for p in folds])
                ok = np.isfinite(rel)
                row = {"method": method, "input": name, "regime": regime,
                       "change (folds)": float(np.exp(np.log(rel[ok]).mean()) - 1),
                       "folds worse": int((rel[ok] > 1).sum()), "folds": int(ok.sum()),
                       **{f"change {p}": float(v - 1) for p, v in zip(folds, rel, strict=True)}}
                if method == "permute" and EVAL in results:
                    e = results[EVAL]
                    row[f"change {EVAL}"] = float(e[method][name][regime]
                                                  / e["reference"][regime] - 1)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize():
    results = load_results()
    if not results:
        log("nothing cached yet")
        return None
    regimes = regime_table(results)
    residuals = residual_table(results, cached_cme_hours)
    importance = importance_table(results)
    for name, df in (("regimes", regimes), ("residuals", residuals),
                     ("importance", importance)):
        df.to_csv(EXPERIMENTS / f"model_diagnostics_{name}.csv", index=False,
                  float_format="%.6g")
    folds = [p for p in results if p != EVAL]
    ratios = regimes[(regimes["regime"] == "all") & regimes["period"].isin(folds)]
    gmean = float(np.exp(np.log(ratios["linear / persistence"]).mean()))
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 30):
        log(f"linear / persistence, geometric mean over {len(folds)} folds: {gmean:.4f}")
        log("\nby run regime:\n" + regimes[regimes["period"].isin(["folds pooled", EVAL])]
            .drop(columns=["period"]).to_string(index=False))
        log("\nresidual by hourly regime (observed - forecast, km/s):\n" + residuals[
            residuals["lead"] == 0].drop(columns=["lead"]).to_string(index=False))
        log("\nimportance (all runs):\n" + importance[importance["regime"] == "all"][
            ["method", "input", "change (folds)", "folds worse"]
            + ([f"change {EVAL}"] if f"change {EVAL}" in importance else [])]
            .to_string(index=False))
    return regimes, residuals, importance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        summarize()
    else:
        run(args.folds, args.pilot, args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
