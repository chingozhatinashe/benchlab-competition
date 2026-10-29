"""The Gaussian-process correction on the six yearly folds (docs/gaussian_process.md).

    python -m training.gaussian_process_folds --pilot --folds 2024   # one fold, timed
    python -m training.gaussian_process_folds [--max-seconds 240]    # every fold; resumable
    python -m training.gaussian_process_folds --summary
    python -m training.gaussian_process_folds --metric run_timse ...  # the 2026-10-23 study

The GP was first tried on the 4-year history and two folds (training.gaussian_process,
2026-10-22). This re-scores its two versions on the current setup: 9 years, the
six yearly walk-forward folds (training.folds.YEARLY), and the shipped 9-year
linear model as anchor.

    ml2      models.gaussian_process as designed: 3 principal residual shapes, one GP
             per shape score, hyper-parameters by marginal likelihood (ML-II)
    offset   one GP on each run's TIMSE-weighted mean offset (observed - linear), with an
             isotropic lengthscale and a noise/signal ratio from a grid, chosen on the
             folds rather than by marginal likelihood

ml2 is scored on two folds only (ML2_FOLDS): its hyper-parameter search takes
about 7 minutes per fold, and it was far behind on the 4-year folds already.

Two input sets: A, the linear model's 4 regressors; H, the hybrid's 30. Both add the
last speed and the linear model's mean offset from persistence.

The GP conditions on at most MAX_FIT_RUNS random 6-hourly fit runs per fold: the
full 9-year set (up to 11,000 runs) needs more memory than the dev laptop has. The
linear anchor is fitted on every fit run. References: the shipped linear model and
hybrid refitted per fold (as training.compare_history). Score: per-run mean TIMSE /
persistence; summaries are geometric means over folds. Results:
src/experiments/gaussian_process_folds.csv; analysis:
src/experiments/gaussian_process.ipynb.

--metric official (the default since 2026-10-29): the offset GP only (ml2 overfits and
takes ~7 min per fold), on the submitted linear model (official loss weights,
linear_model_config.json), which also sets the offset's weights. Scored with the official
S_final (training.competition_score) against the official truth (training.pull_ace_1h);
the reference is the linear model alone (the hybrid is no longer submitted). The grid is
the same; its best point is picked on the same folds, which flatters it. Decision rule,
set before running: more than 1% better than the linear model, winning at least 4 of 6
folds. Results: src/experiments/gaussian_process_official.csv (cache: gp_cache/official).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import targets
from models.gaussian_process import GP, Config, GaussianProcessModel
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import competition_score as cs
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "gp_cache" / "yearly"
OFFICIAL_CACHE = CACHE_DIR.with_name("official")
METRICS = ("official", "run_timse")
KEEP_GAIN, KEEP_FOLDS = 0.01, 4  # official: more than 1% better, at least 4 of 6 folds
MAX_FIT_RUNS = 4000
LENGTHSCALES = (1.0, 3.0, 10.0, 30.0)
NOISE_RATIOS = (1.0, 10.0, 100.0)
ML2_COMPONENTS = 3
ML2_FOLDS = (2019, 2024)  # ML-II takes ~7 min per fold; scored on these folds only


def log(msg):
    print(msg, flush=True)


def _x(data, t0s, names):
    return data.table.loc[t0s, list(names)].to_numpy(dtype=float) if names else None


def offset_inputs(model, data, end, fit_pos, x_fit, val_pos, x_val):
    """Training offsets, weights and inputs, and validation inputs and anchor."""
    lin = model.base_model
    anchor = model.anchor(data.values[:end], fit_pos, x_fit)
    y = targets(lin._clean(data.values[:end]), fit_pos, end)
    q = lin.loss_weights(y)
    qq = np.where(np.isfinite(y), q, 0.0)
    r = np.where(qq > 0, y - anchor, 0.0)
    w = qq.sum(axis=1)
    keep = w > 0
    c = (qq * r).sum(axis=1)[keep] / w[keep]
    wn = w[keep] / w[keep].mean()
    raw = model.raw_inputs(data.values[:end], fit_pos, x_fit, anchor)
    model.x_mean = np.nanmean(raw, axis=0)
    sd = np.nanstd(raw, axis=0)
    model.x_std = np.where(sd > 1e-12, sd, 1.0)
    med = np.nanmedian((raw - model.x_mean) / model.x_std, axis=0)
    model.x_median = np.where(np.isfinite(med), med, 0.0)
    x = model.inputs(raw)[keep]
    a_val = model.anchor(data.values, val_pos, x_val)
    xv = model.inputs(model.raw_inputs(data.values, val_pos, x_val, a_val))
    return x, c, wn, xv, a_val


def score_fold(data, start, end, timer, with_ml2=True):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]

    def clock(name, t):
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t

    out = {}
    t = time.perf_counter()
    lin = LinearModel(data.linear_cfg).fit(
        data.values, data.series.index.get_indexer(fit6),
        _x(data, fit6, data.linear_cfg.input_features), end=start_pos)
    out["linear"] = data.ratio(fold, lin.forecast(
        data.values, fold["pos"], _x(data, fold["t0s"], data.linear_cfg.input_features)))
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    hyb = XGBoostModel(data.hybrid_cfg, lin)
    hyb.fit(data.values, data.series.index.get_indexer(fit_h),
            _x(data, fit_h, hyb.input_features), end=start_pos)
    out["hybrid"] = data.ratio(fold, hyb.forecast(
        data.values, fold["pos"], _x(data, fold["t0s"], hyb.input_features)))
    clock("reference models", t)

    rng = np.random.default_rng(0)
    gp_fit = fit6 if len(fit6) <= MAX_FIT_RUNS else pd.DatetimeIndex(
        np.sort(rng.choice(fit6.to_numpy(), MAX_FIT_RUNS, replace=False)), tz="UTC")
    gp_pos = data.series.index.get_indexer(gp_fit)
    out["gp fit runs"] = int(len(gp_fit))
    sets = {"A": tuple(data.linear_cfg.input_features), "H": tuple(data.hybrid_cfg.features)}
    for name, feats in sets.items():
        model = GaussianProcessModel(Config(features=feats, n_components=1, max_components=1),
                                     lin)
        x_fit = _x(data, gp_fit, model.input_features)
        x_val = _x(data, fold["t0s"], model.input_features)
        t = time.perf_counter()
        x, c, wn, xv, a_val = offset_inputs(model, data, start_pos, gp_pos, x_fit,
                                            fold["pos"], x_val)
        var = float(np.average(c**2, weights=wn))
        for ls in LENGTHSCALES:
            for nr in NOISE_RATIOS:
                gp = GP(np.full(x.shape[1], ls), var, var * nr).condition(x, c, wn)
                f = np.clip(a_val + gp.predict(xv)[:, None], 200, 1500)
                out[f"offset {name} ls={ls:g} nr={nr:g}"] = data.ratio(fold, f)
        clock("offset grid", t)
        if not with_ml2:
            continue
        t = time.perf_counter()
        ml2 = GaussianProcessModel(Config(features=feats, n_components=ML2_COMPONENTS,
                                          max_components=ML2_COMPONENTS), lin)
        ml2.fit(data.values, gp_pos, x_fit, end=start_pos)
        out[f"ml2 {name}"] = data.ratio(fold, ml2.forecast(data.values, fold["pos"], x_val))
        out[f"ml2 {name} lengthscales"] = [gp.lengthscales.round(3).tolist() for gp in ml2.gps]
        out[f"ml2 {name} noise"] = [{"sf2": gp.sf2, "sn2": gp.sn2} for gp in ml2.gps]
        clock("ml2", t)
    return out


def score_fold_official(data, truth, start, end, timer):
    """The offset GP on the submitted linear model, scored with the official metric (km/s)."""
    fold = data.fold(start, end)
    t0s = fold["t0s"]
    obs, pers = cs.verifying(truth, t0s), cs.persistence(truth, t0s)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]

    def official(f):
        return cs.score(f, obs, pers)["s_final"]

    out = {"persistence": official(np.repeat(pers[:, None], cs.HORIZON, 1))}
    t = time.perf_counter()
    lin = LinearModel(data.linear_cfg).fit(
        data.values, data.series.index.get_indexer(fit6),
        _x(data, fit6, data.linear_cfg.input_features), end=start_pos)
    out["linear"] = official(lin.forecast(
        data.values, fold["pos"], _x(data, t0s, data.linear_cfg.input_features)))
    timer["linear"] = timer.get("linear", 0.0) + time.perf_counter() - t

    rng = np.random.default_rng(0)
    gp_fit = fit6 if len(fit6) <= MAX_FIT_RUNS else pd.DatetimeIndex(
        np.sort(rng.choice(fit6.to_numpy(), MAX_FIT_RUNS, replace=False)), tz="UTC")
    gp_pos = data.series.index.get_indexer(gp_fit)
    out["gp fit runs"] = int(len(gp_fit))
    sets = {"A": tuple(data.linear_cfg.input_features), "H": tuple(data.hybrid_cfg.features)}
    for name, feats in sets.items():
        model = GaussianProcessModel(Config(features=feats, n_components=1, max_components=1),
                                     lin)
        x_fit = _x(data, gp_fit, model.input_features)
        x_val = _x(data, t0s, model.input_features)
        t = time.perf_counter()
        x, c, wn, xv, a_val = offset_inputs(model, data, start_pos, gp_pos, x_fit,
                                            fold["pos"], x_val)
        var = float(np.average(c**2, weights=wn))
        for ls in LENGTHSCALES:
            for nr in NOISE_RATIOS:
                gp = GP(np.full(x.shape[1], ls), var, var * nr).condition(x, c, wn)
                corr = gp.predict(xv)
                f = np.clip(a_val + corr[:, None], 200, 1500)
                key = f"offset {name} ls={ls:g} nr={nr:g}"
                out[key] = official(f)
                out[f"{key} | mean correction"] = float(np.nanmean(corr))
        timer["offset grid"] = timer.get("offset grid", 0.0) + time.perf_counter() - t
    return out


def summarize_official(rows):
    df = scores_frame(rows)
    df.reset_index().to_csv(EXPERIMENTS / "gaussian_process_official.csv", index=False,
                            float_format="%.6g")
    cols = [c for c in df.columns if c == "linear" or (c.startswith("offset") and "|" not in c)]
    out = {}
    for c in cols:
        ratio = df[c] / df["linear"]
        gain = float(np.exp(np.log(ratio).mean()) - 1)
        won = int((df[c] < df["linear"]).sum())
        out[c] = {"S_final": float(df[c].mean()), "vs linear": gain, "folds won": won,
                  "/ persistence": float(np.exp(np.log(df[c] / df["persistence"]).mean())),
                  "mean correction": float(df[f"{c} | mean correction"].mean())
                  if f"{c} | mean correction" in df else 0.0,
                  "keep": bool(c != "linear" and gain < -KEEP_GAIN and won >= KEEP_FOLDS)}
    table = pd.DataFrame(out).T.sort_values("S_final")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(df)} folds (official truth; S in km/s), best 12:\n"
            + table.head(12).to_string())
        best = table.index[0] if table.index[0] != "linear" else table.index[1]
        log(f"\nby fold, linear vs {best}:\n" + df[["persistence", "linear", best]].to_string())
    return df


def scores_frame(rows):
    return pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (float, str))}
                         for r in rows]).set_index("fold")


def summarize(rows):
    df = scores_frame(rows)
    df.reset_index().to_csv(EXPERIMENTS / "gaussian_process_folds.csv", index=False,
                            float_format="%.6g")
    cols = [c for c in df.columns if c != "seconds"]
    geo = pd.Series({c: float(np.exp(np.log(df[c].dropna()).mean())) for c in cols})
    summary = pd.DataFrame({
        "fold score": geo,
        "vs linear": geo / geo["linear"] - 1,
        "vs hybrid": geo / geo["hybrid"] - 1,
        "folds": {c: int(df[c].notna().sum()) for c in cols},
        "folds won vs linear": {c: int((df[c] < df["linear"]).sum()) for c in cols},
    }).sort_values("fold score")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(df)} folds (best 12):\n{summary.head(12).to_string()}")
        ml2 = [c for c in cols if c.startswith("ml2")]
        if ml2:
            log("\nml2 (its folds only; compare with those folds' linear and hybrid):\n"
                + df[["linear", "hybrid", *ml2]].dropna().to_string())
    return df


def run(years=None, pilot=False, max_seconds=None, metric="official"):
    started = time.perf_counter()
    official = metric == "official"
    data = ch.Data()
    truth = None
    if official:
        from training import pull_ace_1h

        truth = pull_ace_1h.load()
    log(f"data ready in {time.perf_counter() - started:.0f} s (metric {metric})")
    cache = OFFICIAL_CACHE if official else CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = cache / f"{start:%Y}.json"
        if path.exists() and not pilot:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        if official:
            r = {"fold": f"{start:%Y-%m}", **score_fold_official(data, truth, start, end, timer)}
            r["seconds"] = time.perf_counter() - t
            grid = [k for k in r if k.startswith("offset") and "|" not in k]
            best = min(grid, key=r.get)
            log(f"  fold {r['fold']} ({r['seconds']:.0f} s), S_final km/s: linear "
                f"{r['linear']:.2f}, best {best} {r[best]:.2f} "
                f"({r[best] / r['linear'] - 1:+.2%})")
            if pilot:
                from training.xgboost_model import peak_mb

                log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                    + f"; peak memory {peak_mb():.0f} MB")
                log("  grid: " + ", ".join(f"{k.removeprefix('offset ')} {r[k]:.2f}"
                                           for k in sorted(grid, key=r.get)[:6]))
                log(f"  six folds -> ~{6 * r['seconds'] / 60:.0f} min (earlier folds fit on less)")
                return r
            path.write_text(json.dumps(r), encoding="utf-8")
            rows.append(r)
            continue
        r = {"fold": f"{start:%Y-%m}",
             **score_fold(data, start, end, timer, start.year in ML2_FOLDS)}
        r["seconds"] = time.perf_counter() - t
        best = min((k for k, v in r.items() if k.startswith("offset")), key=r.get)
        ml2 = ", ".join(f"ml2 {n} {r[f'ml2 {n}']:.4f}" for n in "AH" if f"ml2 {n}" in r)
        log(f"  fold {r['fold']} ({r['seconds']:.0f} s): linear {r['linear']:.4f}, "
            f"hybrid {r['hybrid']:.4f}, best offset {best} {r[best]:.4f}"
            + (f", {ml2}" if ml2 else ""))
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB")
            log(f"  six folds -> ~{6 * r['seconds'] / 60:.0f} min (earlier folds fit on less)")
            return r
        path.write_text(json.dumps(r), encoding="utf-8")
        rows.append(r)
    return summarize_official(rows) if official else summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--metric", choices=METRICS, default="official")
    args = parser.parse_args(argv)
    if args.summary:
        official = args.metric == "official"
        rows = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((OFFICIAL_CACHE if official else CACHE_DIR).glob("*.json"))]
        (summarize_official if official else summarize)(rows)
    else:
        run(args.folds, args.pilot, args.max_seconds, args.metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
