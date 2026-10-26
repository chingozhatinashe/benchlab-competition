"""Re-tune the linear model and the hybrid for the official metric, on the yearly folds.

    python -m training.official_retune --pilot --folds 2024   # one fold, timed, not cached
    python -m training.official_retune [--max-seconds 480]    # every fold; resumable
    python -m training.official_retune --summary
    python -m training.official_retune --eval-year --old-linear OLD.json --old-baseline OLD.json

training.official_rescore found the shipped models upward-biased (+55 km/s on average,
+95 km/s at 72 h). Their loss weights are each run's own TIMSE weights
(models.final_baseline.run_timse_weights), which favour over-forecasting. The official
metric (training.competition_score) counts RMSE as much as the tail, per lead over all runs.

Variants (each fold fits on 2016-10 .. fold start, as training.compare_history; the
features and every other setting are the shipped ones):
    shipped             run_timse weights (the reference)
    shipped - bias      the shipped models minus their per-lead mean in-sample error on
                        the fit runs (a post-step; nothing refitted)
    official b=<blend>  both models refitted with weights="official": per lead,
                        (1 - blend) + blend * the percentile-TIMSE weights of the official
                        metric over all runs at that lead (models.final_baseline.
                        official_weights). blend 0 = plain squared error; 1 = the TIMSE term
                        only. The hybrid's trees correct the linear model of the same blend.
    final baseline tp=<p>  the last model fallback (models.final_baseline) with its loss
                        weights raised to tail_power p: 1 is shipped, 0 is plain least squares
Score: the official S_final against the official truth (training.pull_ace_1h); the summary
is the geometric mean over folds of S_final / the shipped model's. Decision rule (set
before running): more than 1% better than shipped, winning at least 4 of 6 folds. The same
table settles hybrid vs linear for the November upload.
Results: src/experiments/official_retune.csv.

--eval-year scores the trained assets once on the held-out eval year (2025-10 .. 2026-10,
never used for choices), official metric and truth: the new linear model and final
baseline (src/assets/), the hybrid (its asset embeds its own linear anchor), the previous
assets given by path, and the spec baselines. Results: src/experiments/official_eval_year.csv.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import targets
from models.final_baseline import FinalBaseline
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training import competition_score as cs
from training import final_baseline as tf
from training import pull_ace_1h
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "official_retune_cache"
BLENDS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
VARIANTS = ("shipped", "shipped - bias", *(f"official b={b:g}" for b in BLENDS))
TAIL_POWERS = (1.0, 0.5, 0.25, 0.0)
BASELINE_VARIANTS = tuple(f"final baseline tp={p:g}" for p in TAIL_POWERS)
MODELS = ("linear", "hybrid")
STATS = ("s_final", "rmse", "trmse", "bias", "s_49_72")
LEAD_BIAS = (0, 23, 71)  # leads (1, 24, 72 h) whose bias is reported
_SHIPPED = {}  # fold start -> the shipped (linear, hybrid, fit runs), shared by two variants


def log(msg):
    print(msg, flush=True)


def fit_pair(data, start, linear_cfg, hybrid_cfg):
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    lin = LinearModel(linear_cfg).fit(
        data.values, data.series.index.get_indexer(fit6),
        data.x(fit6, linear_cfg.input_features), end=start_pos)
    hyb = XGBoostModel(hybrid_cfg, lin)
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    hyb.fit(data.values, data.series.index.get_indexer(fit_h),
            data.x(fit_h, hyb.input_features), end=start_pos)
    return lin, hyb, fit6


def predict(data, lin, hyb, t0s):
    pos = data.series.index.get_indexer(t0s)
    return {"linear": lin.forecast(data.values, pos, data.x(t0s, lin.config.input_features)),
            "hybrid": hyb.forecast(data.values, pos, data.x(t0s, hyb.input_features))}


def score_fold(data, truth, start, end, timer, names):
    fold = data.fold(start, end)
    t0s = fold["t0s"]
    obs, pers = cs.verifying(truth, t0s), cs.persistence(truth, t0s)
    start_pos = data.series.index.get_loc(start)
    row = {"fold": f"{start:%Y-%m}", "runs": len(t0s)}
    row["persistence | s_final"] = cs.score(np.repeat(pers[:, None], cs.HORIZON, 1), obs,
                                            pers)["s_final"]

    def record(name, forecasts):
        for model, f in forecasts.items():
            s = cs.score(f, obs, pers)
            for k in STATS:
                row[f"{name} | {model} | {k}"] = s[k]
            for j in LEAD_BIAS:
                row[f"{name} | {model} | bias {j + 1}h"] = s["curves"]["bias"][j]

    for name in names:
        t = time.perf_counter()
        if name in BASELINE_VARIANTS:
            power = float(name.split("=")[1])
            base = FinalBaseline(replace(tf.load_config(), tail_power=power)).fit(
                data.series.iloc[:start_pos])
            s = cs.score(base.forecast(data.values, fold["pos"]), obs, pers)
            for k in STATS:
                row[f"{name} | {k}"] = s[k]
            timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
            continue
        if name in ("shipped", "shipped - bias"):
            if start not in _SHIPPED:
                _SHIPPED.clear()
                _SHIPPED[start] = fit_pair(data, start, data.linear_cfg, data.hybrid_cfg)
            lin, hyb, fit6 = _SHIPPED[start]
            forecasts = predict(data, lin, hyb, t0s)
            if name == "shipped - bias":  # per-lead mean in-sample error on the fit runs
                y_fit = targets(data.values, data.series.index.get_indexer(fit6), start_pos)
                for model, f_fit in predict(data, lin, hyb, fit6).items():
                    bias = np.nanmean(f_fit - y_fit, axis=0)
                    forecasts[model] = forecasts[model] - bias
        else:
            blend = float(name.split("=")[1])
            lin_cfg = replace(data.linear_cfg, weights="official", blend=blend)
            hyb_cfg = replace(data.hybrid_cfg, weights="official", blend=blend)
            lin, hyb, _ = fit_pair(data, start, lin_cfg, hyb_cfg)
            forecasts = predict(data, lin, hyb, t0s)
        record(name, forecasts)
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
    return row


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "official_retune.csv", index=False,
                            float_format="%.6g")
    out = {}
    for model in MODELS:
        ref = df[f"shipped | {model} | s_final"]
        for name in VARIANTS:
            if f"{name} | {model} | s_final" not in df:
                continue
            s = df[f"{name} | {model} | s_final"]
            out[(model, name)] = {
                "S_final": float(s.mean()),
                "vs shipped": float(np.exp(np.log(s / ref).mean()) - 1),
                "folds won": int((s < ref).sum()),
                "/ persistence": float(np.exp(np.log(s / df["persistence | s_final"]).mean())),
                **{k: float(df[f"{name} | {model} | {k}"].mean())
                   for k in ("rmse", "trmse", "bias", "s_49_72")},
                **{f"bias {j + 1}h": float(df[f"{name} | {model} | bias {j + 1}h"].mean())
                   for j in LEAD_BIAS}}
    ref = df.get(f"{BASELINE_VARIANTS[0]} | s_final")
    for name in BASELINE_VARIANTS:
        if f"{name} | s_final" not in df:
            continue
        s = df[f"{name} | s_final"]
        out[("final baseline", name.removeprefix("final baseline "))] = {
            "S_final": float(s.mean()),
            "vs shipped": float(np.exp(np.log(s / ref).mean()) - 1),
            "folds won": int((s < ref).sum()),
            "/ persistence": float(np.exp(np.log(s / df["persistence | s_final"]).mean())),
            **{k: float(df[f"{name} | {k}"].mean()) for k in ("rmse", "trmse", "bias", "s_49_72")}}
    table = pd.DataFrame(out).T
    hl = {}
    for name in VARIANTS:
        if f"{name} | hybrid | s_final" not in df:
            continue
        h, lin = df[f"{name} | hybrid | s_final"], df[f"{name} | linear | s_final"]
        hl[name] = {"hybrid vs linear": float(np.exp(np.log(h / lin).mean()) - 1),
                    "hybrid wins": int((h < lin).sum())}
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(df)} folds (official truth; S in km/s):\n" + table.to_string())
        log("\nhybrid vs linear, same weights:\n" + pd.DataFrame(hl).T.to_string())
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = ch.Data()
    truth = pull_ace_1h.load()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        cached = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = [n for n in VARIANTS if f"{n} | linear | s_final" not in cached]
        todo += [n for n in BASELINE_VARIANTS if f"{n} | s_final" not in cached]
        if not todo:
            rows.append(cached)
            continue
        t = time.perf_counter()
        row = dict(cached)
        for n in todo:  # one variant at a time, saved as it goes
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
            row.update(score_fold(data, truth, start, end, timer, (n,)))
            if not pilot:
                path.write_text(json.dumps(row), encoding="utf-8")
        log(f"  fold {row['fold']} ({time.perf_counter() - t:.0f} s): " + ", ".join(
            f"{n} {row[f'{n} | linear | s_final']:.1f}/{row[f'{n} | hybrid | s_final']:.1f}"
            for n in VARIANTS) + " km/s (linear/hybrid); " + ", ".join(
            f"{n.removeprefix('final baseline ')} {row[f'{n} | s_final']:.1f}"
            for n in BASELINE_VARIANTS) + " km/s (final baseline)")
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB; six folds -> ~"
                f"{6 * (time.perf_counter() - t) / 60:.0f} min")
            summarize([row])
            return row
        rows.append(row)
    return summarize(rows)


def eval_year(old_linear=None, old_baseline=None):
    """The trained assets on the held-out eval year, once (see the module docstring)."""
    from models.xgboost_model import XGBoostModel as Hybrid
    from training import xgboost_model as tx
    from training.datasets import load_hourly
    from training.evaluate import forecast_times

    series = load_hourly()
    values = series.to_numpy(dtype=float)
    t0s = forecast_times()
    pos = series.index.get_indexer(t0s)
    models = {"linear (new)": LinearModel.load(), "hybrid (shipped)": Hybrid.load()}
    if old_linear:
        models["linear (previous)"] = LinearModel.load(old_linear)
    names = tuple(dict.fromkeys(n for m in models.values() for n in (
        m.input_features if isinstance(m, Hybrid) else m.config.input_features)))
    table = tx.feature_table(t0s, names)

    def x(m):
        cols = m.input_features if isinstance(m, Hybrid) else m.config.input_features
        return table[list(cols)].to_numpy(dtype=float)

    forecasts = {k: m.forecast(values, pos, x(m)) for k, m in models.items()}
    forecasts["final baseline (new)"] = FinalBaseline.load().forecast(values, pos)
    if old_baseline:
        forecasts["final baseline (previous)"] = FinalBaseline.load(old_baseline).forecast(
            values, pos)
    truth = pull_ace_1h.load()
    obs, pers = cs.verifying(truth, t0s), cs.persistence(truth, t0s)
    forecasts["persistence"] = np.repeat(pers[:, None], cs.HORIZON, 1)
    forecasts["recurrence 27 d"] = cs.recurrence(truth, t0s)
    out = {}
    for k, f in forecasts.items():
        s = cs.score(f, obs, pers)
        out[k] = {**{m: s[m] for m in STATS}, "n_pairs": s["n_pairs"],
                  "bias 72h": s["curves"]["bias"][71]}
    df = pd.DataFrame(out).T
    df["/ persistence"] = df["s_final"] / df.loc["persistence", "s_final"]
    df.sort_values("s_final").to_csv(EXPERIMENTS / "official_eval_year.csv",
                                     float_format="%.6g")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        log(f"eval year {t0s[0]:%Y-%m-%d} .. {t0s[-1]:%Y-%m-%d}, {len(t0s)} runs, "
            f"{np.isfinite(obs).mean():.1%} of pairs with official truth:" + chr(10)
            + df.sort_values("s_final").to_string())
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--eval-year", action="store_true")
    parser.add_argument("--old-linear", default=None)
    parser.add_argument("--old-baseline", default=None)
    args = parser.parse_args(argv)
    if args.eval_year:
        eval_year(args.old_linear, args.old_baseline)
    elif args.summary:
        summarize([json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(CACHE_DIR.glob("*.json"))])
    else:
        run(args.folds, args.pilot, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
