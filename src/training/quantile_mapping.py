"""A quantile-mapping post-step for the linear model and the hybrid, chosen on the yearly folds.

    python -m training.quantile_mapping --pilot --folds 2024   # one fold, timed, not cached
    python -m training.quantile_mapping [--max-seconds 240]    # every fold; resumable
    python -m training.quantile_mapping --summary

Models fitted by least squares shrink their forecasts towards the mean: quiet wind is
over-forecast and fast streams under-forecast (Collin et al. 2025 report a 22% lower peak
error after a distribution correction). The post-step maps each lead's forecast through
F_y^-1(F_f(f)): the forecast's quantile in the training forecasts is replaced by the same
quantile of the observed speeds, learned on the training runs before the fold. A blend
f' = f + alpha (map(f) - f) is scored for each alpha in ALPHAS; alpha = 0 is the model as
shipped. The training forecasts are in-sample (the models are refitted on the same runs),
which is close for models this constrained.

Per fold (training.folds.YEARLY), the shipped linear config (with its data fixes) and the
shipped hybrid's trees on top are refitted on the data before the fold start, as in
training.regressor_cap. Score: per-run mean TIMSE / persistence; summaries are geometric
means over folds. Results: src/experiments/quantile_mapping.csv. Motivation:
docs/model_review.md, item 2d.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import targets
from models.linear_model import LinearModel
from models.xgboost_model import XGBoostModel
from training import compare_history as ch
from training.datasets import DATA_START
from training.folds import YEARLY

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "quantile_mapping_cache"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PROBS = np.linspace(0.005, 0.995, 100)  # the map's knots
FAST = 600.0  # km/s: a "fast" run peaks above this (diagnostic)


def log(msg):
    print(msg, flush=True)


class QuantileMap:
    """Per-lead monotone map from forecast quantiles to observed quantiles.

    Beyond the outer knots the map shifts by the end offsets, so it never clips.
    """

    def __init__(self, forecast, y):
        n_leads = y.shape[1]
        self.f_knots = np.empty((n_leads, len(PROBS)))
        self.y_knots = np.empty((n_leads, len(PROBS)))
        for k in range(n_leads):
            ok = np.isfinite(forecast[:, k]) & np.isfinite(y[:, k])
            self.f_knots[k] = np.quantile(forecast[ok, k], PROBS)
            self.y_knots[k] = np.quantile(y[ok, k], PROBS)

    def __call__(self, forecast, alpha=1.0):
        out = np.array(forecast, dtype=float, copy=True)
        for k in range(out.shape[1]):
            fk, yk = self.f_knots[k], self.y_knots[k]
            mapped = np.interp(out[:, k], fk, yk)
            mapped = np.where(out[:, k] < fk[0], out[:, k] + yk[0] - fk[0], mapped)
            mapped = np.where(out[:, k] > fk[-1], out[:, k] + yk[-1] - fk[-1], mapped)
            out[:, k] += alpha * (mapped - out[:, k])
        return out


def spread(forecast, y):
    """Mean over leads of std(forecast) / std(y) on the rows where both exist."""
    ratios = []
    for k in range(y.shape[1]):
        ok = np.isfinite(forecast[:, k]) & np.isfinite(y[:, k])
        ratios.append(forecast[ok, k].std() / y[ok, k].std())
    return float(np.mean(ratios))


def fast_ratio(data, fold, forecast, ref_forecast):
    """Mean TIMSE on the fold's fast runs relative to `ref_forecast` on the same runs."""
    from training import linear_model as tl

    peak = np.where(np.isfinite(fold["y"]), fold["y"], -np.inf).max(axis=1)
    fast = fold["scored"] & (peak > FAST)
    s = tl.run_scores(forecast, fold["y"], fold["q"])[fast]
    r = tl.run_scores(ref_forecast, fold["y"], fold["q"])[fast]
    return float(s.mean() / r.mean()), int(fast.sum())


def score_fold(data, start, end, timer):
    fold = data.fold(start, end)
    start_pos = data.series.index.get_loc(start)
    fit6 = data.runs6[(data.runs6 >= DATA_START) & (data.runs6 < start)]
    fit_h = data.hourly[(data.hourly >= DATA_START) & (data.hourly < start)]
    pos6 = data.series.index.get_indexer(fit6)
    y_fit = targets(data.values, pos6, end=start_pos)
    out = {}

    t = time.perf_counter()
    lin = LinearModel(data.linear_cfg).fit(
        data.values, pos6, data.x(fit6, data.linear_cfg.input_features), end=start_pos)
    hyb = XGBoostModel(data.hybrid_cfg, lin)
    hyb.fit(data.values, data.series.index.get_indexer(fit_h),
            data.x(fit_h, hyb.input_features), end=start_pos)
    timer["fit"] = timer.get("fit", 0.0) + time.perf_counter() - t

    t = time.perf_counter()
    for name, model, feats in (("linear", lin, data.linear_cfg.input_features),
                               ("hybrid", hyb, hyb.input_features)):
        train_f = model.forecast(data.values, pos6, data.x(fit6, feats))
        qmap = QuantileMap(train_f, y_fit)
        f = model.forecast(data.values, fold["pos"], data.x(fold["t0s"], feats))
        out[f"{name} spread train"] = spread(train_f, y_fit)
        for a in ALPHAS:
            g = qmap(f, a)
            out[f"{name} a={a:g}"] = data.ratio(fold, g)
            out[f"{name} a={a:g} spread"] = spread(g, fold["y"])
            out[f"{name} a={a:g} fast"], out["fast runs"] = fast_ratio(data, fold, g, f)
    timer["map + score"] = timer.get("map + score", 0.0) + time.perf_counter() - t
    return out


def summarize(rows):
    df = pd.DataFrame(rows).set_index("fold")
    df.reset_index().to_csv(EXPERIMENTS / "quantile_mapping.csv", index=False,
                            float_format="%.6g")
    geo = {c: float(np.exp(np.log(df[c]).mean())) for c in df.columns
           if c not in ("seconds", "fast runs")}
    table = pd.DataFrame({
        a: {f"{m}{s}": v for m in ("linear", "hybrid") for s, v in (
            ("", geo[f"{m} a={a:g}"]),
            (" vs a=0", geo[f"{m} a={a:g}"] / geo[f"{m} a=0"] - 1),
            (" folds won", int((df[f"{m} a={a:g}"] < df[f"{m} a=0"]).sum())),
            (" fast runs vs a=0", geo[f"{m} a={a:g} fast"] - 1),
            (" spread", float(df[f"{m} a={a:g} spread"].mean())))}
        for a in ALPHAS}).T
    table.index.name = "alpha"
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        log("\nper fold:\n" + df[[c for c in df.columns if c.endswith(tuple(
            f"a={a:g}" for a in ALPHAS))]].T.to_string())
        log(f"\nover {len(df)} folds:\n{table.T.to_string()}")
        log("training-forecast spread (std f / std y): "
            + ", ".join(f"{m} {df[f'{m} spread train'].mean():.3f}" for m in ("linear", "hybrid")))
    return df


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = ch.Data()
    log(f"data ready in {time.perf_counter() - started:.0f} s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timer, rows = {}, []
    for start, end in YEARLY:
        if years is not None and start.year not in years:
            continue
        path = CACHE_DIR / f"{start:%Y}.json"
        if path.exists() and not pilot:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        r = {"fold": f"{start:%Y-%m}", **score_fold(data, start, end, timer)}
        r["seconds"] = time.perf_counter() - t
        log(f"  fold {r['fold']} ({r['seconds']:.0f} s): " + ", ".join(
            f"{k} {v:.4f}" for k, v in r.items()
            if isinstance(v, float) and k.endswith(tuple(f"a={a:g}" for a in ALPHAS))))
        if pilot:
            from training.xgboost_model import peak_mb

            log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
                + f"; peak memory {peak_mb():.0f} MB")
            log("  spread: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items() if "spread" in k))
            log(f"  fast runs ({r['fast runs']}): " + ", ".join(
                f"{k} {v:.3f}" for k, v in r.items() if k.endswith("fast")))
            log(f"  six folds -> ~{6 * r['seconds'] / 60:.0f} min")
            return r
        path.write_text(json.dumps(r), encoding="utf-8")
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
