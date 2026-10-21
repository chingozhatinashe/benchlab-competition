"""Tune, train and score the reference baseline (models.final_baseline).

    python -m training.final_baseline tune       # grid search on the training years only
        [--max-seconds 500] [--fresh]           # resumable: re-run until it completes
    python -m training.final_baseline train      # fit the chosen config, write src/assets/
    python -m training.final_baseline evaluate   # score it on the eval year (once, at the end)

Tuning never touches the eval year. It uses two walk-forward folds inside the
training years: fit on years 1-2 and validate on year 3, then fit on years 1-3 and
validate on year 4. Each fold forecasts every 6-hourly t0 of its validation year
from the hours before t0, exactly as the eval harness and the live container do.
The objective is per-run mean TIMSE relative to persistence, averaged over the
folds as a geometric mean so each year counts equally.
"""

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import last_observed, targets
from models.final_baseline import ASSET, Config, FinalBaseline, cycle_positions
from training.datasets import DATA_START, TRAIN_END, load_train
from training.evaluate import score

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "final_baseline_config.json"
RESULTS_PATH = HERE.parent / "experiments" / "final_baseline_tuning.csv"
CACHE_DIR = HERE.parent / "data" / "tuning_cache"  # gitignored; one CSV per grid slice

YEAR = pd.Timedelta(days=365)
FOLDS = [  # (validation start, validation end); the fit uses every hour before the start
    (TRAIN_END - 2 * YEAR, TRAIN_END - YEAR),
    (TRAIN_END - YEAR, TRAIN_END),
]

GRID = {
    "lags_days": [(27.0,), (27.0, 13.5), (27.0, 13.5, 9.0)],
    "tail_power": [0.0, 0.25, 0.5, 0.75, 1.0],
    "loss": ["run_timse", "timse"],
    "window_days": [7.0, 14.0, 30.0, 60.0, 90.0],
    "rolling_weight": [0.0, 0.25, 0.5, 0.75, 1.0],
    "blend_persistence": [False, True],
}
OBJECTIVE = "run_timse_mean"
METRICS = ["timse", "run_timse_mean", "run_timse_median", "run_timse_p90", "rmse"]


def log(msg):
    print(msg, flush=True)


def fold_data(series, start, end):
    """Everything the fold needs that doesn't depend on hyper-parameters."""
    fit = series[series.index < start]
    values = series[series.index < end].to_numpy(dtype=float)
    val_pos = cycle_positions(series.index[: len(values)], start, end)
    fit_values = fit.to_numpy(dtype=float)
    return {
        "fit": fit,
        "fit_values": fit_values,
        "values": values,
        "val_pos": val_pos,
        "val_last": last_observed(values, val_pos),
        "val_y": targets(values, val_pos),
    }


def persistence_scores(fold):
    last = fold["val_last"][:, None].repeat(72, axis=1)
    return score(last, fold["val_y"])


def evaluate_grid(fold, lags, tail_power, loss):
    """Score every (window, rolling weight, blend) for one (lags, tail power, loss) on one fold.

    Mirrors FinalBaseline.fit step by step, but shares the expensive pieces: the two
    frozen fits don't depend on the window or the rolling weight, and the rolling
    coefficients don't depend on the rolling weight.
    """
    fit, fit_values = fold["fit"], fold["fit_values"]
    base_cfg = Config(lags_days=lags, tail_power=tail_power, loss=loss, rolling_weight=1.0)
    model = FinalBaseline(base_cfg)
    model.prepare(fit_values)

    split = len(fit_values) - int(round(base_cfg.stack_days * 24))
    frozen_stack = model.fit_frozen(fit_values[:split])
    frozen_full = model.fit_frozen(fit_values)
    stack_pos = cycle_positions(fit.index, fit.index[split], fit.index[-1])
    stack_last = last_observed(fit_values, stack_pos)
    stack_y = targets(fit_values, stack_pos)

    rows = []
    for window in GRID["window_days"]:
        model.config = replace(base_cfg, window_days=window)
        model.coef = frozen_stack
        stack_roll = model.rolling(fit_values, stack_pos)
        model.coef = frozen_full
        val_roll = model.rolling(fold["values"], fold["val_pos"])
        for lam in GRID["rolling_weight"]:
            model.coef = frozen_stack
            weight = model.blend_weight(model.combine(*stack_roll, lam), stack_last, stack_y)
            model.coef = frozen_full
            regression = model.combine(*val_roll, lam)
            for blend in GRID["blend_persistence"]:
                w = weight if blend else np.zeros_like(weight)
                forecasts = model.finish(regression, fold["val_last"], w)
                result = score(forecasts, fold["val_y"])
                rows.append({
                    "lags_days": "/".join(f"{d:g}" for d in lags),
                    "tail_power": tail_power,
                    "loss": loss,
                    "window_days": window,
                    "rolling_weight": lam,
                    "blend_persistence": blend,
                    **{m: result[m] for m in METRICS},
                })
    return rows


def tune(series=None, max_seconds=None, fresh=False):
    """Run the grid; each (fold, lags, power, loss) slice is cached as it finishes.

    With max_seconds, no new slice starts after that time and the call returns None
    if the grid is incomplete; run it again to continue. fresh=True clears the cache
    (do that whenever the model code changes).
    """
    series = load_train() if series is None else series
    started = time.perf_counter()
    if fresh and CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.csv"):
            f.unlink()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames, missing = [], 0
    for i, (start, end) in enumerate(FOLDS, 1):
        fold = fold_data(series, start, end)
        ref = persistence_scores(fold)
        log(f"fold {i}: fit {DATA_START:%Y-%m-%d}..{start:%Y-%m-%d}, "
            f"validate {start:%Y-%m-%d}..{end:%Y-%m-%d} ({len(fold['val_pos'])} runs); "
            f"persistence run TIMSE mean {ref[OBJECTIVE]:.4g}")
        for lags, power, loss in itertools.product(
            GRID["lags_days"], GRID["tail_power"], GRID["loss"]
        ):
            if power == 0 and loss != GRID["loss"][0]:
                continue  # without tail weighting every loss is least squares
            lag_name = "-".join(f"{d:g}" for d in lags)
            path = CACHE_DIR / f"fold{i}_lags{lag_name}_power{power:g}_{loss}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
                continue
            if max_seconds is not None and time.perf_counter() - started > max_seconds:
                missing += 1
                continue
            df = pd.DataFrame(evaluate_grid(fold, lags, power, loss))
            df["fold"] = i
            for m in METRICS:
                df[f"{m}_vs_persistence"] = df[m] / ref[m]
            df.to_csv(path, index=False)
            frames.append(pd.read_csv(path))  # same dtypes as cached slices
            log(f"  {path.stem} done ({time.perf_counter() - started:.0f} s)")
    if missing:
        log(f"\n{missing} slices still to run; run the same command again to continue.")
        return None

    keys = ["lags_days", "tail_power", "loss", "window_days", "rolling_weight",
            "blend_persistence"]
    long = pd.concat(frames, ignore_index=True)
    ratios = [f"{m}_vs_persistence" for m in METRICS]
    # geometric mean over folds of each score relative to persistence
    summary = long.groupby(keys)[ratios].agg(lambda s: float(np.exp(np.log(s).mean())))
    summary = summary.sort_values(f"{OBJECTIVE}_vs_persistence").reset_index()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(RESULTS_PATH, index=False, float_format="%.6g")

    # (i) asks for a persistence blend, so the chosen config always has one.
    best = summary[summary["blend_persistence"]].iloc[0]
    config = Config(
        lags_days=tuple(float(d) for d in best["lags_days"].split("/")),
        tail_power=float(best["tail_power"]),
        loss=str(best["loss"]),
        window_days=float(best["window_days"]),
        rolling_weight=float(best["rolling_weight"]),
        blend_persistence=True,
    )
    chosen = asdict(config)
    chosen["lags_days"] = list(chosen["lags_days"])
    chosen["tuning"] = {
        "objective": f"geometric mean over folds of {OBJECTIVE} / persistence",
        "folds": [[f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for a, b in FOLDS],
        "score": {r: round(float(best[r]), 4) for r in ratios},
    }
    CONFIG_PATH.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        log("\nTop 10 configs (scores relative to persistence, lower is better):")
        log(summary.head(10).to_string(index=False))
    log(f"\nChosen: {config}\nwritten to {CONFIG_PATH}\nall results in {RESULTS_PATH}")
    return summary, config


def load_config(path=CONFIG_PATH):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d.pop("tuning", None)
    d["lags_days"] = tuple(d["lags_days"])
    return Config(**d)


def train(config=None, out=ASSET):
    config = load_config() if config is None else config
    model = FinalBaseline(config).fit(load_train())
    model.save(out)
    log(f"Fitted {config} on {model.fit_info['start']}..{model.fit_info['end']}; wrote {out}")
    return model


def evaluate_eval_year(config=None):
    """Score the chosen config and the reference baselines on the eval year."""
    from training import run_baselines
    from training.baselines import ALL

    config = load_config() if config is None else config
    names = ["persistence", "recurrence_27d", "recurrence_27d_rolling"]
    table = run_baselines.run({n: ALL[n] for n in names})
    table = pd.concat(
        [table, run_baselines.run({"final_baseline": lambda: FinalBaseline(config)})]
    )
    cols = ["model", *METRICS]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        log(table[cols].to_string(index=False))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["tune", "train", "evaluate"])
    parser.add_argument("--max-seconds", type=float, help="tune: stop starting slices after this")
    parser.add_argument("--fresh", action="store_true", help="tune: clear cached slices first")
    args = parser.parse_args(argv)
    if args.command == "tune":
        return 0 if tune(max_seconds=args.max_seconds, fresh=args.fresh) else 3
    {"train": train, "evaluate": evaluate_eval_year}[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
