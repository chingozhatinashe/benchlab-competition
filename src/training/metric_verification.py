"""Checks our official-metric code (training.competition_score) against the organisers' own
scoring code (scoring_demo/swp_scoring, an unmodified copy of what scores the leaderboard).

    python -m training.metric_verification

The organisers' code needs Python 3.11 (tomllib); on our 3.10 it runs unmodified with tomli,
the same parser, registered as tomllib. Four checks:
    1. demo      the organisers' demo (scoring_demo/data: Jul-Sep 2026, three baselines)
                 reproduces expected_report.json through their code, and our code scores the
                 same forecasts the same way
    2. metric    the same obs and forecast matrices through both metrics: the submitted
                 model on the eval year, the demo baselines, and synthetic edge cases
                 (speeds on a coarse grid, so many ties; missing observations; few runs)
    3. truth     our parser (competition_score.parse_monthly) against theirs on the demo
                 files, and our pulled truth (training.pull_ace_1h) against those files
    4. persistence  our rule (the latest valid hour labelled <= t0 - 1 h) against theirs
                 (the latest valid hour labelled <= t0, within 72 h): how often they differ,
                 and what that does to the persistence score and to a failed run's score
Results: src/experiments/metric_verification.json.
"""

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from training import competition_score as cs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEMO = ROOT / "scoring_demo"
EXPERIMENTS = HERE.parent / "experiments"
TOL = 1e-9  # relative: the two metrics should agree to rounding


def log(msg):
    print(msg, flush=True)


def official():
    """The organisers' score.py module (its swp_scoring package on the path)."""
    if not DEMO.exists():
        raise FileNotFoundError(f"{DEMO}: merge the upstream starter kit first")
    try:
        import tomllib  # noqa: F401
    except ModuleNotFoundError:
        import tomli

        sys.modules["tomllib"] = tomli
    if str(DEMO) not in sys.path:
        sys.path.insert(0, str(DEMO))
    return importlib.import_module("score")


def rel_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return 0.0
    return float(np.max(np.abs(a[both] - b[both]) / np.maximum(np.abs(b[both]), 1e-12)))


def compare_metric(obs, pred, pers=None):
    """Both metrics on the same matrices; pred has no gaps where obs is valid."""
    from swp_scoring.core import metric

    pers = pred[:, 0] if pers is None else pers
    ours = cs.score(pred, obs, pers)
    theirs = metric.score(obs, pred)
    return {"ours s_final": ours["s_final"], "theirs s_final": theirs.s_final,
            "s_final rel diff": rel_diff(ours["s_final"], theirs.s_final),
            "s curve max rel diff": rel_diff(ours["curves"]["s"], theirs.s_curve),
            "rmse curve max rel diff": rel_diff(ours["curves"]["rmse"], theirs.rmse_curve),
            "trmse curve max rel diff": rel_diff(ours["curves"]["trmse"], theirs.trmse_curve),
            "leads ours": int(np.isfinite(ours["curves"]["s"]).sum()),
            "leads theirs": int(np.isfinite(theirs.s_curve).sum())}


def utc_index(cycles):
    return pd.DatetimeIndex([pd.Timestamp(t) for t in cycles]).tz_convert("UTC")


def demo_truth_series():
    """Our parser on the demo's monthly files: hourly speed, NaN where invalid."""
    texts = sorted((DEMO / "data" / "ace").glob("*_ace_swepam_1h.txt"))
    s = pd.concat([cs.parse_monthly(p.read_text(encoding="ascii", errors="replace"))
                   for p in texts])
    return s[~s.index.duplicated(keep="last")].sort_index()


def check_demo(sc):
    from swp_scoring.core.evaluate import build_grid, observation_window, score_forecasts
    from swp_scoring.core.grid import prediction_matrix
    from swp_scoring.core.types import load_competition_contract, parse_valid_time

    window = json.loads((DEMO / "data" / "window.json").read_text())
    expected = json.loads((DEMO / "expected_report.json").read_text())["scores"]
    contract = load_competition_contract()
    start, end = parse_valid_time(window["start"]), parse_valid_time(window["end"])
    observations = sc.load_observations(*observation_window(start, end, contract),
                                        DEMO / "data" / "ace", contract)
    grid = build_grid(observations, start, end, contract)
    truth = demo_truth_series()
    t0s = utc_index(grid.cycles)
    obs_ours = cs.verifying(truth, t0s)
    out = {"cycles": len(grid.cycles),
           "obs matrices identical": bool(np.array_equal(grid.obs, obs_ours, equal_nan=True))}
    for path in sorted((DEMO / "data" / "forecasts").glob("*.json")):
        name, forecasts, _ = sc.load_forecasts(path, grid, contract)
        theirs = asdict(score_forecasts(grid, forecasts, contract))
        pred = prediction_matrix(grid.cycles, forecasts, grid.persistence,
                                 contract.horizon_hours)
        raw = np.full_like(pred, np.nan)  # the submitted values only; our score fills gaps
        for i, t0 in enumerate(grid.cycles):
            if t0 in forecasts:
                raw[i] = forecasts[t0]
        ours = cs.score(raw, obs_ours, cs.persistence(truth, t0s))
        out[name] = {"expected": expected[name]["s_final"],
                     "theirs (rerun)": theirs["s_final"], "ours": ours["s_final"],
                     "rerun reproduces expected": bool(
                         abs(theirs["s_final"] - expected[name]["s_final"]) < 1e-6),
                     **{f"matrix {k}": v for k, v in compare_metric(grid.obs, pred).items()
                        if "diff" in k}}
    return out, grid, truth


def check_synthetic(seed=0):
    rng = np.random.default_rng(seed)
    cases = {}
    for name, (runs, step, frac_missing) in {
        "continuous, 1448 runs": (1448, 0.0, 0.05),
        "1 km/s grid (ties), 1448 runs": (1448, 1.0, 0.05),
        "10 km/s grid (many ties), 300 runs": (300, 10.0, 0.2),
        "0.1 km/s grid (ACE resolution), 1448 runs": (1448, 0.1, 0.05),
        "few runs (12)": (12, 1.0, 0.0),
    }.items():
        obs = 300 + 500 * rng.beta(2, 5, (runs, cs.HORIZON))
        if step:
            obs = np.round(obs / step) * step
        pred = obs + rng.normal(0, 60, obs.shape)
        obs[rng.random(obs.shape) < frac_missing] = np.nan
        cases[name] = compare_metric(obs, pred)
    # a lead whose valid observations are all equal: spec TIMSE 0 (theirs) vs NaN (ours)
    obs = 300 + 500 * rng.beta(2, 5, (50, cs.HORIZON))
    obs[:, 0] = 400.0
    pred = obs + rng.normal(0, 60, obs.shape)
    cases["one lead with constant observations"] = compare_metric(obs, pred)
    return cases


def check_truth(truth_demo):
    from training import pull_ace_1h

    pulled = pull_ace_1h.load()
    idx = truth_demo.index
    ours = pulled.reindex(idx)
    both = truth_demo.notna() & ours.notna()
    return {"hours in demo files": int(len(idx)),
            "valid in demo files": int(truth_demo.notna().sum()),
            "valid in our pulled truth": int(ours.notna().sum()),
            "valid in one only": int((truth_demo.notna() != ours.notna()).sum()),
            "values differ (both valid)": int((np.abs(truth_demo[both] - ours[both]) > 1e-6).sum())}


def official_persistence(truth, t0s, lookback=72):
    """Their rule on our truth: the latest valid hour labelled <= t0, within `lookback` h."""
    valid = truth.dropna()
    v = valid.asof(t0s)
    stamp = pd.Series(valid.index, index=valid.index).asof(t0s)
    too_old = (t0s - pd.DatetimeIndex(stamp)) > pd.Timedelta(hours=lookback)
    return np.where(too_old, np.nan, v.to_numpy(float))


def check_persistence_and_eval():
    from swp_scoring.core import metric

    from models.linear_model import LinearModel
    from training import model_diagnostics as dg

    data = dg.Data()
    t0s = data.eval_t0s
    obs = cs.verifying(data.truth, t0s)
    ours_p = cs.persistence(data.truth, t0s)
    theirs_p = official_persistence(data.truth, t0s)
    differ = ~np.isclose(ours_p, theirs_p, equal_nan=True)
    model = LinearModel.load()
    f = model.forecast(data.full_values, data.full.index.get_indexer(t0s),
                       data.eval_table[list(model.config.input_features)].to_numpy(float))
    out = {"eval runs": len(t0s),
           "persistence differs (runs)": int(differ.sum()),
           "persistence mean abs diff km/s": float(np.nanmean(np.abs(ours_p - theirs_p))),
           "submitted model, same matrices": compare_metric(obs, f, ours_p)}
    for name, p in (("ours", ours_p), ("theirs", theirs_p)):
        pers_fc = np.repeat(p[:, None], cs.HORIZON, 1)
        out[f"persistence S_final, {name} rule"] = metric.score(obs, pers_fc).s_final
        failed = f.copy()  # every 10th run failed: scored as persistence
        failed[::10] = pers_fc[::10]
        out[f"10% failed runs S_final, {name} rule"] = metric.score(obs, failed).s_final
    out["submitted model S_final"] = metric.score(obs, f).s_final
    return out


def run(seed=0):
    sc = official()
    log("1. the organisers' demo")
    demo, _, truth_demo = check_demo(sc)
    log(json.dumps(demo, indent=1))
    log("2. synthetic edge cases")
    synth = check_synthetic(seed)
    log(json.dumps(synth, indent=1))
    log("3. truth")
    truth = check_truth(truth_demo)
    log(json.dumps(truth, indent=1))
    log("4. persistence and the eval year")
    pe = check_persistence_and_eval()
    log(json.dumps(pe, indent=1))
    out = {"run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "demo": demo, "synthetic": synth, "truth": truth, "persistence and eval year": pe}
    (EXPERIMENTS / "metric_verification.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run(args.seed)


if __name__ == "__main__":
    sys.exit(main())
