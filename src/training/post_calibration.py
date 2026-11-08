"""Post-hoc corrections of the linear forecast (persistence blend, calibration), yearly folds.

    python -m training.post_calibration --pilot --folds 2024   # one fold, timed, not cached
    python -m training.post_calibration [--max-seconds N]      # every fold; resumable
    python -m training.post_calibration --summary

Items (i b) and (i d) of the follow-up plan to docs/model_diagnostics.md: the slow-wind
over-forecast and the too-high forecasts at high speeds (residual -90 km/s for forecasts of
650-700 km/s). Quantile mapping, a distribution map, failed (docs/quantile_mapping.md); these
are regressions of the residual, with the official-metric loss weights.

Nested, so the correction never sees the fold. Per fold starting at S:
    inner   the submitted config fitted on the runs before S - 1 year, forecasting the
            year before S; the correction is fitted on those forecasts (official truth)
    outer   the submitted config fitted on the runs before S (the reference), forecasting
            the fold; the correction is applied to those forecasts
    in-sample  (added after the pilot) the fold's own model on all its fit runs: more
            data and the same model, but the residuals are in-sample
Corrections (residual r = observed - forecast; loss weights official_weights(blend 0.25)):
    bias                one offset per lead (as the retune's "shipped - bias", nested)
    persistence blend   f + w(k) (persistence - f), w a quadratic in the lead
    blend, slow apart   the same with a separate w(k) when the last speed is below 450
    calibration         per lead, ridge regression of r on the forecast, the last speed and
                        hinges of both (forecast at 450/550/650 km/s, last speed below 450
                        and above 650); ridge 0.1 or 1 (relative, intercept spared)
    calibration, slow   the same fitted on and applied to the runs with last speed < 450
Score: official S_final / persistence, geometric mean over the six yearly folds, plus the
run regimes and the slow runs' mean residual. Decision rule, set before running: keep if
> 1% better than the reference and winning >= 4 of 6 folds, without the slow runs getting
worse; a gain >= 0.3% on >= 4 folds goes forward to the combination step. The eval year is
not used. Results: src/experiments/post_calibration.csv (cache: src/data/post_calibration_cache/).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.recurrence import last_observed
from models.final_baseline import VALID_RANGE, official_weights, solve
from models.linear_model import LinearModel
from training import competition_score as cs
from training import model_diagnostics as dg
from training import xgboost_model as tx
from training.folds import YEAR, YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "post_calibration_cache"
KEEP_GAIN, KEEP_FOLDS, NEAR_GAIN = 0.01, 4, 0.003
BLEND = 0.25
SLOW = 450.0
F_KNOTS = (450.0, 550.0, 650.0)
VARIANTS = ("reference", "bias", "persistence blend", "blend, slow apart",
            "calibration r0.1", "calibration r1", "calibration, slow",
            "blend, slow apart, in-sample", "calibration r1, in-sample")
LEADS = np.arange(1, cs.HORIZON + 1)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------- corrections
def _fill(last, f):
    return np.where(np.isfinite(last), last, np.nanmean(f))


def calibration_design(f, last):
    """(n, 72, 8): 1, forecast, last speed (both (v - 450) / 100) and their hinges."""
    lastf = _fill(last, f)[:, None] * np.ones_like(f)
    cols = [np.ones_like(f), (f - SLOW) / 100, (lastf - SLOW) / 100]
    cols += [np.maximum(0.0, f - k) / 100 for k in F_KNOTS]
    cols += [np.maximum(0.0, SLOW - lastf) / 100, np.maximum(0.0, lastf - 650.0) / 100]
    return np.stack(cols, axis=-1)


def blend_design(f, pers, last, slow_apart):
    """(n, 72, 3 or 6): (persistence - f) times 1, k/72, (k/72)^2 [, the same for slow runs]."""
    d = (pers[:, None] - f)[:, :, None]
    basis = np.stack([np.ones(cs.HORIZON), LEADS / 72, (LEADS / 72) ** 2], axis=-1)[None]
    x = d * basis
    if slow_apart:
        slow = (_fill(last, f) < SLOW)[:, None, None]
        x = np.concatenate([x * ~slow, x * slow], axis=-1)
    return x


def fit_correction(kind, f, obs, pers, last, ridge=0.1):
    """Coefficients of a correction, fitted on inner forecasts."""
    w = official_weights(obs, BLEND)
    ok = (w > 0) & np.isfinite(f) & np.isfinite(pers)[:, None]
    w = np.where(ok, w, 0.0)
    r = np.where(ok, obs - f, 0.0)
    if kind == "bias":
        return {"offset": (w * r).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-12)}
    if kind.startswith("blend"):
        x = blend_design(f, np.nan_to_num(pers, nan=0.0), last, kind == "blend, slow apart")
        x = np.where(ok[:, :, None], x, 0.0)
        a = np.einsum("nkf,nk,nkg->fg", x, w, x)
        b = np.einsum("nkf,nk->f", x, w * r)
        return {"coef": np.linalg.solve(a + 1e-9 * np.eye(len(a)), b)}
    rows = np.ones(len(f), dtype=bool) if kind != "calibration, slow" else _fill(last, f) < SLOW
    x = calibration_design(f, last)
    wk = w * rows[:, None]
    a = np.einsum("nkf,nk,nkg->kfg", x, wk, x)
    b = np.einsum("nkf,nk->kf", x, wk * r)
    return {"coef": solve(a, b, ridge)}


def apply_correction(kind, params, f, pers, last):
    if kind == "bias":
        out = f + params["offset"][None]
    elif kind.startswith("blend"):
        x = blend_design(f, np.where(np.isfinite(pers), pers, np.nanmean(f)), last,
                         kind == "blend, slow apart")
        out = f + x @ params["coef"]
    else:
        corr = np.einsum("nkf,kf->nk", calibration_design(f, last), params["coef"])
        if kind == "calibration, slow":
            corr = corr * (_fill(last, f) < SLOW)[:, None]
        out = f + corr
    return np.clip(out, *VALID_RANGE)


KIND = {"bias": ("bias", None), "persistence blend": ("blend", None),
        "blend, slow apart": ("blend, slow apart", None),
        "calibration r0.1": ("calibration", 0.1), "calibration r1": ("calibration", 1.0),
        "calibration, slow": ("calibration, slow", 1.0),
        "blend, slow apart, in-sample": ("blend, slow apart", None),
        "calibration r1, in-sample": ("calibration", 1.0)}


# ---------------------------------------------------------------------- folds
def linear_forecasts(data, table, fit_end, *t0_sets):
    """The config fitted on the runs before fit_end; (forecast, last speed) per t0 set."""
    cfg = data.cfg
    cols = list(cfg.input_features)
    fit6 = data.runs6[data.runs6 < fit_end]
    lin = LinearModel(cfg).fit(data.values, data.series.index.get_indexer(fit6),
                               table.loc[fit6, cols].to_numpy(dtype=float),
                               end=data.series.index.get_loc(fit_end))
    out = []
    for t0s in t0_sets:
        pos = data.series.index.get_indexer(t0s)
        out.append((lin.forecast(data.values, pos, table.loc[t0s, cols].to_numpy(dtype=float)),
                    last_observed(data.values, pos)))
    return out


def score_fold(data, start, end, timer):
    inner_start = start - YEAR
    fit6 = data.runs6[data.runs6 < start]
    inner = data.runs6[(data.runs6 >= inner_start) & (data.runs6 < start)]
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    table = data.table.copy()
    # PFSS gaps: the fit runs' median, per fit (as training.linear_model.with_pfss)
    t = time.perf_counter()
    t_in = table.copy()
    fit_in = data.runs6[data.runs6 < inner_start]
    t_in[data.pfss] = t_in[data.pfss].fillna(t_in.loc[fit_in, data.pfss].median())
    ((f_in, last_in),) = linear_forecasts(data, t_in, inner_start, inner)
    table[data.pfss] = table[data.pfss].fillna(table.loc[fit6, data.pfss].median())
    # in-sample: the fold's own model on its fit runs whose 72 h ends before the fold
    own = fit6[fit6 < start - pd.Timedelta(hours=cs.HORIZON + 1)]
    (f, last), (f_own, last_own) = linear_forecasts(data, table, start, val, own)
    timer["linear fits"] = timer.get("linear fits", 0.0) + time.perf_counter() - t
    obs_in, pers_in = cs.verifying(data.truth, inner), cs.persistence(data.truth, inner)
    obs_own, pers_own = cs.verifying(data.truth, own), cs.persistence(data.truth, own)
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    regime = data.run_regimes(val)
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"]}
    t = time.perf_counter()
    for name in VARIANTS:
        if name == "reference":
            fc = f
        else:
            kind, ridge = KIND[name]
            src = ((f_own, obs_own, pers_own, last_own) if name.endswith("in-sample")
                   else (f_in, obs_in, pers_in, last_in))
            params = fit_correction(kind, *src, ridge or 0.1)
            fc = apply_correction(kind, params, f, pers, last)
        row = dg.regime_scores(fc, obs, pers, regime)
        res = obs - fc
        for r in dg.RUN_REGIMES:
            row[f"residual {r}"] = float(np.nanmean(res[regime == r]))
        row["residual all"] = float(np.nanmean(res))
        out[name] = row
    timer["corrections"] = timer.get("corrections", 0.0) + time.perf_counter() - t
    return out


def run(years=None, pilot=False, max_seconds=None):
    started = time.perf_counter()
    data = dg.Data()
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
        log(f"  {year}: " + ", ".join(
            f"{n} {rows[year][n]['all'] / rows[year]['persistence']:.4f}" for n in VARIANTS)
            + f" ({time.perf_counter() - t:.0f} s)")
    if pilot:
        log("  time: " + ", ".join(f"{k} {v:.1f} s" for k, v in timer.items())
            + f"; peak memory {tx.peak_mb():.0f} MB; six folds -> ~"
            f"{6 * sum(timer.values()) / 60:.0f} min after the data")
    return summarize(rows, pilot)


def summarize(rows, pilot=False):
    from training import slow_wind

    table = slow_wind.summarize(rows, pilot=True)  # same table layout and rules
    if not pilot:
        years = sorted(rows)
        pd.DataFrame([{"fold": y, "variant": n, "persistence": rows[y]["persistence"],
                       **rows[y][n]} for n in VARIANTS for y in years]).to_csv(
            EXPERIMENTS / "post_calibration.csv", index=False, float_format="%.6g")
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
