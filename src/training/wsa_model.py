"""WSA/HUX speed from the PFSS rings: auto-calibrated on the yearly folds, alone and as a regressor.

    python -m training.wsa_model --sample 500      # timed pilot (cached separately)
    python -m training.wsa_model [--max-seconds N]  # every training run; resumable
    python -m training.wsa_model --summary

Model: models.wsa. Each run's PFSS ring (training.pull_gong; the newest GONG map posted by
t0 - 10 min) gives the WSA source speed per Carrington longitude, HUX carries it to 1 AU, and
lead k reads the longitude facing Earth at t0 + k.

Auto-calibration, per fold, on the fold's fit runs only (training years before the fold):
    offline   the six WSA parameters (models.wsa.Params: v0, dv, alpha, w, delta, shift) by
              Nelder-Mead on the official S_final of the WSA forecast alone, from the
              literature values (Arge et al. 2003), on CALIBRATION_RUNS runs
    rolling   then, in each run, an offset and a scale re-fitted on the run's own past days:
              the same map's hindcast at the longitudes that faced Earth then, against the ACE
              hours that ended by t0 - 10 min (models.wsa.rolling_fit, ridge toward 0 / 1);
              ROLLING_DAYS windows
Variants, scored on the fold's validation runs:
    alone     WSA offline, and with each rolling window (diagnostics against persistence)
    + linear  "wsa_speed_lead_<D>h" added to the submitted linear model (lead k: the WSA
              speed at the longitude facing Earth at t0 + k - D), D in LEAD_DELAYS, offline
              and rolling; missing values (no usable map) filled with the fit rows' median
Score: official S_final against the official truth, relative to persistence; geometric means
over the six folds. Decision rule, set before running: keep a linear-model variant only if it
is more than 1% better than the submitted config and wins at least 4 of 6 folds. The eval
year is not used. Results: src/experiments/wsa_model.csv (cache: src/data/wsa_model_cache/).
"""

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from feature_engineering import pfss
from models import wsa
from models.linear_model import lag_column, lead_lags
from training import competition_score as cs
from training import linear_model as tl
from training.datasets import load_train

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "wsa_model_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
CALIBRATION_RUNS = 1500
FREE = ("v0", "dv", "alpha", "w", "delta", "shift")
BOUNDS = {"v0": (150, 450), "dv": (100, 900), "alpha": (0.02, 1.5), "w": (0.2, 15),
          "delta": (0.3, 4), "shift": (-60, 60)}
ROLLING_DAYS = (14, 27)
ROLLING_RIDGE = 1.0
ROLLING_GAP_HOURS = 2  # hindcast hours: labels up to t0 - 2 h (the hour ending by t0 - 1 h)
LEAD_DELAYS = (0, 12, 24)
SIGNAL = "wsa_speed"
HOUR = pd.Timedelta(hours=1)


def log(msg):
    print(msg, flush=True)


class Data:
    """Runs, their rings, Earth's longitudes for the forecast and hindcast hours, and truth."""

    def __init__(self, sample=None, seed=0):
        from training import pull_ace_1h, pull_gong

        self.series = load_train()
        self.t0s = tl.training_runs(self.series, sample, seed)
        self.truth = pull_ace_1h.load()
        rt = pull_gong.load_ring_table()
        i = rt.t0.searchsorted(self.t0s, side="right") - 1
        ok = i >= 0
        ok[ok] = (self.t0s[ok] - rt.t0[i[ok]]) < pd.Timedelta(hours=6)
        ring = np.full((len(self.t0s), *pfss.ring_shape()), np.nan, np.float32)
        ring[ok] = rt.ring[i[ok]]
        self.log_fs = ring[:, pfss.RING_VARS.index("log_fs")]
        self.theta_b = ring[:, pfss.RING_VARS.index("theta_b")]
        self.has_ring = np.isfinite(self.log_fs).mean(axis=1) > 0.5
        lead_hours = np.arange(1, cs.HORIZON + 1)
        past_hours = -np.arange(ROLLING_GAP_HOURS, max(ROLLING_DAYS) * 24 + ROLLING_GAP_HOURS)
        self.lead_lon = self.longitudes(lead_hours)
        self.past_lon = self.longitudes(past_hours)
        # lag L of a lead-aligned column: the longitude facing Earth at t0 - L
        self.lags = sorted({int(lag) for d in LEAD_DELAYS
                            for lag in np.unique(lead_lags(d, SIGNAL))})
        self.lag_lon = self.longitudes(-np.array(self.lags))
        labels = self.t0s.values[:, None] + (past_hours * 3600e9).astype("timedelta64[ns]")
        self.past_obs = self.series.reindex(
            pd.DatetimeIndex(labels.ravel(), tz="UTC")).to_numpy(float).reshape(labels.shape)

    def longitudes(self, hours):
        """Earth's Carrington longitude at t0 + h for each run and each h: (runs, len(hours))."""
        hours = np.asarray(hours)
        times = self.t0s.values[:, None] + (hours * 3600e9).astype("timedelta64[ns]")
        flat = pd.DatetimeIndex(times.ravel(), tz="UTC")
        return pfss.earth_longitude(flat).reshape(times.shape)

    def v1(self, rows, params):
        """HUX speed at 1 AU per longitude (runs, 360), NaN rows where there is no ring."""
        out = np.full((len(rows), wsa.N_LON), np.nan)
        ok = self.has_ring[rows]
        sel = np.asarray(rows)[ok]
        if len(sel):
            out[ok] = wsa.hux(wsa.source_speed(self.log_fs[sel], self.theta_b[sel], params))
        return out

    def forecast(self, rows, params, v1=None):
        v1 = self.v1(rows, params) if v1 is None else v1
        return wsa.at_longitude(np.nan_to_num(v1, nan=0.0), self.lead_lon[rows] + params.shift) \
            * np.where(self.has_ring[rows], 1.0, np.nan)[:, None]

    def rolled(self, rows, params, days, v1=None):
        """(offset, scale) per run from its hindcast over the last `days` days."""
        v1 = self.v1(rows, params) if v1 is None else v1
        n = days * 24
        past = wsa.at_longitude(np.nan_to_num(v1, nan=0.0), self.past_lon[rows, :n]
                                + params.shift)
        coef = np.array([wsa.rolling_fit(past[j], self.past_obs[r, :n], ROLLING_RIDGE)
                         if self.has_ring[r] else (0.0, 1.0) for j, r in enumerate(rows)])
        return coef[:, 0], coef[:, 1]


def persistence_filled(forecast, pers):
    """Runs without a ring scored as persistence (as a failed live forecast would be)."""
    return np.where(np.isfinite(forecast), forecast, pers[:, None])


def calibrate(data, rows, seed=0, max_evals=400):
    """Offline WSA parameters on `rows` (fit runs with a ring) by Nelder-Mead on S_final."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)[data.has_ring[rows]]
    if len(rows) > CALIBRATION_RUNS:
        rows = np.sort(rng.choice(rows, CALIBRATION_RUNS, replace=False))
    t = data.t0s[rows]
    obs, pers = cs.verifying(data.truth, t), cs.persistence(data.truth, t)
    ok = np.isfinite(pers)
    rows, obs, pers = rows[ok], obs[ok], pers[ok]
    lo = np.array([BOUNDS[k][0] for k in FREE])
    hi = np.array([BOUNDS[k][1] for k in FREE])
    x0 = np.array([getattr(wsa.Params(), k) for k in FREE])
    scale = hi - lo

    def params(z):
        return wsa.Params.from_vector(np.clip(lo + z * scale, lo, hi), FREE)

    def loss(z):
        f = data.forecast(rows, params(z))
        return cs.score(persistence_filled(f, pers), obs, pers)["s_final"]

    res = minimize(loss, (x0 - lo) / scale, method="Nelder-Mead",
                   options={"maxfev": max_evals, "xatol": 1e-3, "fatol": 1e-3})
    return params(res.x), float(res.fun), int(res.nfev), float(loss((x0 - lo) / scale))


def wsa_columns(data, rows, params, v1=None, offset=None, scale=None):
    """{"wsa_speed_at_<L>h": values} at every lag of LEAD_DELAYS (NaN without a ring)."""
    v1 = data.v1(rows, params) if v1 is None else v1
    vals = wsa.at_longitude(np.nan_to_num(v1, nan=0.0), data.lag_lon[rows] + params.shift)
    if offset is not None:
        vals = offset[:, None] + scale[:, None] * vals
    vals[~data.has_ring[rows]] = np.nan
    return {lag_column(SIGNAL, lag): vals[:, j] for j, lag in enumerate(data.lags)}


def run(sample=None, seed=0, max_seconds=None):
    started = time.perf_counter()
    tracemalloc.start()
    tag = f"pilot{sample}_seed{seed}" if sample else "full"
    cache = CACHE_DIR / tag
    cache.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    data = Data(sample, seed)
    base = tl.candidate_table(data.t0s)
    t_table = time.perf_counter() - t
    log(f"{len(data.t0s)} runs ({tag}); tables {t_table:.0f} s; runs with a ring: "
        f"{data.has_ring.mean():.0%}")
    folds = tl.Folds(data.series, base, data.series.index.get_indexer(data.t0s),
                     truth=data.truth)
    shipped = tl.with_fixes(tl.load_config())
    rows, times = {}, []

    for fi, ((start, _), f) in enumerate(zip(tl.FOLDS, folds.folds, strict=True)):
        year = f"{start:%Y}"
        path = cache / f"{year}.json"
        if path.exists():
            rows[year] = json.loads(path.read_text(encoding="utf-8"))
            continue
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run the same command again to continue")
            return None
        t = time.perf_counter()
        params, s_cal, nfev, s_default = calibrate(data, f["fit"], seed=fi)
        log(f"{year}: calibrated in {time.perf_counter() - t:.0f} s ({nfev} evaluations): "
            f"S_final {s_default:.2f} -> {s_cal:.2f} on the fit runs; {params}")
        out = {"params": params.to_dict(), "s_calibration": s_cal, "s_default": s_default}
        val = f["val"]
        pers = f["pers"]
        v1_val = data.v1(val, params)
        alone = data.forecast(val, params, v1_val)
        out["alone offline"] = cs.score(persistence_filled(alone, pers), f["obs"], pers)[
            "s_final"]
        coefs = {}
        for days in ROLLING_DAYS:
            o, s = data.rolled(val, params, days, v1_val)
            coefs[days] = (o, s)
            rolled = o[:, None] + s[:, None] * alone
            out[f"alone rolling {days}d"] = cs.score(persistence_filled(rolled, pers),
                                                     f["obs"], pers)["s_final"]
        out["persistence"] = f["ref"]
        # the linear model with WSA columns: offline, and rolling (fit rows re-fitted too)
        v1_fit = data.v1(f["fit"], params)
        fit_coefs = {d: data.rolled(f["fit"], params, d, v1_fit) for d in ROLLING_DAYS}
        columns = {}
        for days in (None, *ROLLING_DAYS):
            for key, part, v1, coef in (("fit", f["fit"], v1_fit, fit_coefs),
                                        ("val", val, v1_val, coefs)):
                o, s = coef[days] if days else (None, None)
                columns[key, days] = pd.DataFrame(wsa_columns(data, part, params, v1, o, s))
        variants = [("linear shipped", None, None)]
        variants += [(f"linear + {SIGNAL}_lead_{d}h", d, None) for d in LEAD_DELAYS]
        variants += [(f"linear + {SIGNAL}_lead_{d}h rolling {days}d", d, days)
                     for d in LEAD_DELAYS for days in ROLLING_DAYS]
        for name, delay, days in variants:
            cfg = shipped if delay is None else replace(
                shipped, features=(*shipped.features, f"{SIGNAL}_lead_{delay}h"))
            feats = list(cfg.input_features)
            xs = []
            for key, part in (("fit", f["fit"]), ("val", val)):
                frame = base.iloc[part].reset_index(drop=True)
                if delay is not None:
                    frame = pd.concat([frame, columns[key, days]], axis=1)
                xs.append(frame[feats].to_numpy(dtype=float))
            x_fit, x_val = xs
            new = [i for i, n in enumerate(feats) if n.startswith(SIGNAL)]
            if new:
                med = np.nanmedian(x_fit[:, new], axis=0)
                for x in (x_fit, x_val):
                    x[:, new] = np.where(np.isnan(x[:, new]), med, x[:, new])
            model = tl.LinearModel(cfg).fit(folds.values, folds.positions[f["fit"]], x_fit,
                                            end=f["start"])
            fc = model.forecast(folds.values, folds.positions[val], x_val)
            out[name] = cs.score(fc, f["obs"], pers)["s_final"]
        times.append(time.perf_counter() - t)
        rows[year] = out
        path.write_text(json.dumps(out), encoding="utf-8")
        log(f"  {year} done in {times[-1]:.0f} s: " + ", ".join(
            f"{k} {v / f['ref']:.4f}" for k, v in out.items()
            if isinstance(v, float) and k.startswith(("alone", "linear"))))
    df = summarize(rows, sample)
    if times:
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        log(f"\ntiming ({len(data.t0s)} runs): {time.perf_counter() - started:.0f} s in all, "
            f"{np.mean(times):.0f} s per fold, peak traced memory {peak:.0f} MB")
        if sample:
            n_full = len(tl.training_runs(data.series))
            log(f"  estimate for {n_full} runs: ~"
                f"{(len(tl.FOLDS) * np.mean(times) * n_full / len(data.t0s) * 0.6 + t_table * n_full / len(data.t0s)) / 60:.0f}"  # noqa: E501
                " min (calibration runs are capped, the rest scales with runs)")
    return df


def summarize(rows, sample=None):
    years = sorted(rows)
    names = [k for k in rows[years[0]] if k.startswith(("alone", "linear", "persistence"))]
    ratio = pd.DataFrame({y: {n: rows[y][n] / rows[y]["persistence"] for n in names}
                          for y in years})
    ref = ratio.loc["linear shipped"]
    out = {}
    for name, r in ratio.iterrows():
        vs = (r / ref).to_numpy()
        gain = float(np.exp(np.log(vs).mean()) - 1)
        won = int((vs < 1).sum())
        out[name] = {"/ persistence": float(np.exp(np.log(r.to_numpy()).mean())),
                     "vs shipped": gain, "folds won": won,
                     "keep": bool(name.startswith("linear +") and gain < -KEEP_GAIN
                                  and won >= KEEP_FOLDS)}
    table = pd.DataFrame(out).T.sort_values("/ persistence")
    if not sample:
        flat = [{"fold": y, **{k: v for k, v in rows[y].items() if k != "params"},
                 **{f"param {k}": v for k, v in rows[y]["params"].items()}} for y in years]
        pd.DataFrame(flat).to_csv(EXPERIMENTS / "wsa_model.csv", index=False,
                                  float_format="%.6g")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        log(f"\nover {len(years)} folds (official S_final / persistence):\n" + table.to_string())
        log("calibrated parameters per fold:\n"
            + pd.DataFrame({y: rows[y]["params"] for y in years}).T.round(3).to_string())
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        rows = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((CACHE_DIR / "full").glob("*.json"))}
        summarize(rows)
    else:
        run(args.sample, args.seed, args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
