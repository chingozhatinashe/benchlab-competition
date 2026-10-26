"""The official competition score (organisers' spec, docs/timse_implementation.pdf, 2026-10-25).

    from training.competition_score import score
    score(forecasts, observations, persistence)   # (runs x 72), (runs x 72), (runs,)

Per lead tau = 1..72, over every run's (t0, tau) pair whose verifying observation is valid:
    RMSE_tau   root mean squared error
    TIMSE_tau  percentile TIMSE: 101 quantiles q_k of that lead's observations (k = 0..100),
               M_k = MSE over the pairs with v >= q_k,
               TIMSE = sum_k (q_{k+1} - q_k) M_k / (v_max - v_min)   (velocity-width weights)
    TRMSE_tau  sqrt(TIMSE_tau)
    S_tau      sqrt(RMSE_tau * TRMSE_tau)
and the ranking score S_final = the mean of S_tau over the 72 leads, each lead weighted
equally whatever its number of pairs. Leads are never pooled, and runs are not scored one
by one.

Missing values (spec sections 3-4):
- a pair whose observation is missing contributes to nothing;
- a missing or non-finite forecast is replaced by the run's persistence value, and the
  pair stays in (a model can't improve its score by leaving hard cases out). A pair
  whose persistence is missing too can't be scored and is counted in `n_unscorable`.

Ground truth (spec section 2): SWPC's hourly ACE/SWEPAM file YYYYMM_ace_swepam_1h.txt
(Bulk Speed), `parse_monthly` and `MONTHLY_URL`; status != 0 or -9999.9 is missing.

Conventions the spec says must be fixed (section 11, "Reproducibility"), as constants:
QUANTILE_METHOD, UTC times, hours labelled by their start as in the official file, a
duplicate timestamp keeps the last row, and PERSISTENCE_LAG (which hours persistence may use).
"""

import io

import numpy as np
import pandas as pd

HORIZON = 72
N_INTERVALS = 100  # part of the metric's definition (spec 6.1), not a resolution setting
QUANTILE_METHOD = "linear"  # numpy's default; the spec says it must be fixed, not which one
LONG_LEADS = slice(48, 72)  # S_49:72, the long-lead diagnostic (spec section 9)

MONTHLY_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/{month:%Y%m}_ace_swepam_1h.txt"
MISSING_SPEED = -9999.9

# Persistence is "the most recent valid observation available at or before the forecast
# issue time", using no observation after t0 (spec 4, 10.1). The hour labelled t0 averages
# t0 .. t0+59 min and is published after t0, so the latest hour available at t0 is the one
# labelled t0 - 1 h. (The spec's remark that persistence "reduces to v(t0)" if an observation
# exists exactly at t0 would use the hour labelled t0; set PERSISTENCE_LAG to 0 h for that.)
PERSISTENCE_LAG = pd.Timedelta(hours=1)
RECURRENCE = pd.Timedelta(days=27)


# ---------------------------------------------------------------------- ground truth
def parse_monthly(text, any_status=False):
    """Bulk Speed from one YYYYMM_ace_swepam_1h.txt: a float Series indexed by the hour's
    start (UTC), NaN where the status is not 0 or the speed is the fill value.

    The file's status 1-8 means "bad data record" and 9 "no data"; the spec treats
    invalid observations as missing, so only status 0 counts. `any_status=True` keeps
    every non-fill speed whatever its status (a sensitivity check, not the official truth).
    """
    rows = [line for line in text.splitlines()
            if line[:1].isdigit() and len(line.split()) >= 10]
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="speed")
    f = pd.read_csv(io.StringIO("\n".join(rows)), sep=r"\s+", header=None,
                    usecols=range(10), dtype={3: str},
                    names=["yr", "mo", "da", "hhmm", "mjd", "sod", "s", "dens", "speed", "temp"])
    t = pd.to_datetime(
        dict(year=f.yr, month=f.mo, day=f.da,
             hour=f.hhmm.str.zfill(4).str[:2].astype(int),
             minute=f.hhmm.str.zfill(4).str[2:].astype(int)),
        utc=True)
    ok = (f.speed > MISSING_SPEED + 1) & ((f.s == 0) | any_status)
    speed = f.speed.where(ok).to_numpy(dtype=float)
    out = pd.Series(speed, index=pd.DatetimeIndex(t), name="speed")
    return out[~out.index.duplicated(keep="last")].sort_index()


def verifying(truth, t0s, horizon=HORIZON):
    """Observations for every run: (runs x horizon), row i = the hours labelled
    t0s[i] + 1 h .. + horizon h; NaN where missing."""
    t0s = pd.DatetimeIndex(t0s)
    leads = np.arange(1, horizon + 1) * np.timedelta64(1, "h")
    idx = pd.DatetimeIndex((t0s.tz_convert(None).to_numpy()[:, None] + leads).ravel(), tz="UTC")
    return truth.reindex(idx).to_numpy(dtype=float).reshape(len(t0s), horizon)


def persistence(truth, t0s, lag=PERSISTENCE_LAG):
    """The latest valid observation labelled <= t0 - lag, per run (NaN if none)."""
    valid = truth.dropna()
    return valid.asof(pd.DatetimeIndex(t0s) - lag).to_numpy(dtype=float)


def recurrence(truth, t0s, horizon=HORIZON):
    """The 27-day recurrence baseline: v(t0 + tau - 27 d), NaN where that hour is missing
    (score() then uses persistence, as the spec says)."""
    return verifying(truth, pd.DatetimeIndex(t0s) - RECURRENCE, horizon)


# ---------------------------------------------------------------------- the metric
def quantiles(sorted_obs, n_intervals=N_INTERVALS):
    """The n_intervals + 1 empirical quantiles q_k = Q(k / n_intervals) of sorted data,
    QUANTILE_METHOD "linear" (numpy's default: position k (n - 1) / n_intervals, linear
    between neighbours), with the positions in integer arithmetic. np.quantile computes
    them in floating point, so a quantile that falls exactly on an observation can come
    out a hair above it (70.00000000000001), and the spec's v >= q_k would then leave
    that observation out."""
    v = np.asarray(sorted_obs, dtype=float)
    k = np.arange(n_intervals + 1)
    lo, rem = np.divmod(k * (len(v) - 1), n_intervals)
    hi = np.minimum(lo + 1, len(v) - 1)
    return v[lo] + (rem / n_intervals) * (v[hi] - v[lo])


def percentile_timse(obs, fc):
    """TIMSE of one lead's valid pairs, spec section 6 (NaN for fewer than two distinct
    observations, where v_max - v_min is 0)."""
    obs, fc = np.asarray(obs, dtype=float), np.asarray(fc, dtype=float)
    span = obs.max() - obs.min() if len(obs) else 0.0
    if span <= 0:
        return np.nan
    order = np.argsort(obs, kind="stable")
    v, e2 = obs[order], (obs[order] - fc[order]) ** 2
    q = quantiles(v)
    tail = np.concatenate([np.cumsum(e2[::-1])[::-1], [0.0]])  # tail[i] = sum of e2[i:]
    first = np.searchsorted(v, q[:-1], side="left")  # first pair with v >= q_k
    mse = tail[first] / (len(v) - first)  # q_k <= v_max, so every set is non-empty
    return float(np.sum(np.diff(q) * mse) / span)


def lead_scores(forecasts, observations, persistence_values):
    """Per-lead arrays (length = number of leads): n, rmse, timse, trmse, s, bias, and the
    counts of forecasts replaced by persistence and of unscorable pairs."""
    f = np.array(forecasts, dtype=float)
    y = np.asarray(observations, dtype=float)
    p = np.asarray(persistence_values, dtype=float)
    if f.shape != y.shape or f.ndim != 2 or p.shape != (len(f),):
        raise ValueError(f"shapes: forecasts {f.shape}, observations {y.shape}, "
                         f"persistence {p.shape}")
    missing = ~np.isfinite(f)
    f = np.where(missing, p[:, None], f)  # spec 4: replace, don't drop
    has_obs = np.isfinite(y)
    use = has_obs & np.isfinite(f)
    out = {k: np.full(f.shape[1], np.nan) for k in ("rmse", "timse", "bias")}
    out["n"] = use.sum(axis=0)
    out["n_replaced"] = (missing & has_obs).sum(axis=0)
    out["n_unscorable"] = (has_obs & ~np.isfinite(f)).sum(axis=0)
    for j in range(f.shape[1]):
        m = use[:, j]
        if not m.any():
            continue
        err = f[m, j] - y[m, j]
        out["rmse"][j] = np.sqrt(np.mean(err**2))
        out["bias"][j] = np.mean(err)
        out["timse"][j] = percentile_timse(y[m, j], f[m, j])
    out["trmse"] = np.sqrt(out["timse"])
    out["s"] = np.sqrt(out["rmse"] * out["trmse"])
    return out


def score(forecasts, observations, persistence_values):
    """The official score and the diagnostics spec section 9 asks for.

    S_final is the mean of S_tau over the leads that have a score (all 72 in practice),
    each weighted equally."""
    per = lead_scores(forecasts, observations, persistence_values)
    s = per["s"]
    return {
        "s_final": float(np.nanmean(s)),
        "rmse": float(np.nanmean(per["rmse"])),
        "trmse": float(np.nanmean(per["trmse"])),
        "bias": float(np.nanmean(per["bias"])),
        "s_49_72": float(np.nanmean(s[LONG_LEADS])) if len(s) >= HORIZON else np.nan,
        "leads_scored": int(np.isfinite(s).sum()),
        "n_runs": int(len(np.asarray(forecasts))),
        "n_pairs": int(per["n"].sum()),
        "n_replaced": int(per["n_replaced"].sum()),
        "n_unscorable": int(per["n_unscorable"].sum()),
        "curves": {k: per[k].tolist() for k in ("s", "rmse", "trmse", "bias", "n")},
    }
