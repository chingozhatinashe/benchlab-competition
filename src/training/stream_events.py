"""Structural model v2 (models.stream_events) on the yearly folds.

    python -m training.stream_events --pilot --folds 2024   # one fold, timed, not cached
    python -m training.stream_events [--max-seconds N]      # every fold; resumable
    python -m training.stream_events --summary

Follow-up to docs/structural_model.md (user, 2026-11-09): each stream event has (i) an
arrival time, a duration and speeds over that duration without an assumed shape; (ii)
several events may occur in the 72 h and overlapping effects add. Fitted in one step
(torch L-BFGS, float64, dev only) on

    speed loss (official weights, blend 0.25, despiked; as v1)
    + beta * Poisson NLL of the detected stream arrivals (training.solar_wind_events)

The arrival term anchors the rate apart from the size (with additive events the expected
speed only fixes their product). Labels: 1 at the detected arrival hours; CME and
possible-CME hours and every hour from LABEL_GUARD h before the fold start on get weight 0
(detection looks up to 48 h ahead). The event tensor is evaluated in chunks of CHUNK runs;
the event models fit on every FIT_STEP-th fit run (12-hourly) to bound time and memory.

Variants (the submitted config's regressors and fixes, refitted per fold):
    events               beta = 1
    events speed only    beta = 0
    single event         beta = 1, only the first arrival counts (isolates change ii)
    events no PFSS       beta = 1, the current-sheet distance left out
    linear + events      the submitted linear model with the events forecast as a
                         lead-aligned regressor (structural_speed_lead_0h)
    reference            the submitted linear model (0.7794 of persistence)
Also per fold: the arrival skill on the validation runs (Poisson log-likelihood skill
against the fit runs' constant rate; AUC of P(>= 1 arrival in 72 h)) and the fitted
quantities. Decision rule, set before running: keep if > 1% better than the reference and
winning >= 4 of 6 folds. The eval year is not used.
Results: src/experiments/stream_events.csv and stream_events_params.csv
(cache: src/data/stream_events_cache/, with the 2024 fold's parts for the notebook).
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from models import stream_events as se
from models import structural as st
from models.linear_model import LinearModel
from training import competition_score as cs
from training import model_diagnostics as dg
from training import structural as ts
from training import xgboost_model as tx
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "stream_events_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
LOG_RATES = (-4.8,)
MAX_ITER = 100
CHUNK = 500  # runs per chunk of the event tensor (memory)
FIT_STEP = 2  # every second fit run (12-hourly) for the event model's fit
LABEL_GUARD = 48  # h
DHCS = "pfss_d_hcs_lead_72h"
VARIANTS = {
    "events": {},
    "events speed only": {"beta": 0.0},
    "single event": {"single": True},
    "events no PFSS": {"exclude": (DHCS,)},
}
ALL = ("reference", *VARIANTS, "linear + events")


def log(msg):
    print(msg, flush=True)


def events_config(cfg, spec=None):
    base = ts.structural_config(cfg)
    fields = {k: getattr(base, k) for k in st.Config.__dataclass_fields__}
    return replace(se.Config(**fields), **(spec or {}))


def torch_ops():
    import torch

    base = vars(ts.torch_ops())
    return type(ts.torch_ops())(**base, floor=torch.floor, take=lambda v, i: v[i.long()],
                                concat=torch.cat, stack=torch.stack, log=torch.log)


def penalty(p, cfg, mask):
    """Ridge on the input weights; second differences of the profile knots."""
    total = cfg.ridge * sum((mask[k] * p[k] ** 2).sum() for k in p)
    for k in ("phi", "psi"):
        if len(p[k]) >= 3:
            d2 = p[k][2:] - 2 * p[k][1:-1] + p[k][:-2]
            total = total + cfg.smooth * (d2 ** 2).sum()
    return total


def fit_inputs(cfg, inp, y, q, n_lab, m_lab, log_rates=LOG_RATES, max_iter=MAX_ITER,
               chunk=CHUNK):
    """(params, info): every parameter fitted in one optimisation, the loss evaluated in
    chunks of runs with the gradients accumulated."""
    import torch

    t = {k: ts.to_torch(inp[k]) for k in ("x", "amb", "last")}
    leads = ts.to_torch(inp["leads"])
    yt, qt, nt, mt = (ts.to_torch(a) for a in (y, q, n_lab, m_lab))
    qsum, msum = float(np.sum(q)), max(float(np.sum(m_lab)), 1.0)
    mask = {k: ts.to_torch(v) for k, v in se.penalty_mask(cfg).items()}
    ops = torch_ops()
    n = len(y)
    slices = [slice(i, min(i + chunk, n)) for i in range(0, n, chunk)]

    def parts_of(p, sl):
        v, parts = se.forward(p, {"x": t["x"][sl], "amb": t["amb"][sl],
                                  "last": t["last"][sl], "leads": leads}, cfg, ops, parts=True)
        speed = (qt[sl] * (yt[sl] - v) ** 2).sum() / qsum / st.LOSS_SCALE
        eta = parts["eta"]
        events = (mt[sl] * (torch.exp(eta) - nt[sl] * eta)).sum() / msum
        return speed, events

    def total_loss(p, backward):
        speed_sum = events_sum = 0.0
        for sl in slices:
            speed, events = parts_of(p, sl)
            value = speed + cfg.beta * events
            if backward:
                value.backward()
            speed_sum += float(speed)
            events_sum += float(events)
        reg = penalty(p, cfg, mask)
        if backward:
            reg.backward()
        return speed_sum + cfg.beta * events_sum + float(reg), speed_sum, events_sum

    best, info = None, {"restarts": []}
    for rate in log_rates:
        p = {k: ts.to_torch(v).clone().requires_grad_(True)
             for k, v in se.initial_params(cfg, rate).items()}
        opt = torch.optim.LBFGS(list(p.values()), lr=1.0, max_iter=max_iter, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                tolerance_change=1e-12)

        def closure(p=p, opt=opt):
            opt.zero_grad()
            return torch.tensor(total_loss(p, True)[0], dtype=torch.float64)

        opt.step(closure)
        with torch.no_grad():
            value, speed, events = total_loss(p, False)
        n_iter = opt.state[opt._params[0]].get("n_iter", 0)
        info["restarts"].append({"log_rate0": rate, "loss": value, "iterations": n_iter})
        if best is None or value < best[0]:
            best = (value, {k: ts.to_numpy(v) for k, v in p.items()}, speed, events)
    info.update(loss=best[0], speed_loss=best[2], event_nll=best[3],
                rmse_weighted=float(np.sqrt(best[2] * st.LOSS_SCALE)))
    return best[1], info


def labels(data, t0s, cutoff=None):
    """(n, m): 1 at detected stream-arrival hours t0 + 1..72; weight 0 in CME and
    possible-CME hours and from `cutoff` (a timestamp) on."""
    full = data.full.index
    hours = pd.DatetimeIndex(data.arrivals).floor("h")
    flag = full.isin(hours)
    blocked = (data.cme | data.possible).to_numpy()
    idx = full.get_indexer(t0s)[:, None] + np.arange(1, cs.HORIZON + 1)
    inside = idx < len(full)
    safe = np.where(inside, idx, 0)
    n = np.where(inside, flag[safe], False).astype(float)
    m = (inside & ~blocked[safe]).astype(float)
    if cutoff is not None:
        m = m * (idx < full.searchsorted(cutoff))
    return n, m


def fit(model, data, positions, t0s, regressors, end, cutoff, step=FIT_STEP, **kw):
    positions, t0s, regressors = positions[::step], t0s[::step], regressors[::step]
    model.set_statistics(data.values, positions, regressors, end)
    inp = model.prepare(data.values, positions, regressors)
    y, q = model.training_targets(data.values, positions, end, inp["has_last"])
    n_lab, m_lab = labels(data, t0s, cutoff)
    model.params, info = fit_inputs(model.config, inp, y, q, n_lab, m_lab, **kw)
    info["label_rate"] = float((n_lab * m_lab).sum() / max(m_lab.sum(), 1))
    return info


def arrival_skill(parts, n_lab, m_lab, rate):
    """Poisson log-likelihood skill against a constant rate; AUC of P(>= 1 in 72 h)."""
    eta = parts["eta"]
    lam = np.exp(eta)
    ll = (m_lab * (n_lab * eta - lam)).sum()
    ll0 = (m_lab * (n_lab * np.log(rate) - rate)).sum()
    any_obs = (n_lab * m_lab).sum(axis=1) > 0
    p_any = 1 - np.exp(-(lam * m_lab).sum(axis=1))
    auc = roc_auc_score(any_obs, p_any) if 0 < any_obs.sum() < len(any_obs) else np.nan
    return {"arrival LL skill": float((ll - ll0) / abs(ll0)), "arrival AUC": float(auc),
            "arrivals observed": int((n_lab * m_lab).sum()),
            "arrivals expected": float((lam * m_lab).sum())}


def describe(model, parts):
    p = model.params
    phi, psi = se.profiles(p, se.NUMPY)
    return {"lambda mean /h": float(np.exp(parts["eta"]).mean()),
            "duration mean h": float(parts["duration"].mean()),
            "amplitude mean km/s": float(parts["amplitude"].mean()),
            "ambient mean km/s": float(parts["ambient"].mean()),
            "phi": [round(float(v), 4) for v in phi], "psi": [round(float(v), 4) for v in psi]}


def score_fold(data, start, end, names, timer):
    start_pos = data.series.index.get_loc(start)
    cutoff = start - pd.Timedelta(hours=LABEL_GUARD)
    fit6 = data.runs6[data.runs6 < start]
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    regime = data.run_regimes(val)
    table = data.table.copy()
    table[data.pfss] = table[data.pfss].fillna(table.loc[fit6, data.pfss].median())
    pos_fit, pos_val = data.series.index.get_indexer(fit6), data.series.index.get_indexer(val)
    cols = list(data.cfg.input_features)
    x_fit = table.loc[fit6, cols].to_numpy(dtype=float)
    x_val = table.loc[val, cols].to_numpy(dtype=float)
    n_val, m_val = labels(data, val)
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"], "fits": {}}
    forecasts = {}
    for name in names:
        t = time.perf_counter()
        if name == "reference":
            lin = LinearModel(data.cfg).fit(data.values, pos_fit, x_fit, end=start_pos)
            f = lin.forecast(data.values, pos_val, x_val)
        elif name == "linear + events":
            model = se.StreamEventsModel(events_config(data.cfg))
            fit(model, data, pos_fit, fit6, x_fit, start_pos, cutoff)
            s_fit = ts.structural_columns(model.forecast(data.values, pos_fit, x_fit))
            s_val = ts.structural_columns(model.forecast(data.values, pos_val, x_val))
            cfg2 = replace(data.cfg, features=(*data.cfg.features, f"{ts.SIGNAL}_lead_0h"))
            c2 = list(cfg2.input_features)
            lin = LinearModel(cfg2).fit(data.values, pos_fit, table.loc[fit6].assign(**s_fit)[
                c2].to_numpy(dtype=float), end=start_pos)
            f = lin.forecast(data.values, pos_val,
                             table.loc[val].assign(**s_val)[c2].to_numpy(dtype=float))
        else:
            model = se.StreamEventsModel(events_config(data.cfg, VARIANTS[name]))
            info = fit(model, data, pos_fit, fit6, x_fit, start_pos, cutoff)
            f, parts = model.forecast(data.values, pos_val, x_val, parts=True)
            out["fits"][name] = {**info, **describe(model, parts),
                                 **arrival_skill(parts, n_val, m_val, info["label_rate"]),
                                 "params": {k: v.tolist() for k, v in model.params.items()}}
            if name == "events":
                forecasts["parts"] = parts
        forecasts[name] = f
        row = dg.regime_scores(f, obs, pers, regime)
        row["residual slow wind"] = float(np.nanmean((obs - f)[regime == "slow wind"]))
        row["residual all"] = float(np.nanmean(obs - f))
        out[name] = row
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
        log(f"    {name}: S_final {row['all']:.2f} ({time.perf_counter() - t:.0f} s)")
    if "events" in forecasts and "reference" in forecasts:
        parts = forecasts["parts"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_DIR / f"{start:%Y}_parts.npz", t0s=val.asi8, obs=obs,
                            pers=pers, regime=regime, f_ref=forecasts["reference"],
                            f_events=forecasts["events"], n_lab=n_val, m_lab=m_val,
                            **{k: parts[k] for k in ("ambient", "current", "stream", "eta",
                                                     "duration", "amplitude")})
    return out


def run(years=None, pilot=False, max_seconds=None, names=ALL):
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
        row = {} if pilot or not path.exists() else json.loads(path.read_text("utf-8"))
        todo = [n for n in names if n not in row]
        if todo:
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run the same command again to continue")
                return None
            t = time.perf_counter()
            log(f"  {year}:")
            new = score_fold(data, start, end, todo, timer)
            fits = {**row.get("fits", {}), **new.pop("fits")}
            row.update(new)
            row["fits"] = fits
            if not pilot:
                path.write_text(json.dumps(row), encoding="utf-8")
            log(f"  {year}: {len(todo)} variants in {time.perf_counter() - t:.0f} s")
        rows[year] = row
    if pilot:
        log("  time: " + ", ".join(f"{k} {v:.0f} s" for k, v in timer.items())
            + f"; peak memory {tx.peak_mb():.0f} MB; six folds -> ~"
            f"{6 * sum(timer.values()) / max(len(rows), 1) / 60:.0f} min after the data "
            "(later folds have fewer fit runs)")
    return summarize(rows, pilot)


def summarize(rows, pilot=False):
    years = sorted(rows)
    names = [n for n in ALL if all(n in rows[y] for y in years)]
    out, long, params = {}, [], []
    for name in names:
        rel = np.array([rows[y][name]["all"] / rows[y]["reference"]["all"] for y in years])
        gain = float(np.exp(np.log(rel).mean()) - 1)
        won = int((rel < 1).sum())
        out[name] = {
            "/ persistence": float(np.exp(np.mean([np.log(rows[y][name]["all"]
                                                          / rows[y]["persistence"])
                                                   for y in years]))),
            "vs reference": gain, "folds won": won,
            **{f"{r} vs reference": float(np.exp(np.nanmean(
                [np.log(rows[y][name][r] / rows[y]["reference"][r]) for y in years])) - 1)
               for r in dg.RUN_REGIMES},
            "keep": bool(gain < -KEEP_GAIN and won >= KEEP_FOLDS),
        }
        long += [{"fold": y, "variant": name, "persistence": rows[y]["persistence"],
                  **rows[y][name]} for y in years]
        params += [{"fold": y, "variant": name,
                    **{k: (" ".join(f"{v:.2f}" for v in val) if isinstance(val, list) else val)
                       for k, val in rows[y]["fits"][name].items()
                       if k not in ("params", "restarts")}}
                   for y in years if name in rows[y].get("fits", {})]
    table = pd.DataFrame(out).T.sort_values("vs reference")
    if not pilot:
        pd.DataFrame(long).to_csv(EXPERIMENTS / "stream_events.csv", index=False,
                                  float_format="%.6g")
        pd.DataFrame(params).to_csv(EXPERIMENTS / "stream_events_params.csv", index=False,
                                    float_format="%.6g")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 30, "display.max_colwidth", 80):
        log(f"\nfolds {', '.join(years)}:\n{table.to_string()}")
        if params:
            log("\nfitted:\n" + pd.DataFrame(params).to_string(index=False))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--variants", nargs="*", default=list(ALL))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        rows = {p.stem: json.loads(p.read_text("utf-8")) for p in sorted(CACHE_DIR.glob("*.json"))}
        summarize(rows)
        return
    run(args.folds, args.pilot, args.max_seconds, args.variants)


if __name__ == "__main__":
    sys.exit(main())
