"""Structural model (ambient wind + stream arrivals, models.structural) on the yearly folds.

    python -m training.structural --pilot --folds 2024   # one fold, timed, not cached
    python -m training.structural [--max-seconds N]      # every fold; resumable
    python -m training.structural --summary

The question (user, 2026-11-08): can one model, trained in one step, impose a structure on
the wind: (i) a steady ambient (slow) wind, (ii) stream arrivals with a timing, duration
and speed? models.structural writes it as one differentiable forecast equation; here all
its parameters are fitted together (PyTorch L-BFGS, float64, dev only) by minimising the
linear model's loss (official weights, blend 0.25, despiked targets) over every fit run
and lead, from physical starting values with HAZARDS restarts (the lowest loss is kept).

Variants (the submitted config's regressors and fixes; refitted per fold on the runs
before it, PFSS gaps -> the fit runs' median as in training.model_diagnostics):
    structural            the full model
    no arrivals           ambient + decay of the current excess only
    no PFSS               the current-sheet distance left out of x(u)
    fixed decay           one tau_c for every run
    linear + structural   the submitted linear model with the structural forecast as a
                          lead-aligned regressor (structural_speed_lead_0h; the fit runs get the
                          structural model's in-sample forecasts: ~40 parameters)
    reference             the submitted linear model (0.7794 of persistence)
Score: official S_final / persistence, geometric mean over the six yearly folds, plus the
run regimes (training.model_diagnostics). Decision rule, set before running: keep if > 1%
better than the reference and winning >= 4 of 6 folds. The eval year is not used.
Results: src/experiments/structural_model.csv and structural_model_params.csv
(cache: src/data/structural_cache/, with the 2024 fold's forecast parts for the notebook).
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from models import structural as st
from models.linear_model import LinearModel, lag_column, lead_lags
from training import competition_score as cs
from training import model_diagnostics as dg
from training import xgboost_model as tx
from training.folds import YEARLY

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "structural_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
HAZARDS = (-6.0, -4.5)
MAX_ITER = 400
DHCS = "pfss_d_hcs_lead_72h"
SIGNAL = "structural_speed"
VARIANTS = {
    "structural": {},
    "no arrivals": {"arrivals": False},
    "no PFSS": {"exclude": (DHCS,)},
    "fixed decay": {"decay_inputs": False},
}
ALL = ("reference", *VARIANTS, "linear + structural")


def log(msg):
    print(msg, flush=True)


def torch_ops():
    import torch

    return SimpleNamespace(
        exp=torch.exp, sigmoid=torch.sigmoid, softplus=torch.nn.functional.softplus,
        cumsum=lambda x, axis: torch.cumsum(x, dim=axis), clip=torch.clamp,
        where=torch.where, einsum=torch.einsum, max=torch.max)


def to_torch(a):
    """numpy -> float64 tensor via DLPack (the dev venv's torch 2.3 wheel predates numpy 2's
    C API, so torch.as_tensor(ndarray) fails there)."""
    import torch

    return torch.from_dlpack(np.ascontiguousarray(a, dtype=np.float64))


def to_numpy(t):
    return np.from_dlpack(t.detach().contiguous()).copy()


def structural_config(cfg, spec=None):
    """models.structural.Config with the submitted linear config's regressors and fixes."""
    base = st.Config(features=tuple(cfg.features), optional=tuple(cfg.optional),
                     cap_quantile=cfg.cap_quantile, blend=cfg.blend,
                     lags_days=tuple(cfg.lags_days), half_width=cfg.half_width,
                     despike=cfg.despike)
    return replace(base, **(spec or {}))


def fit_inputs(cfg, inp, y, q, hazards=HAZARDS, max_iter=MAX_ITER):
    """(params, info): every parameter fitted in one optimisation on prepared inputs."""
    import torch

    t = {k: to_torch(inp[k]) for k in ("x", "amb", "last", "leads", "lag", "ahead")}
    yt, qt = to_torch(y), to_torch(q)
    qsum = float(np.sum(q))
    mask = {k: to_torch(v) for k, v in st.penalty_mask(cfg).items()}
    ops = torch_ops()

    def loss(p):
        v = st.forward(p, t, cfg, ops)
        fit_term = (qt * (yt - v) ** 2).sum() / qsum / st.LOSS_SCALE
        return fit_term + cfg.ridge * sum((mask[k] * p[k] ** 2).sum() for k in p)

    best, info = None, {"restarts": []}
    for hazard in hazards if cfg.arrivals else hazards[:1]:
        p = {k: to_torch(v).clone().requires_grad_(True)
             for k, v in st.initial_params(cfg, hazard).items()}
        opt = torch.optim.LBFGS(list(p.values()), lr=1.0, max_iter=max_iter, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                tolerance_change=1e-12)

        def closure(p=p, opt=opt):
            opt.zero_grad()
            value = loss(p)
            value.backward()
            return value

        opt.step(closure)
        with torch.no_grad():
            value = float(loss(p))
        n_iter = opt.state[opt._params[0]].get("n_iter", 0)
        info["restarts"].append({"hazard0": hazard, "loss": value, "iterations": n_iter})
        if best is None or value < best[0]:
            best = (value, {k: to_numpy(v) for k, v in p.items()})
    info["loss"] = best[0]
    info["rmse_weighted"] = float(np.sqrt(best[0] * st.LOSS_SCALE))
    return best[1], info


def fit(model, values, positions, regressors, end=None, hazards=HAZARDS, max_iter=MAX_ITER):
    """Fit every parameter of `model` in one optimisation; returns the fit info."""
    model.set_statistics(values, positions, regressors, end)
    inp = model.prepare(values, positions, regressors)
    y, q = model.training_targets(values, positions, end, inp["has_last"])
    model.params, info = fit_inputs(model.config, inp, y, q, hazards, max_iter)
    return info


def describe(model):
    """Readable fitted quantities (typical decay and profile times, hazard, amplitude)."""
    p = model.params
    sp = lambda z: float(np.logaddexp(0.0, z))  # noqa: E731
    return {"tau_c0 h": 1 + st.TAU_SCALE * sp(p["dec"][0]),
            "tau_neg h": 1 + st.TAU_SCALE * sp(p["neg"][0]),
            "rise h": 1 + st.RISE_SCALE * sp(p["rise"][0]),
            "stream decay h": 1 + st.TAU_SCALE * sp(p["sdec"][0]),
            "hazard0 /h": float(1 / (1 + np.exp(-p["haz"][0]))),
            "amplitude0 km/s": st.AMP_SCALE * sp(p["amp"][0]),
            "ambient0 km/s": float(st.AMBIENT_RANGE[0] + np.diff(st.AMBIENT_RANGE)[0]
                                   / (1 + np.exp(-p["amb"][0])))}


def structural_columns(f):
    """{"structural_at_<L>h": forecast at lead max(-L, 1)} for the lead-aligned regressor."""
    lags = np.unique(lead_lags(0, SIGNAL))
    return {lag_column(SIGNAL, lag): f[:, max(-int(lag), 1) - 1] for lag in lags}


def score_fold(data, start, end, names, timer):
    start_pos = data.series.index.get_loc(start)
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
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"], "fits": {}}
    forecasts = {}
    for name in names:
        t = time.perf_counter()
        if name == "reference":
            lin = LinearModel(data.cfg).fit(data.values, pos_fit, x_fit, end=start_pos)
            f = lin.forecast(data.values, pos_val, x_val)
        elif name == "linear + structural":
            model = st.StructuralModel(structural_config(data.cfg))
            fit(model, data.values, pos_fit, x_fit, end=start_pos)
            s_fit = structural_columns(model.forecast(data.values, pos_fit, x_fit))
            s_val = structural_columns(model.forecast(data.values, pos_val, x_val))
            cfg2 = replace(data.cfg, features=(*data.cfg.features, f"{SIGNAL}_lead_0h"))
            c2 = list(cfg2.input_features)
            t_fit = table.loc[fit6].assign(**s_fit)
            t_val = table.loc[val].assign(**s_val)
            lin = LinearModel(cfg2).fit(data.values, pos_fit,
                                        t_fit[c2].to_numpy(dtype=float), end=start_pos)
            f = lin.forecast(data.values, pos_val, t_val[c2].to_numpy(dtype=float))
        else:
            model = st.StructuralModel(structural_config(data.cfg, VARIANTS[name]))
            info = fit(model, data.values, pos_fit, x_fit, end=start_pos)
            f, parts = model.forecast(data.values, pos_val, x_val, parts=True)
            out["fits"][name] = {**info, **describe(model),
                                 "params": {k: v.tolist() for k, v in model.params.items()}}
            if name == "structural":
                forecasts["parts"] = parts
        forecasts[name] = f
        row = dg.regime_scores(f, obs, pers, regime)
        row["residual slow wind"] = float(np.nanmean((obs - f)[regime == "slow wind"]))
        row["residual all"] = float(np.nanmean(obs - f))
        out[name] = row
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
        log(f"    {name}: S_final {row['all']:.2f} ({time.perf_counter() - t:.0f} s)")
    if "structural" in forecasts and "reference" in forecasts:
        parts = forecasts["parts"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_DIR / f"{start:%Y}_parts.npz", t0s=val.asi8, obs=obs,
                            pers=pers, regime=regime, f_ref=forecasts["reference"],
                            f_struct=forecasts["structural"], ambient=parts["ambient"],
                            current=parts["current"], stream=parts["stream"])
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
            f"{6 * sum(timer.values()) / max(len(rows), 1) / 60:.0f} min after the data")
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
            "slow residual": float(np.mean([rows[y][name]["residual slow wind"]
                                            for y in years])),
            "keep": bool(gain < -KEEP_GAIN and won >= KEEP_FOLDS),
        }
        long += [{"fold": y, "variant": name, "persistence": rows[y]["persistence"],
                  **rows[y][name]} for y in years]
        params += [{"fold": y, "variant": name,
                    **{k: v for k, v in rows[y]["fits"][name].items()
                       if k not in ("params", "restarts")},
                    "restart losses": " / ".join(f"{r['loss']:.5f}"
                                                 for r in rows[y]["fits"][name]["restarts"])}
                   for y in years if name in rows[y].get("fits", {})]
    table = pd.DataFrame(out).T.sort_values("vs reference")
    if not pilot:
        pd.DataFrame(long).to_csv(EXPERIMENTS / "structural_model.csv", index=False,
                                  float_format="%.6g")
        pd.DataFrame(params).to_csv(EXPERIMENTS / "structural_model_params.csv", index=False,
                                    float_format="%.6g")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 30):
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
