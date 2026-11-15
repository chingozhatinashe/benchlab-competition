"""Structural model v3 (models.bayes_stream): a Bayesian generative model, on the yearly folds.

    python -m training.bayes_stream --pilot --folds 2024   # one fold, timed, not cached
    python -m training.bayes_stream [--max-seconds N]      # every fold; resumable
    python -m training.bayes_stream --summary

Follow-up to docs/structural_model.md and docs/stream_events.md (user, 2026-11-09): a slow
wind component that can be sampled as a 72 h path, and a hierarchical stream component
(the number of arrivals, then their times by stick-breaking, then each event's peak, rise
and decay, then the 72 h speed). Posterior by NumPyro NUTS (dev only: jax, numpyro).

Training windows: fit runs every 72 h (non-overlapping, so each hour counts once and the
likelihood is exact given each window's start), without CME or possible-CME hours in
t0..t0+84 h, with at least MIN_OBSERVED target hours (missing hours are skipped exactly in
the AR(1) likelihood), and ending LABEL_GUARD h before the fold start
(the arrival detector looks up to 48 h ahead). Likelihood: the observed speed minus the
model's mean path is AR(1) Gaussian (exact: the model adds its noise after the
streams), and

    observed   the detected arrivals in hours 1..84 are data: Poisson count, stick-breaking
               times, latent marks per event (peak excess, rise width, decay) with
               population priors (partial pooling)
    latent     arrivals summed out (N <= 1 per window; arrival times in LATENT_STEP-hour
               bins at their middle hour, 28 bins + none); marks at their population values
               (no event effects, which the labels would be needed for)
NUTS runs in jax's default float32, with the chains in parallel (time; see the pilot).

Variants (refitted per fold; the submitted config's inputs, PFSS gaps at the fit median):
    bayes              observed arrivals, smooth max, predictive mean
    bayes official     the same posterior, Bayes action under the official-weighted loss
    bayes additive     observed arrivals, additive streams, predictive mean
    bayes latent       latent arrivals, smooth max, predictive mean
    slow wind only     bayes' posterior with N = 0
    linear + bayes     the submitted linear model + bayes' mean as a lead-aligned regressor
    reference          the submitted linear model
Also per fold: arrival skill on the validation runs, CRPS and 80% coverage, posterior
summaries, NUTS diagnostics (R-hat, ESS, divergences) and posterior predictive checks on the
training windows. Decision rule, set before running: keep if > 1% better than the reference
and winning >= 4 of 6 folds. The eval year is not used.
Results: src/experiments/bayes_stream.csv and bayes_stream_params.csv (cache:
src/data/bayes_stream_cache/, with each fold's validation parts and posterior draws).
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from models import bayes_stream as bs
from models.linear_model import LinearModel
from training import competition_score as cs
from training import model_diagnostics as dg
from training import structural as ts
from training import xgboost_model as tx
from training.folds import YEARLY
from training.stream_events import LABEL_GUARD
from training.stream_events import labels as run_labels

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent / "experiments"
CACHE_DIR = HERE.parent / "data" / "bayes_stream_cache"
KEEP_GAIN, KEEP_FOLDS = 0.01, 4
WARMUP, SAMPLES, CHAINS = 500, 500, 2
N_PATHS, N_PATHS_FIT = 400, 100
MIN_OBSERVED = 36  # h of the 72 observed, for a training window
LATENT_STEP = 3  # h: arrival bins summed out in the latent fit (84 / 3 = 28)
FITS = {"observed": bs.Config(), "additive": bs.Config(rule="add"),
        "latent": bs.Config(nmax=1, latent=True)}
VARIANTS = {  # name: (fit, decision, slow only)
    "bayes": ("observed", "mean", False),
    "bayes official": ("observed", "official", False),
    "bayes additive": ("additive", "mean", False),
    "bayes latent": ("latent", "mean", False),
    "slow wind only": ("observed", "mean", True),
}
ALL = ("reference", *VARIANTS, "linear + bayes")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------- training windows
def window_labels(data, t0s, hours=bs.N_ARRIVAL):
    """(n, hours) detected arrivals at t0 + 1..hours, whether t0..t0 + hours touches a CME,
    possible-CME or missing hour, and the hour positions (n, hours + 1)."""
    full = data.full.index
    flag = full.isin(pd.DatetimeIndex(data.arrivals).floor("h"))
    blocked = (data.cme | data.possible).to_numpy()
    idx = full.get_indexer(t0s)[:, None] + np.arange(hours + 1)
    inside = (idx >= 0) & (idx < len(full))
    safe = np.where(inside, idx, 0)
    n = np.where(inside, flag[safe], False)[:, 1:].astype(float)
    bad = (~inside | blocked[safe]).any(axis=1)
    return n, bad, idx


def training_windows(data, model, table, cols, fit6, start_pos, cutoff):
    """Inputs, targets and arrival labels of the training windows (see the docstring)."""
    hours = (fit6 - fit6[0]) / pd.Timedelta(hours=1)
    t0s = fit6[(hours % cs.HORIZON) == 0]
    pos = data.series.index.get_indexer(t0s)
    inp = model.prepare(data.values, pos, table.loc[t0s, cols].to_numpy(dtype=float))
    y, q = model.training_targets(data.values, pos, start_pos, inp["has_last"])
    n_lab, bad, idx = window_labels(data, t0s)
    observed = q > 0
    keep = (~bad & (observed.sum(axis=1) >= MIN_OBSERVED)
            & (idx[:, -1] < data.full.index.searchsorted(cutoff)) & (n_lab.sum(axis=1) <= 2))
    d = window_data(inp["x"][keep], inp["amb"][keep], inp["last"][keep], y[keep], n_lab[keep],
                    observed[keep])
    sim_inp = {"x": inp["x"][keep], "amb": inp["amb"][keep], "last": inp["last"][keep]}
    return d, sim_inp, {"windows": int(len(t0s)), "windows kept": int(keep.sum()),
                        "events": int(len(d["ev_win"]))}


def window_data(x, amb, v0, y, n_lab, observed=None):
    """The NumPyro model's data: inputs (x held to 84 h), targets (missing where not
    `observed`) and arrivals (n, 84)."""
    observed = np.ones(y.shape, dtype=bool) if observed is None else observed
    prev, lag, obs = bs.gap_structure(observed)
    win, hour = np.nonzero(n_lab)
    slot = np.zeros(len(win), dtype=int)
    slot[1:] = (win[1:] == win[:-1]).astype(int)
    tau = np.ones((len(y), 2))
    tau[win, slot] = hour + 1
    count = n_lab.sum(axis=1)
    return {"x": bs.extend(x), "xbar": x.mean(axis=1), "amb": amb, "v0": v0, "y": y,
            "N": count.astype(int), "tau": tau,
            "present": np.stack([count >= 1, count >= 2], axis=1).astype(float),
            "ev_win": win, "ev_slot": slot, "ev_tau": hour + 1, "prev": prev, "lag": lag,
            "obs": obs}


# ---------------------------------------------------------------------- NumPyro model
def numpyro_model(d, cfg):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    p = d["x"].shape[-1]
    n = len(d["y"])

    def vec(name, loc0, scale0, scale):
        loc = jnp.zeros(p).at[0].set(loc0)
        sd = jnp.full(p, scale).at[0].set(scale0)
        return numpyro.sample(name, dist.Normal(loc, sd))

    w_a = numpyro.sample("w_a", dist.Normal(jnp.zeros(3), 1.5))
    rho_up = numpyro.sample("rho_up", dist.Beta(19.0, 1.0))
    rho_dn = numpyro.sample("rho_dn", dist.Beta(19.0, 1.0))
    rho_eta = numpyro.sample("rho_eta", dist.Beta(8.0, 2.0))
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(50.0))
    w_N = vec("w_N", np.log(0.25), 1.0, 0.5)
    w_tau = vec("w_tau", -3.7, 1.5, 0.5)
    mu_E = numpyro.sample("mu_E", dist.Normal(np.log(250.0), 0.5))
    w_E = numpyro.sample("w_E", dist.Normal(jnp.zeros(p - 1), 0.3))
    mu_R = numpyro.sample("mu_R", dist.Normal(np.log(6.0), 0.5))
    mu_D = numpyro.sample("mu_D", dist.Normal(np.log(48.0), 0.5))
    w_D = numpyro.sample("w_D", dist.Normal(jnp.zeros(p - 1), 0.3))

    gaps = (d["prev"], d["lag"], d["obs"])
    a = bs.ambient(w_a, d["amb"], jnp)
    slow = bs.slow_mean(a, d["v0"], rho_up, rho_dn, jnp)
    clp = bs.count_logp(d["xbar"] @ w_N, cfg.nmax, jnp)
    lp = bs.stick_logp(d["x"] @ w_tau, jnp)
    xc = d["x"][:, :, 1:]

    if cfg.latent:  # an arrival in each LATENT_STEP-hour bin, at its middle hour
        hours = bs.ARRIVALS[LATENT_STEP // 2::LATENT_STEP]
        xb = xc[:, LATENT_STEP // 2::LATENT_STEP]
        lp = bs._logsumexp(lp.reshape(n, -1, LATENT_STEP), -1, jnp)
        E = jnp.exp(mu_E + xb @ w_E)  # (n, bins)
        D = jnp.exp(mu_D + xb @ w_D)
        R = jnp.exp(mu_R) * jnp.ones_like(E)
        excess = bs.profile(jnp.broadcast_to(hours, E.shape), E, R, D, jnp)
        mean = bs.combine(jnp.broadcast_to(slow[:, None, :], excess.shape), a[:, None],
                          excess[:, :, None, :],
                          jnp.ones(E.shape + (1,)), cfg, jnp)
        gaps_u = (d["prev"][:, None], d["lag"][:, None], d["obs"][:, None])
        ll_u = bs.ar1_loglik(d["y"][:, None, :] - mean, rho_eta, sigma_eta, jnp, gaps_u)
        ll0 = bs.ar1_loglik(d["y"] - slow, rho_eta, sigma_eta, jnp, gaps)
        ll1 = bs._logsumexp(lp + ll_u, -1, jnp)
        numpyro.factor("speed", jnp.logaddexp(clp[:, 0] + ll0, clp[:, 1] + ll1).sum())
        return

    sigma_E = numpyro.sample("sigma_E", dist.HalfNormal(0.5))
    sigma_R = numpyro.sample("sigma_R", dist.HalfNormal(0.5))
    sigma_D = numpyro.sample("sigma_D", dist.HalfNormal(0.5))
    win, slot, ev_tau = d["ev_win"], d["ev_slot"], d["ev_tau"]
    xe = xc[win, ev_tau - 1]
    with numpyro.plate("events", len(win)):  # centred: each event's speeds pin its marks
        log_E = numpyro.sample("log_E", dist.Normal(mu_E + xe @ w_E, sigma_E))
        log_R = numpyro.sample("log_R", dist.Normal(mu_R, sigma_R))
        log_D = numpyro.sample("log_D", dist.Normal(mu_D + xe @ w_D, sigma_D))
    E = jnp.zeros((n, 2)).at[win, slot].set(jnp.exp(log_E))
    R = jnp.ones((n, 2)).at[win, slot].set(jnp.exp(log_R))
    D = jnp.ones((n, 2)).at[win, slot].set(jnp.exp(log_D))
    mean = bs.combine(slow, a, bs.profile(d["tau"], E, R, D, jnp), d["present"], cfg, jnp)
    numpyro.factor("speed", bs.ar1_loglik(d["y"] - mean, rho_eta, sigma_eta, jnp, gaps).sum())
    numpyro.factor("count", clp[np.arange(n), d["N"]].sum())
    first = bs.restrict(lp, (d["N"][:, None] < 2) | (bs.N_ARRIVAL - cfg.gap >= bs.ARRIVALS),
                        jnp)
    second = bs.restrict(lp, d["tau"][:, :1] + cfg.gap <= bs.ARRIVALS, jnp)
    first_t = d["tau"][:, 0].astype(int) - 1
    second_t = d["tau"][:, 1].astype(int) - 1
    numpyro.factor("times", (d["present"][:, 0] * first[np.arange(n), first_t]).sum()
                   + (d["present"][:, 1] * second[np.arange(n), second_t]).sum())


def fit_nuts(d, cfg, warmup=WARMUP, samples=SAMPLES, chains=CHAINS, seed=0):
    """(draws, info): posterior draws of the global parameters and NUTS diagnostics."""
    import jax
    from jax import random
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS, init_to_median

    kernel = NUTS(partial(numpyro_model, d, cfg), target_accept_prob=0.9,
                  init_strategy=init_to_median, max_tree_depth=8)
    parallel = jax.local_device_count() >= chains
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel" if parallel else "sequential", progress_bar=False)
    mcmc.run(random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
    by_chain = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    stats = summary(by_chain, group_by_chain=True)
    glob = [k for k in by_chain if not k.startswith("log_")]
    events = [k for k in by_chain if k.startswith("log_")]

    def worst(keys, stat, fn):
        vals = [np.atleast_1d(stats[k][stat]) for k in keys]
        return float(fn(np.concatenate(vals))) if vals else np.nan

    extra = mcmc.get_extra_fields()
    info = {"r_hat max": worst(glob, "r_hat", np.nanmax),
            "ess min": worst(glob, "n_eff", np.nanmin),
            "r_hat max events": worst(events, "r_hat", np.nanmax),
            "ess min events": worst(events, "n_eff", np.nanmin),
            "divergences": int(np.asarray(extra["diverging"]).sum()),
            "leapfrog mean": float(np.asarray(extra["num_steps"]).mean())}
    draws = {k: v.reshape((-1,) + v.shape[2:]) for k, v in by_chain.items()
             if k in bs.GLOBALS}
    for k in ("sigma_E", "sigma_R", "sigma_D"):  # latent fits: marks at population values
        draws.setdefault(k, np.zeros(len(draws["mu_E"])))
    return draws, info


def posterior_summary(draws):
    m = {k: float(np.median(np.atleast_2d(v.T)[0])) for k, v in draws.items()}
    return {"ambient w0": m["w_a"], "rho_up": m["rho_up"], "rho_dn": m["rho_dn"],
            "rho_eta": m["rho_eta"], "sigma_eta": m["sigma_eta"],
            "Lambda base /84h": float(np.exp(m["w_N"])),
            "excess median km/s": float(np.exp(m["mu_E"])), "sigma_E": m["sigma_E"],
            "rise width h": float(np.exp(m["mu_R"])),
            "decay h": float(np.exp(m["mu_D"])), "sigma_D": m["sigma_D"]}


def ppc(draws, sim_inp, d, cfg, seed=1):
    """Posterior predictive checks on the training windows: arrivals in hours 1..72, the
    80% interval coverage and the speed distribution."""
    paths, arrived = bs.simulate_paths(draws, sim_inp, cfg, N_PATHS_FIT,
                                       np.random.default_rng(seed))
    observed = ((d["tau"] <= cs.HORIZON) * d["present"]).sum()
    lo, hi = np.quantile(paths, [0.1, 0.9], axis=1)
    ok = d["obs"] > 0
    q = [5, 25, 50, 75, 95]
    return {"ppc arrivals observed": float(observed),
            "ppc arrivals expected": float(arrived.mean(axis=1).sum()),
            "ppc coverage 80": float(np.mean(((d["y"] >= lo) & (d["y"] <= hi))[ok])),
            "ppc speed q obs": [float(v) for v in np.percentile(d["y"][ok], q)],
            "ppc speed q sim": [float(v) for v in np.percentile(paths, q)]}


# ---------------------------------------------------------------------- study
def arrival_skill(out, n_lab, m_lab):
    clean = m_lab.all(axis=1)
    any_obs = n_lab[clean].sum(axis=1) > 0
    auc = (roc_auc_score(any_obs, out["p_any"][clean])
           if 0 < any_obs.sum() < len(any_obs) else np.nan)
    return {"arrival AUC": float(auc), "arrivals observed": float(n_lab[clean].sum()),
            "arrivals expected": float(out["arrivals"][clean].sum())}


def score_fold(data, start, end, names, timer, pilot=False):
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
    n_val, m_val = run_labels(data, val)
    year = f"{start:%Y}"
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"], "fits": {}}

    model = bs.BayesStreamModel(ts.structural_config(data.cfg))
    model.set_statistics(data.values, pos_fit, x_fit, start_pos)
    y_fit, q_fit = model.training_targets(data.values, pos_fit, start_pos)
    model.table = bs.weight_table(np.where(q_fit > 0, y_fit, np.nan), data.cfg.blend)
    inp_val = model.prepare(data.values, pos_val, x_val)
    d, sim_inp, wininfo = training_windows(data, model, table, cols, fit6, start_pos, cutoff)
    log(f"    windows: {wininfo}")
    draws, sims, forecasts = {}, {}, {}

    def fitted(key):
        if key not in draws:
            t = time.perf_counter()
            draws[key], info = fit_nuts(d, FITS[key])
            info["fit s"] = time.perf_counter() - t
            info.update(wininfo, **posterior_summary(draws[key]))
            t = time.perf_counter()
            info.update(ppc(draws[key], sim_inp, d, FITS[key]))
            info["ppc s"] = time.perf_counter() - t
            out["fits"][key] = info
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE_DIR / f"{year}_{key}_draws.npz", **draws[key])
            log(f"    fit {key}: {info['fit s']:.0f} s, R-hat max {info['r_hat max']:.3f}, "
                f"ESS min {info['ess min']:.0f}, divergences {info['divergences']}")
        return draws[key]

    def simulated(key, slow):
        if (key, slow) not in sims:
            sims[key, slow] = bs.summarize_paths(fitted(key), inp_val, FITS[key], model.table,
                                                 obs=obs, n_paths=N_PATHS, slow_only=slow)
        return sims[key, slow]

    for name in names:
        t = time.perf_counter()
        extra = {}
        if name == "reference":
            lin = LinearModel(data.cfg).fit(data.values, pos_fit, x_fit, end=start_pos)
            f = lin.forecast(data.values, pos_val, x_val)
        elif name == "linear + bayes":
            fit_inp = model.prepare(data.values, pos_fit, x_fit)
            s_fit = ts.structural_columns(bs.summarize_paths(
                fitted("observed"), fit_inp, FITS["observed"], n_paths=N_PATHS_FIT)["mean"])
            s_val = ts.structural_columns(simulated("observed", False)["mean"])
            cfg2 = replace(data.cfg, features=(*data.cfg.features, f"{ts.SIGNAL}_lead_0h"))
            c2 = list(cfg2.input_features)
            lin = LinearModel(cfg2).fit(data.values, pos_fit, table.loc[fit6].assign(**s_fit)[
                c2].to_numpy(dtype=float), end=start_pos)
            f = lin.forecast(data.values, pos_val,
                             table.loc[val].assign(**s_val)[c2].to_numpy(dtype=float))
        else:
            key, decision, slow = VARIANTS[name]
            sim = simulated(key, slow)
            f = sim[decision]
            extra = {"crps": float(np.nanmean(sim["crps"])),
                     "coverage 80": float(np.nanmean(sim["covered"]))}
            if not slow:
                extra.update(arrival_skill(sim, n_val, m_val))
        forecasts[name] = f
        row = dg.regime_scores(f, obs, pers, regime)
        row["residual slow wind"] = float(np.nanmean((obs - f)[regime == "slow wind"]))
        row["residual all"] = float(np.nanmean(obs - f))
        row.update(extra)
        out[name] = row
        timer[name] = timer.get(name, 0.0) + time.perf_counter() - t
        log(f"    {name}: S_final {row['all']:.2f} ({time.perf_counter() - t:.0f} s)")
    if ("observed", False) in sims and "reference" in forecasts:
        sim = sims["observed", False]
        np.savez_compressed(CACHE_DIR / f"{year}_parts.npz", t0s=val.asi8, obs=obs, pers=pers,
                            regime=regime, f_ref=forecasts["reference"], n_lab=n_val,
                            m_lab=m_val, **{k: sim[k] for k in (
                                "mean", "official", "q10", "q50", "q90", "p_any",
                                "arrivals")})
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
            new = score_fold(data, start, end, todo, timer, pilot)
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
            "(later folds have more fit windows)")
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
            **{k: float(np.mean([rows[y][name].get(k, np.nan) for y in years]))
               for k in ("crps", "coverage 80", "arrival AUC")},
            "keep": bool(gain < -KEEP_GAIN and won >= KEEP_FOLDS),
        }
        long += [{"fold": y, "variant": name, "persistence": rows[y]["persistence"],
                  **rows[y][name]} for y in years]
    for y in years:
        for key, info in rows[y].get("fits", {}).items():
            params.append({"fold": y, "fit": key,
                           **{k: (" ".join(f"{v:.0f}" for v in val) if isinstance(val, list)
                                  else val) for k, val in info.items()}})
    table = pd.DataFrame(out).T.sort_values("vs reference")
    if not pilot:
        pd.DataFrame(long).to_csv(EXPERIMENTS / "bayes_stream.csv", index=False,
                                  float_format="%.6g")
        pd.DataFrame(params).to_csv(EXPERIMENTS / "bayes_stream_params.csv", index=False,
                                    float_format="%.6g")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 40, "display.max_colwidth", 60):
        log(f"\nfolds {', '.join(years)}:\n{table.to_string()}")
        if params:
            log("\nfits:\n" + pd.DataFrame(params).T.to_string())
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--variants", nargs="*", default=list(ALL))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    import numpyro

    numpyro.set_host_device_count(CHAINS)  # chains in parallel (before jax starts)
    if args.summary:
        rows = {p.stem: json.loads(p.read_text("utf-8")) for p in sorted(CACHE_DIR.glob("*.json"))}
        summarize(rows)
        return
    run(args.folds, args.pilot, args.max_seconds, args.variants)


if __name__ == "__main__":
    sys.exit(main())
