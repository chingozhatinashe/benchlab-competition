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
MAP_STEPS = 4000
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
ARRIVAL_SITES = ("w_N", "w_tau")


def numpyro_model(d, cfg, block="all"):
    """block: "all", or with observed arrivals one of the posterior's two independent factors,
    "arrivals" (w_N, w_tau: the count and time terms) or "speed" (the rest: the speed term)."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    p = d["x"].shape[-1]
    n = len(d["y"])
    if block != "all" and cfg.latent:
        raise ValueError("the latent fit does not split: arrivals enter the speed term")

    def vec(name, loc0, scale0, scale):
        loc = jnp.zeros(p).at[0].set(loc0)
        sd = jnp.full(p, scale).at[0].set(scale0)
        return numpyro.sample(name, dist.Normal(loc, sd))

    if block == "arrivals":
        w_N = vec("w_N", np.log(0.25), 1.0, 0.5)
        w_tau = vec("w_tau", -3.7, 1.5, 0.5)
        arrival_factors(d, cfg, w_N, w_tau, jnp, numpyro)
        return

    w_a = numpyro.sample("w_a", dist.Normal(jnp.zeros(3), 1.5))
    rho_up = numpyro.sample("rho_up", dist.Beta(19.0, 1.0))
    rho_dn = numpyro.sample("rho_dn", dist.Beta(19.0, 1.0))
    rho_eta = numpyro.sample("rho_eta", dist.Beta(8.0, 2.0))
    sigma_eta = numpyro.sample("sigma_eta", dist.HalfNormal(50.0))
    if block == "all":
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
    xc = d["x"][:, :, 1:]

    if cfg.latent:  # an arrival in each LATENT_STEP-hour bin, at its middle hour
        clp = bs.count_logp(d["xbar"] @ w_N, cfg.nmax, jnp)
        lp = bs.stick_logp(d["x"] @ w_tau, jnp)
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
    if block == "all":
        arrival_factors(d, cfg, w_N, w_tau, jnp, numpyro)


def arrival_factors(d, cfg, w_N, w_tau, jnp, numpyro):
    """The observed arrivals' count and time terms (they involve w_N and w_tau only)."""
    n = len(d["y"])
    clp = bs.count_logp(d["xbar"] @ w_N, cfg.nmax, jnp)
    lp = bs.stick_logp(d["x"] @ w_tau, jnp)
    numpyro.factor("count", clp[np.arange(n), d["N"]].sum())
    first = bs.restrict(lp, (d["N"][:, None] < 2) | (bs.N_ARRIVAL - cfg.gap >= bs.ARRIVALS),
                        jnp)
    second = bs.restrict(lp, d["tau"][:, :1] + cfg.gap <= bs.ARRIVALS, jnp)
    first_t = d["tau"][:, 0].astype(int) - 1
    second_t = d["tau"][:, 1].astype(int) - 1
    numpyro.factor("times", (d["present"][:, 0] * first[np.arange(n), first_t]).sum()
                   + (d["present"][:, 1] * second[np.arange(n), second_t]).sum())


def global_sites(d, cfg, block="all"):
    """Names of the model's global (non-event) sample sites."""
    from jax import random
    from numpyro import handlers

    seeded = handlers.seed(partial(numpyro_model, d, cfg, block=block), random.PRNGKey(0))
    tr = handlers.trace(seeded).get_trace()
    return tuple(k for k, s in tr.items() if s["type"] == "sample" and not s["is_observed"]
                 and not k.startswith("log_"))


MARKS = ("log_E", "log_R", "log_D")


def map_point(d, cfg, steps=MAP_STEPS, seed=0, block="all"):
    """A starting point for the centred NUTS fit: the posterior mode by SVI with a point-mass
    guide (AutoDelta, Adam 0.01), found in the non-centred form of the event marks and mapped
    back to centred values.

    The centred joint density grows without bound as sigma_E, sigma_R, sigma_D -> 0 with every
    mark at its population mean, so a mode found directly in the centred form collapses the
    spreads (smoke run 2026-11-15: sigma_E 0.003) and starts NUTS in the funnel's neck. In the
    non-centred form the marks' standard-normal offsets carry the prior, and the mode keeps
    the spreads (experiment 1: sigma_E 1.46)."""
    import numpyro
    from jax import random
    from numpyro import handlers
    from numpyro.infer import SVI, Trace_ELBO, init_to_median
    from numpyro.infer.autoguide import AutoDelta
    from numpyro.infer.reparam import LocScaleReparam

    model = handlers.reparam(partial(numpyro_model, d, cfg, block=block),
                             config={k: LocScaleReparam(centered=0) for k in MARKS})
    guide = AutoDelta(model, init_loc_fn=init_to_median)
    svi = SVI(model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    res = svi.run(random.PRNGKey(seed), steps, progress_bar=False)
    point = guide.median(res.params)
    tr = handlers.trace(handlers.substitute(handlers.seed(model, random.PRNGKey(seed)),
                                            data=point)).get_trace()
    # both forms of the marks: centred, and the offsets for a fit with fit_nuts(noncentred=)
    init = {k: s["value"] for k, s in tr.items() if s["type"] == "sample"
            and not s["is_observed"]}
    init.update({k: tr[k]["value"] for k in MARKS if k in tr})
    return init, float(np.asarray(res.losses)[-1])


def fit_nuts(d, cfg, warmup=WARMUP, samples=SAMPLES, chains=CHAINS, seed=0, init="median",
             dense="none", sequential=False, max_tree_depth=8, block="all", noncentred=(),
             mass_init="identity"):
    """(draws, info): posterior draws of the global parameters and NUTS diagnostics.

    init: "median" (prior medians) or "map" (start every chain at map_point). dense: the mass
    matrix, "none" (diagonal), "globals" (dense block on the global sites, diagonal on the
    event marks; memory grows with the globals only) or "all". sequential: run the chains one
    after another (less memory than in parallel). noncentred: mark sites (of MARKS) that NUTS
    samples as standard-normal offsets (LocScaleReparam; the same model). mass_init: the
    adaptation's starting inverse mass matrix, "identity" or "hessian" (1 / |diagonal| of
    the potential's Hessian at the MAP start; needs init="map")."""
    import jax
    from jax import random
    from numpyro import handlers
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value
    from numpyro.infer.reparam import LocScaleReparam

    info = {}
    if init == "map":
        t = time.perf_counter()
        point, loss = map_point(d, cfg, seed=seed, block=block)
        info.update({"map s": time.perf_counter() - t, "map loss": loss,
                     **{f"map {k}": float(point[k]) for k in ("sigma_E", "sigma_R", "sigma_D")
                        if k in point}})
        strategy = init_to_value(values=point)
    else:
        strategy = init_to_median
    dense_sites = global_sites(d, cfg, block)
    dense_mass = {"none": False, "all": True, "globals": [dense_sites]}[dense]
    inverse_mass = None
    if mass_init == "hessian":  # the adaptation starts at the MAP's curvature scales
        if init != "map" or dense == "all":
            raise ValueError("mass_init='hessian' needs init='map' and dense 'none'/'globals'")
        t = time.perf_counter()
        H, sites = potential_hessian(d, cfg, point, block, noncentred, seed)
        inverse_mass = hessian_inverse_mass(H, sites, dense_sites if dense == "globals" else ())
        info["mass init s"] = time.perf_counter() - t
    model = partial(numpyro_model, d, cfg, block=block)
    if noncentred:
        model = handlers.reparam(model, config={k: LocScaleReparam(centered=0)
                                                for k in noncentred})
    kernel = NUTS(model, target_accept_prob=0.9,
                  init_strategy=strategy, max_tree_depth=max_tree_depth, dense_mass=dense_mass,
                  inverse_mass_matrix=inverse_mass)
    parallel = not sequential and jax.local_device_count() >= chains
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
    steps = np.asarray(extra["num_steps"])
    info.update({"r_hat max": worst(glob, "r_hat", np.nanmax),
                 "ess min": worst(glob, "n_eff", np.nanmin),
                 "r_hat max events": worst(events, "r_hat", np.nanmax),
                 "ess min events": worst(events, "n_eff", np.nanmin),
                 "divergences": int(np.asarray(extra["diverging"]).sum()),
                 "leapfrog mean": float(steps.mean()),
                 "tree depth hit": float(np.mean(steps >= 2 ** max_tree_depth - 1)),
                 "worst r_hat": {k: float(np.nanmax(np.atleast_1d(stats[k]["r_hat"])))
                                 for k in glob},
                 "chain medians": {k: [float(np.median(c)) for c in
                                       by_chain[k].reshape(chains, samples, -1)[..., 0]]
                                   for k in glob}})
    draws = {k: v.reshape((-1,) + v.shape[2:]) for k, v in by_chain.items()
             if k in bs.GLOBALS}
    if cfg.latent:  # marks at population values
        for k in ("sigma_E", "sigma_R", "sigma_D"):
            draws.setdefault(k, np.zeros(len(draws["mu_E"])))
    return draws, info


def potential_hessian(d, cfg, point, block="all", noncentred=(), seed=0):
    """(H, sites): the Hessian of NUTS's potential (minus the log density in unconstrained
    space) at `point`, built a column at a time by Hessian-vector products (memory), and the
    site name of each row, in the order jax flattens the parameters (sorted keys)."""
    import jax
    from jax import random
    from jax.flatten_util import ravel_pytree
    from numpyro import handlers
    from numpyro.infer import init_to_value
    from numpyro.infer.reparam import LocScaleReparam
    from numpyro.infer.util import initialize_model

    model = partial(numpyro_model, d, cfg, block=block)
    if noncentred:
        model = handlers.reparam(model, config={k: LocScaleReparam(centered=0)
                                                for k in noncentred})
    init = initialize_model(random.PRNGKey(seed), model,
                            init_strategy=init_to_value(values=point), dynamic_args=False)
    z = init.param_info.z
    flat, unravel = ravel_pytree(z)
    names = list(z)
    index, _ = ravel_pytree({k: np.full(np.shape(v), float(i)) for i, (k, v) in
                             enumerate(z.items())})  # labels flattened like the parameters
    sites = np.array(names)[np.asarray(index).astype(int)]
    grad = jax.grad(lambda f: init.potential_fn(unravel(f)))
    hvp = jax.jit(lambda v: jax.jvp(grad, (flat,), (v,))[1])
    eye = np.eye(len(flat))
    H = np.stack([np.asarray(hvp(eye[i])) for i in range(len(flat))], axis=1)
    return (H + H.T) / 2, sites


def hessian_inverse_mass(H, sites, dense_sites=()):
    """NUTS's initial inverse mass matrix from the Hessian's diagonal: 1 / |H_ii| (clipped to
    [1e-6, 100]), as numpyro's block structure ({dense block: diagonal matrix, the remaining
    sites: vector}) when `dense_sites`, else as a vector in jax's flattening order."""
    inv = np.clip(1 / np.maximum(np.abs(np.diag(H)), 1e-12), 1e-6, 100.0)
    if not dense_sites:
        return inv
    rest = tuple(sorted(set(sites) - set(dense_sites)))

    def part(keys):  # a block's entries in the order numpyro ravels a tuple of sites
        return np.concatenate([inv[sites == k] for k in keys])
    out = {tuple(dense_sites): np.diag(part(dense_sites))}
    if rest:
        out[rest] = part(rest)
    return out


def curvature(d, cfg, block="speed", noncentred=(), seed=0, top=6):
    """The Hessian of NUTS's potential at the MAP start (map_point): its spectrum, raw and
    scaled by its diagonal (what an adapted diagonal mass matrix leaves: correlations and
    sign, not scale), and which sites make up the extreme eigenvectors (share of the squared
    loadings per site)."""
    point, _ = map_point(d, cfg, seed=seed, block=block)
    t = time.perf_counter()
    H, sites = potential_hessian(d, cfg, point, block, noncentred, seed)
    z = [str(k) for k in dict.fromkeys(sites)]
    out = {"dims": int(len(H)), "hessian s": time.perf_counter() - t}
    diag = np.diag(H)
    out["diagonal by site"] = {k: [float(diag[sites == k].min()), float(diag[sites == k].max())]
                               for k in z}
    scale = 1 / np.sqrt(np.abs(diag))
    for name, M in (("raw", H), ("scaled", H * scale[:, None] * scale[None, :])):
        w, V = np.linalg.eigh(M)
        pos = w[w > 0]
        res = {"eig max": float(w[-1]), "eig min positive": float(pos.min()) if len(pos) else
               np.nan, "negative": int((w < 0).sum()), "eig min": float(w[0]),
               "condition": float(w[-1] / pos.min()) if len(pos) else np.nan}

        def loads(i, V=V):
            share = {k: float((V[sites == k, i] ** 2).sum()) for k in z}
            return {k: round(v, 3) for k, v in sorted(share.items(), key=lambda kv: -kv[1])
                    if v >= 0.02}
        res["stiffest"] = [{"eig": float(w[-1 - j]), "sites": loads(len(w) - 1 - j)}
                           for j in range(top)]
        res["flattest"] = [{"eig": float(pos[j]), "sites": loads(int(np.nonzero(w > 0)[0][j]))}
                           for j in range(min(top, len(pos)))]
        res["most negative"] = [{"eig": float(w[j]), "sites": loads(j)}
                                for j in range(min(top, res["negative"]))]
        out[name] = res
    return out


def fit_split(d, cfg, **kw):
    """fit_nuts on the observed-arrival posterior's two independent factors, the arrival block
    (w_N, w_tau) and the speed block (the rest), with the draws joined by index (exact: the
    blocks are independent a posteriori). (draws, {"arrivals": info, "speed": info})."""
    draws, infos = {}, {}
    for block in ("arrivals", "speed"):
        part, infos[block] = fit_nuts(d, cfg, block=block, **kw)
        draws.update(part)
    return draws, infos


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


def fold_fit(data, start):
    """The fold's fit runs, the model with its statistics and the training windows."""
    f = {"start_pos": data.series.index.get_loc(start),
         "fit6": data.runs6[data.runs6 < start], "cols": list(data.cfg.input_features)}
    f["table"] = table = data.table.copy()
    table[data.pfss] = table[data.pfss].fillna(table.loc[f["fit6"], data.pfss].median())
    f["pos_fit"] = data.series.index.get_indexer(f["fit6"])
    f["x_fit"] = table.loc[f["fit6"], f["cols"]].to_numpy(dtype=float)
    f["model"] = model = bs.BayesStreamModel(ts.structural_config(data.cfg))
    model.set_statistics(data.values, f["pos_fit"], f["x_fit"], f["start_pos"])
    y_fit, q_fit = model.training_targets(data.values, f["pos_fit"], f["start_pos"])
    model.table = bs.weight_table(np.where(q_fit > 0, y_fit, np.nan), data.cfg.blend)
    f["d"], f["sim_inp"], f["wininfo"] = training_windows(
        data, model, table, f["cols"], f["fit6"], f["start_pos"],
        start - pd.Timedelta(hours=LABEL_GUARD))
    return f


def score_fold(data, start, end, names, timer, pilot=False):
    ff = fold_fit(data, start)
    start_pos, fit6, cols, table, model = (ff[k] for k in ("start_pos", "fit6", "cols",
                                                           "table", "model"))
    pos_fit, x_fit, d, sim_inp, wininfo = (ff[k] for k in ("pos_fit", "x_fit", "d", "sim_inp",
                                                           "wininfo"))
    val = data.runs6[(data.runs6 >= start) & (data.runs6 < end)]
    obs, pers = cs.verifying(data.truth, val), cs.persistence(data.truth, val)
    regime = data.run_regimes(val)
    pos_val = data.series.index.get_indexer(val)
    x_val = table.loc[val, cols].to_numpy(dtype=float)
    n_val, m_val = run_labels(data, val)
    year = f"{start:%Y}"
    out = {"runs": len(val), "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1),
                                                     obs, pers)["s_final"], "fits": {}}
    inp_val = model.prepare(data.values, pos_val, x_val)
    log(f"    windows: {wininfo}")
    draws, sims, forecasts = {}, {}, {}

    def fitted(key):
        if key not in draws:
            t = time.perf_counter()
            draws[key], info = fit_nuts(d, FITS[key])
            info = {k: v for k, v in info.items() if not isinstance(v, dict)}
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


def sampler_test(year, fit, label, block="all", curv=False, **kw):
    """NUTS on a fold's training windows, without scoring: diagnostics, time and peak memory
    per block ("all", "arrivals", "speed", or "split": arrivals then speed), written to
    CACHE_DIR / sampler_<label>.json."""
    t = time.perf_counter()
    data = dg.Data()
    start = next(s for s, _ in YEARLY if s.year == year)
    ff = fold_fit(data, start)
    log(f"data and windows ready in {time.perf_counter() - t:.0f} s: {ff['wininfo']}; "
        f"peak memory {tx.peak_mb():.0f} MB; settings {kw}")
    if curv:
        res = curvature(ff["d"], FITS[fit], block=block,
                        noncentred=kw.get("noncentred", ()))
        (CACHE_DIR / f"sampler_{label}.json").write_text(json.dumps(res, indent=1), "utf-8")
        log(f"{res['dims']} dims, Hessian in {res['hessian s']:.0f} s")
        log("diagonal (min, max) by site: " + ", ".join(
            f"{k} {lo:.3g}..{hi:.3g}" for k, (lo, hi) in res["diagonal by site"].items()))
        for name in ("raw", "scaled"):
            r = res[name]
            log(f"{name}: eig max {r['eig max']:.4g}, min positive {r['eig min positive']:.4g}, "
                f"condition {r['condition']:.3g}, negative {r['negative']} "
                f"(most negative {r['eig min']:.4g})")
            for part in ("stiffest", "flattest", "most negative"):
                for e in r[part]:
                    log(f"  {part:13s} {e['eig']:11.4g}  {e['sites']}")
        return res
    draws, out = {}, {"year": year, "fit": fit, "settings": kw, **ff["wininfo"]}
    for b in (("arrivals", "speed") if block == "split" else (block,)):
        t = time.perf_counter()
        part, info = fit_nuts(ff["d"], FITS[fit], block=b, **kw)
        draws.update(part)
        info.update({"fit s": time.perf_counter() - t, "peak MB": tx.peak_mb()})
        out[b] = info
        log(f"block {b}: fit {info['fit s']:.0f} s, peak memory {info['peak MB']:.0f} MB")
        for k in ("map s", "map sigma_E", "map sigma_R", "map sigma_D", "r_hat max", "ess min",
                  "r_hat max events", "ess min events", "divergences", "leapfrog mean",
                  "tree depth hit"):
            if k in info and np.isfinite(info[k]):
                log(f"  {k}: {info[k]:.4g}")
        log("  worst R-hat by site (chain medians of the first element):")
        for k, r in sorted(info["worst r_hat"].items(), key=lambda kv: -kv[1]):
            log(f"    {k:10s} {r:7.3f}  "
                + "  ".join(f"{m:9.4g}" for m in info["chain medians"][k]))
    if set(bs.GLOBALS) <= set(draws):
        out.update(posterior_summary(draws))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"sampler_{label}.json").write_text(json.dumps(out, indent=1), "utf-8")
    np.savez_compressed(CACHE_DIR / f"sampler_{label}_draws.npz", **draws)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--variants", nargs="*", default=list(ALL))
    parser.add_argument("--summary", action="store_true")
    sm = parser.add_argument_group("sampler test (--sampler: one fit, no scoring)")
    sm.add_argument("--sampler", metavar="LABEL")
    sm.add_argument("--year", type=int, default=2024)
    sm.add_argument("--fit", choices=list(FITS), default="observed")
    sm.add_argument("--block", choices=("all", "arrivals", "speed", "split"), default="all")
    sm.add_argument("--noncentred", nargs="*", choices=MARKS, default=[],
                    help="mark sites sampled non-centred")
    sm.add_argument("--mass-init", choices=("identity", "hessian"), default="identity")
    sm.add_argument("--curvature", action="store_true",
                    help="the potential's Hessian at the MAP start instead of a NUTS fit")
    sm.add_argument("--warmup", type=int, default=WARMUP)
    sm.add_argument("--samples", type=int, default=SAMPLES)
    sm.add_argument("--chains", type=int, default=CHAINS)
    sm.add_argument("--init", choices=("median", "map"), default="map")
    sm.add_argument("--dense", choices=("none", "globals", "all"), default="globals")
    sm.add_argument("--parallel", action="store_true", help="chains in parallel (more memory)")
    sm.add_argument("--x64", action="store_true")
    args = parser.parse_args(argv)
    import numpyro

    numpyro.set_host_device_count(CHAINS)  # chains in parallel (before jax starts)
    if args.sampler:
        if args.x64:
            numpyro.enable_x64()
        sampler_test(args.year, args.fit, args.sampler, block=args.block,
                     curv=args.curvature, warmup=args.warmup,
                     samples=args.samples, chains=args.chains, init=args.init,
                     dense=args.dense, sequential=not args.parallel,
                     noncentred=tuple(args.noncentred), mass_init=args.mass_init)
        return
    if args.summary:
        rows = {p.stem: json.loads(p.read_text("utf-8")) for p in sorted(CACHE_DIR.glob("*.json"))}
        summarize(rows)
        return
    run(args.folds, args.pilot, args.max_seconds, args.variants)


if __name__ == "__main__":
    sys.exit(main())
