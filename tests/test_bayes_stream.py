"""models.bayes_stream: stick-breaking, overlap rule, simulation, enumeration, recovery, assets."""

import numpy as np
import pytest

from models import bayes_stream as bs
from models.linear_model import Config as LinearConfig

P = 6  # inputs: a constant + 5


def synthetic_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, bs.HORIZON, P)) * 0.5
    x[:, :, 0] = 1.0
    x[:, :, 1] = x[:, :1, 1]  # a window-level input (the count uses the window mean)
    amb = np.column_stack([np.ones(n), rng.normal(size=(n, 2)) * 0.5])
    return {"x": x, "amb": amb, "last": 400.0 + 80.0 * rng.normal(size=n)}


LOG_RATE = float(np.log(0.4))


def true_draws(n_draws=1, log_rate=LOG_RATE):
    w = {"w_a": [0.0, 1.5, 0.3], "rho_up": 0.95, "rho_dn": 0.9, "rho_eta": 0.8,
         "sigma_eta": 12.0, "w_N": [log_rate, 0.6, 0, 0, 0, 0],
         "w_tau": [-3.5, 0, 1.0, 0, 0, 0], "mu_E": np.log(220.0),
         "w_E": [0.2, 0, 0, 0, 0], "sigma_E": 0.2, "mu_R": np.log(5.0), "sigma_R": 0.2,
         "mu_D": np.log(40.0), "w_D": [0, 0.2, 0, 0, 0], "sigma_D": 0.2}
    return {k: np.repeat(np.atleast_1d(np.asarray(v, dtype=float))[None], n_draws, 0).squeeze(
        axis=1 if np.ndim(v) == 0 else ()) for k, v in w.items()}


def test_stick_breaking_probabilities():
    rng = np.random.default_rng(0)
    logit = rng.normal(size=(3, bs.N_ARRIVAL)) - 3
    lp = bs.stick_logp(logit, np)
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1.0)
    v = 1 / (1 + np.exp(-logit))
    direct = v * np.cumprod(np.concatenate([np.ones((3, 1)), 1 - v[:, :-1]], 1), axis=1)
    np.testing.assert_allclose(np.exp(lp), direct / direct.sum(axis=1, keepdims=True))
    allowed = bs.ARRIVALS >= 40
    r = np.exp(bs.restrict(lp, allowed, np))
    assert np.all(r[:, ~allowed] == 0) and np.allclose(r.sum(axis=1), 1.0)


def test_ar1_likelihood_with_gaps_matches_the_marginal_gaussian():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(11)
    rho, sigma = 0.85, 10.0
    r = rng.normal(size=bs.HORIZON) * 20
    observed = rng.random(bs.HORIZON) > 0.3
    observed[[5, 6, 7, 8, 9]] = False
    k = np.arange(1, bs.HORIZON + 1)  # AR(1) from 0 at t0: cov(i, j) = s^2 rho^|i-j| sum
    lo = np.minimum(k[:, None], k[None, :])
    lag = np.abs(k[:, None] - k[None, :])
    cov = sigma ** 2 * rho ** lag * (1 - rho ** (2 * lo)) / (1 - rho ** 2)
    want = multivariate_normal(np.zeros(observed.sum()),
                               cov[np.ix_(observed, observed)]).logpdf(r[observed])
    gaps = bs.gap_structure(observed[None])
    got = bs.ar1_loglik(r[None], rho, sigma, np, gaps)[0]
    np.testing.assert_allclose(got, want, rtol=1e-9)
    full = bs.gap_structure(np.ones((1, bs.HORIZON), bool))
    np.testing.assert_allclose(bs.ar1_loglik(r[None], rho, sigma, np, full),
                               bs.ar1_loglik(r[None], rho, sigma, np))


def test_count_probabilities():
    lp = bs.count_logp(np.log(np.array([0.3, 2.0])), 2, np)
    lam = np.array([0.3, 2.0])[:, None]
    direct = np.hstack([np.ones_like(lam), lam, lam ** 2 / 2])
    np.testing.assert_allclose(np.exp(lp), direct / direct.sum(axis=1, keepdims=True))


def test_max_rule_dominates_components_and_additive_adds():
    rng = np.random.default_rng(1)
    slow = 380 + rng.normal(size=(4, bs.HORIZON)) * 10
    a = np.full(4, 370.0)
    excess = bs.profile(np.array([[10.0, 50.0]] * 4), np.full((4, 2), 250.0),
                        np.full((4, 2), 5.0), np.full((4, 2), 30.0), np)
    present = np.array([[1, 1], [1, 0], [0, 0], [0, 1]], dtype=float)
    v = bs.combine(slow, a, excess, present, bs.Config(), np)
    assert np.all(v >= slow - 1e-9)
    for i in range(4):
        for j in range(2):
            if present[i, j]:
                assert np.all(v[i] >= a[i] + excess[i, j] - 1e-9)
    np.testing.assert_allclose(v[2], slow[2])  # no stream: the slow wind exactly
    add = bs.combine(slow, a, excess, present, bs.Config(rule="add"), np)
    np.testing.assert_allclose(add, slow + (present[..., None] * excess).sum(1))


def test_no_arrivals_gives_the_slow_wind():
    inp = synthetic_inputs(20)
    draws = true_draws(3, log_rate=-60.0)
    cfg = bs.Config()
    v, arrived = bs.simulate_paths(draws, inp, cfg, 50, np.random.default_rng(5))
    s, _ = bs.simulate_paths(draws, inp, cfg, 50, np.random.default_rng(5), slow_only=True)
    assert arrived.sum() == 0
    np.testing.assert_allclose(v, s, atol=1e-6)
    v, arrived, taus, present = bs.simulate_paths(true_draws(3, np.log(3.0)), inp, cfg, 200,
                                                  np.random.default_rng(6), details=True)
    two = present[..., 1] > 0
    assert two.any() and np.all(taus[..., 1][two] - taus[..., 0][two] >= cfg.gap)
    assert np.all((v >= 200) & (v <= 1200))


def test_numpy_jax_parity():
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(2)
    tau = rng.uniform(1, 84, size=(5, 2))
    E, R, D = (rng.uniform(lo, hi, size=(5, 2)) for lo, hi in ((50, 300), (2, 10), (10, 80)))
    slow = 350 + rng.normal(size=(5, bs.HORIZON)) * 20
    a = np.full(5, 360.0)
    present = (rng.random((5, 2)) < 0.6).astype(float)
    for cfg in (bs.Config(), bs.Config(rule="add")):
        ex_np = bs.profile(tau, E, R, D, np)
        v_np = bs.combine(slow, a, ex_np, present, cfg, np)
        ex_j = bs.profile(jnp.asarray(tau), jnp.asarray(E), jnp.asarray(R), jnp.asarray(D), jnp)
        v_j = bs.combine(jnp.asarray(slow), jnp.asarray(a), ex_j, jnp.asarray(present), cfg, jnp)
        np.testing.assert_allclose(np.asarray(v_j), v_np, rtol=1e-5)


def test_latent_enumeration_matches_brute_force():
    pytest.importorskip("numpyro")
    import numpyro
    from numpyro import handlers

    from training import bayes_stream as tbs

    numpyro.enable_x64()
    cfg = bs.Config(nmax=1, latent=True)
    inp = synthetic_inputs(4, seed=3)
    y = 380 + np.random.default_rng(4).normal(size=(4, bs.HORIZON)) * 30
    d = tbs.window_data(inp["x"], inp["amb"], inp["last"], y, np.zeros((4, bs.N_ARRIVAL)))
    p = {k: v[0] for k, v in true_draws(1).items() if not k.startswith("sigma_")}
    p["sigma_eta"] = 12.0
    with handlers.trace() as tr, handlers.substitute(data=p):
        tbs.numpyro_model(d, cfg)
    got = float(tr["speed"]["fn"].log_factor)
    a = bs.ambient(p["w_a"], d["amb"], np)
    slow = bs.slow_mean(a, d["v0"], p["rho_up"], p["rho_dn"], np)
    clp = bs.count_logp(d["xbar"] @ p["w_N"], 1, np)
    lp = bs.stick_logp(d["x"] @ p["w_tau"], np)
    total = 0.0
    for i in range(4):
        terms = [clp[i, 0] + bs.ar1_loglik(y[i] - slow[i], p["rho_eta"], 12.0, np)]
        step = tbs.LATENT_STEP
        for k in range(bs.N_ARRIVAL // step):  # an arrival in each bin, at its middle hour
            u = k * step + step // 2
            xc = d["x"][i, u, 1:]
            E = np.exp(p["mu_E"] + xc @ p["w_E"])
            D = np.exp(p["mu_D"] + xc @ p["w_D"])
            ex = bs.profile(np.array([[u + 1.0]]), np.array([[E]]),
                            np.array([[np.exp(p["mu_R"])]]), np.array([[D]]), np)
            mean = bs.combine(slow[i:i + 1], a[i:i + 1], ex[:, :, :], np.ones((1, 1)), cfg, np)
            terms.append(clp[i, 1] + np.logaddexp.reduce(lp[i, k * step:(k + 1) * step])
                         + bs.ar1_loglik(y[i] - mean[0], p["rho_eta"], 12.0, np))
        total += np.logaddexp.reduce(terms)
    np.testing.assert_allclose(got, total, rtol=1e-6)


def test_split_blocks_add_up_to_the_joint_density():
    pytest.importorskip("numpyro")
    import numpyro
    from jax import random
    from numpyro import handlers
    from numpyro.infer.util import log_density

    from training import bayes_stream as tbs

    numpyro.enable_x64()
    cfg = bs.Config()
    inp = synthetic_inputs(6, seed=5)
    y = 380 + np.random.default_rng(6).normal(size=(6, bs.HORIZON)) * 30
    n_lab = np.zeros((6, bs.N_ARRIVAL))
    n_lab[1, 10] = n_lab[3, 5] = n_lab[3, 40] = 1
    d = tbs.window_data(inp["x"], inp["amb"], inp["last"], y, n_lab)
    for seed in range(3):
        tr = handlers.trace(handlers.seed(tbs.numpyro_model, random.PRNGKey(seed))).get_trace(
            d, cfg)
        p = {k: s["value"] for k, s in tr.items()
             if s["type"] == "sample" and not s["is_observed"]}
        lp = {b: float(log_density(tbs.numpyro_model, (d, cfg), {"block": b},
                                   {k: v for k, v in p.items() if (k in tbs.ARRIVAL_SITES)
                                    == (b == "arrivals") or b == "all"})[0])
              for b in ("all", "arrivals", "speed")}
        np.testing.assert_allclose(lp["arrivals"] + lp["speed"], lp["all"], rtol=1e-10)
    with pytest.raises(ValueError):
        tbs.numpyro_model(d, bs.Config(nmax=1, latent=True), block="speed")


def test_fixed_rise_has_no_event_rise_widths():
    pytest.importorskip("numpyro")
    from jax import random
    from numpyro import handlers

    from training import bayes_stream as tbs

    inp = synthetic_inputs(4, seed=9)
    y = 380 + np.random.default_rng(10).normal(size=(4, bs.HORIZON)) * 30
    n_lab = np.zeros((4, bs.N_ARRIVAL))
    n_lab[0, 20] = n_lab[2, 8] = 1
    d = tbs.window_data(inp["x"], inp["amb"], inp["last"], y, n_lab)
    sites = {cfg.event_rise: set(handlers.trace(handlers.seed(
        tbs.numpyro_model, random.PRNGKey(0))).get_trace(d, cfg))
        for cfg in (bs.Config(), bs.Config(event_rise=False))}
    assert sites[True] - sites[False] == {"log_R", "sigma_R"}
    assert sites[False] <= sites[True]


def test_potential_hessian_rows_carry_the_right_site_labels():
    pytest.importorskip("numpyro")
    import numpyro
    from jax import random
    from numpyro import handlers
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    from training import bayes_stream as tbs

    numpyro.enable_x64()
    cfg = bs.Config()
    inp = synthetic_inputs(5, seed=7)
    y = 380 + np.random.default_rng(8).normal(size=(5, bs.HORIZON)) * 30
    n_lab = np.zeros((5, bs.N_ARRIVAL))
    n_lab[2, 12] = 1
    d = tbs.window_data(inp["x"], inp["amb"], inp["last"], y, n_lab)
    tr = handlers.trace(handlers.seed(tbs.numpyro_model, random.PRNGKey(1))).get_trace(
        d, cfg, block="speed")
    point = {k: s["value"] for k, s in tr.items()
             if s["type"] == "sample" and not s["is_observed"]}
    H, sites = tbs.potential_hessian(d, cfg, point, block="speed")
    init = initialize_model(random.PRNGKey(0), lambda: tbs.numpyro_model(d, cfg, "speed"),
                            init_strategy=init_to_value(values=point), dynamic_args=False)
    z = {k: np.asarray(v, dtype=float) for k, v in init.param_info.z.items()}
    assert sorted(set(sites)) == sorted(z)
    for k in ("sigma_eta", "rho_up", "mu_D"):  # scalar sites: second differences
        def pe(dz, k=k):
            return float(init.potential_fn({**z, k: z[k] + dz}))
        h = 1e-4
        fd = (pe(h) - 2 * pe(0.0) + pe(-h)) / h ** 2
        np.testing.assert_allclose(H[sites == k, sites == k][0], fd, rtol=1e-3)
    inv = tbs.hessian_inverse_mass(H, sites, ("sigma_eta", "rho_up"))
    assert set(inv) == {("sigma_eta", "rho_up"), tuple(sorted(set(sites) - {"sigma_eta",
                                                                             "rho_up"}))}


def test_recovery_with_observed_arrivals():
    pytest.importorskip("numpyro")
    from training import bayes_stream as tbs

    cfg = bs.Config()
    inp = synthetic_inputs(500, seed=7)
    truth = true_draws(1)
    y, _, taus, present = bs.simulate_paths(truth, inp, cfg, 1, np.random.default_rng(8),
                                            details=True)
    n_lab = np.zeros((500, bs.N_ARRIVAL))
    for j in range(2):
        on = present[:, 0, j] > 0
        n_lab[np.flatnonzero(on), taus[on, 0, j].astype(int) - 1] = 1
    d = tbs.window_data(inp["x"], inp["amb"], inp["last"], y[:, 0], n_lab)
    draws, info = tbs.fit_nuts(d, cfg, warmup=300, samples=300, chains=1, seed=1)
    med = {k: np.median(v, axis=0) for k, v in draws.items()}
    assert info["divergences"] < 30
    np.testing.assert_allclose(med["w_N"][:2], truth["w_N"][0][:2], atol=0.3)
    np.testing.assert_allclose(med["w_tau"][2], truth["w_tau"][0][2], atol=0.4)
    np.testing.assert_allclose(med["sigma_eta"], 12.0, rtol=0.1)
    np.testing.assert_allclose(med["rho_eta"], 0.8, atol=0.05)
    np.testing.assert_allclose(np.exp(med["mu_E"]), 220.0, rtol=0.2)
    np.testing.assert_allclose(np.exp(med["mu_D"]), 40.0, rtol=0.35)


def test_asset_round_trip():
    rng = np.random.default_rng(9)
    n_hours = 24 * 300
    t = np.arange(n_hours)
    values = 420 + 120 * np.maximum(0, np.sin(2 * np.pi * t / (27 * 24))) ** 4 \
        + rng.normal(scale=15, size=n_hours)
    positions = np.arange(28 * 24 + 60, n_hours - 80, 12)
    from training import structural as ts

    model = bs.BayesStreamModel(ts.structural_config(LinearConfig()))
    model.set_statistics(values, positions, None, n_hours - 80)
    p = model.config.n_inputs
    draws = true_draws(4)
    for k in ("w_N", "w_tau"):  # the asset's model has p inputs: intercepts only
        draws[k] = np.hstack([draws[k][:, :1], np.zeros((4, p - 1))])
    draws["w_E"] = draws["w_D"] = np.zeros((4, p - 1))
    model.draws = draws
    y, q = model.training_targets(values, positions, n_hours - 80)
    model.table = bs.weight_table(np.where(q > 0, y, np.nan), 0.25)
    f = model.forecast(values, positions[:30], n_paths=50)
    g = model.forecast(values, positions[:30], decision="official", n_paths=50)
    assert f.shape == (30, bs.HORIZON) and np.isfinite(g).all()
    again = bs.BayesStreamModel.from_dict(model.to_dict())
    assert again.bayes == model.bayes and again.config == model.config
    np.testing.assert_allclose(again.forecast(values, positions[:30], n_paths=50), f)
