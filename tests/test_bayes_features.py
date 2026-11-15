"""training.bayes_features: per-run features from a Bayes stream posterior."""

import numpy as np

from models import bayes_stream as bs
from models.linear_model import lead_lags
from training import bayes_features as bf

P = 6


def inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, bs.HORIZON, P)) * 0.5
    x[:, :, 0] = 1.0
    amb = np.column_stack([np.ones(n), rng.normal(size=(n, 2)) * 0.5])
    return {"x": x, "amb": amb, "last": 400.0 + 80.0 * rng.normal(size=n)}


def draws(n_draws, seed=1):
    rng = np.random.default_rng(seed)
    d = {"w_a": rng.normal(size=(n_draws, 3)) * 0.5, "rho_up": rng.uniform(0.9, 0.99, n_draws),
         "rho_dn": rng.uniform(0.9, 0.99, n_draws), "rho_eta": np.full(n_draws, 0.9),
         "sigma_eta": np.full(n_draws, 11.0),
         "w_N": np.column_stack([np.log(0.3) + 0.1 * rng.normal(size=n_draws),
                                 0.2 * rng.normal(size=(n_draws, P - 1))]),
         "w_tau": np.column_stack([np.full(n_draws, -3.7),
                                   0.3 * rng.normal(size=(n_draws, P - 1))]),
         "mu_E": np.log(150.0) + 0.1 * rng.normal(size=n_draws),
         "w_E": 0.1 * rng.normal(size=(n_draws, P - 1)), "sigma_E": np.full(n_draws, 0.3),
         "mu_R": np.log(2.0) + 0.1 * rng.normal(size=n_draws), "sigma_R": np.zeros(n_draws),
         "mu_D": np.log(100.0) + 0.1 * rng.normal(size=n_draws),
         "w_D": 0.1 * rng.normal(size=(n_draws, P - 1)), "sigma_D": np.full(n_draws, 1.4)}
    return d


def test_features_match_a_direct_calculation():
    inp, dr = inputs(3), draws(4)
    cfg = bs.Config(rule="add", event_rise=False)
    f = bf.features(dr, inp, cfg, n_draws=4)
    x84 = bs.extend(inp["x"])
    for i in range(3):
        acc = {k: np.zeros(bs.HORIZON) for k in ("ambient", "slow", "parr", "stream")}
        for j in range(4):
            g = {k: v[j] for k, v in dr.items()}
            a = 250 + 250 / (1 + np.exp(-inp["amb"][i] @ g["w_a"]))
            e0 = inp["last"][i] - a
            rho = g["rho_up"] if e0 > 0 else g["rho_dn"]
            lam = np.exp(inp["x"][i].mean(axis=0) @ g["w_N"])
            pn = lam ** np.arange(cfg.nmax + 1) / np.cumprod(np.r_[1, np.arange(1, cfg.nmax + 1)])
            p_any = 1 - pn[0] / pn.sum()
            v = 1 / (1 + np.exp(-(x84[i] @ g["w_tau"])))
            pi = v * np.cumprod(np.r_[1.0, 1 - v[:-1]])
            pi /= pi.sum()
            stream = np.zeros(bs.HORIZON)
            for u in range(bs.N_ARRIVAL):
                xc = x84[i, u, 1:]
                E, D, R = (np.exp(g["mu_E"] + xc @ g["w_E"]), np.exp(g["mu_D"] + xc @ g["w_D"]),
                           np.exp(g["mu_R"]))
                dk = bs.LEADS - (u + 1)
                prof = E / (1 + np.exp(-dk / R)) * np.exp(-np.maximum(0, dk - 2 * R) / D)
                stream += p_any * pi[u] * prof
            acc["ambient"] += a / 4
            acc["slow"] += (a + rho ** bs.LEADS * e0) / 4
            acc["parr"] += p_any * np.cumsum(pi)[:bs.HORIZON] / 4
            acc["stream"] += stream / 4
        for k, want in acc.items():
            np.testing.assert_allclose(f[k][i], want, rtol=1e-9, err_msg=k)
    np.testing.assert_allclose(f["mean"], f["slow"] + f["stream"])


def test_arrival_probability_rises_with_lead_and_stays_below_p_any():
    f = bf.features(draws(8), inputs(20, seed=2), bs.Config(rule="add", event_rise=False))
    assert (np.diff(f["parr"], axis=1) >= -1e-12).all()
    assert ((f["parr"] >= 0) & (f["parr"] <= 1)).all()
    assert (f["stream"] >= 0).all()


def test_lead_columns_follow_the_lead_aligned_grid():
    f = np.arange(2 * bs.HORIZON, dtype=float).reshape(2, bs.HORIZON)
    cols = bf.lead_columns(bf.signal_name("slow"), f)
    lags = np.unique(lead_lags(0, bf.signal_name("slow")))
    assert len(cols) == len(lags) and lags.min() < 0  # signed lags, as structural_*
    np.testing.assert_array_equal(cols["structural_bayes_slow_at_0h"], f[:, 0])
    np.testing.assert_array_equal(cols["structural_bayes_slow_at_-24h"], f[:, 23])
