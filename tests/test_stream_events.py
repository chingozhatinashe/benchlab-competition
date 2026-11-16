"""models.stream_events: profiles, additivity, numpy/torch parity, chunking, recovery, assets."""

import numpy as np
import pytest

from models import stream_events as se
from models.linear_model import Config as LinearConfig


def synthetic_inputs(cfg, n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, se.st.HORIZON, cfg.n_inputs)) * 0.5
    x[:, :, 0] = 1.0
    x[:, :, 1] = x[:, :1, 1]
    amb = np.column_stack([np.ones(n), rng.normal(size=(n, 2)) * 0.5])
    return {"x": x, "amb": amb, "last": 400.0 + 100.0 * x[:, 0, 1], "leads": se.LEADS}


def random_params(cfg, seed=1, log_rate=-3.5):
    rng = np.random.default_rng(seed)
    p = se.initial_params(cfg, log_rate)
    return {k: v + 0.2 * rng.normal(size=v.shape) for k, v in p.items()}


def test_piecewise_profile():
    values = np.array([0.0, 1.0, 0.5, 0.0])
    phase = np.array([-0.1, 0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 1.0, 1.2])
    out = se.piecewise(values, phase, se.NUMPY)
    np.testing.assert_allclose(out, [0, 0, 0.5, 1.0, 0.75, 0.5, 0.0, 0.0])


def test_events_add_and_vanish_outside_their_duration():
    cfg = se.Config()
    inp = synthetic_inputs(cfg, n=5)
    p = random_params(cfg)
    phi, _ = se.profiles(p, se.NUMPY)
    _, lam, dur, amp = se.event_terms(p, inp["x"], cfg, se.NUMPY)
    k = se.LEADS
    single = []
    for u in (20, 40):  # each event alone, by hand
        phase = (k - u + cfg.lead_in) / (dur[:, u - 1, None] + cfg.lead_in)
        single.append(lam[:, u - 1, None] * amp[:, u - 1, None]
                      * se.piecewise(phi, phase, se.NUMPY))
    for s, u in zip(single, (20, 40), strict=True):
        before = k < u - cfg.lead_in
        after = k > u + dur[:, u - 1, None]
        assert np.all(s[:, before] == 0) and np.all(s[after] == 0)
    # the model's stream term is the sum over every arrival hour: check with two hours only
    weights = np.zeros_like(lam)
    weights[:, [19, 39]] = 1.0
    gap = k[None, :, None] - k[None, None, :] + cfg.lead_in
    phase = gap / (dur[:, None, :] + cfg.lead_in)
    total = np.einsum("nku,nu->nk", se.piecewise(phi, phase, se.NUMPY), weights * lam * amp)
    np.testing.assert_allclose(total, single[0] + single[1])
    v, parts = se.forward(p, inp, cfg, parts=True)
    assert np.isfinite(v).all() and (parts["stream"] >= 0).all()


def test_numpy_torch_parity_and_chunking():
    torch = pytest.importorskip("torch")
    from training import stream_events as tse
    from training import structural as ts

    for cfg in (se.Config(), se.Config(single=True)):
        inp = synthetic_inputs(cfg, n=50)
        p = random_params(cfg)
        v_np = se.forward(p, inp, cfg)
        with torch.no_grad():
            v_t = se.forward({k: ts.to_torch(v) for k, v in p.items()},
                             {k: ts.to_torch(v) for k, v in inp.items()}, cfg, tse.torch_ops())
        np.testing.assert_allclose(ts.to_numpy(v_t), v_np, atol=1e-6)
    cfg = se.Config()
    inp = synthetic_inputs(cfg, n=90)
    y = se.forward(random_params(cfg, 2), inp, cfg)
    q = np.ones_like(y)
    n_lab = (np.random.default_rng(3).random(y.shape) < 0.02).astype(float)
    m_lab = np.ones_like(y)
    a = tse.fit_inputs(cfg, inp, y, q, n_lab, m_lab, max_iter=5, chunk=1000)
    b = tse.fit_inputs(cfg, inp, y, q, n_lab, m_lab, max_iter=5, chunk=25)
    np.testing.assert_allclose(a[1]["loss"], b[1]["loss"], rtol=1e-8)


def test_recovery_with_poisson_labels():
    pytest.importorskip("torch")
    from training import stream_events as tse

    cfg = se.Config(ridge=0.0, smooth=0.0)
    inp = synthetic_inputs(cfg, n=800, seed=4)
    true = random_params(cfg, seed=5)
    rng = np.random.default_rng(6)
    clean = se.forward(true, inp, cfg)
    y = clean + rng.normal(scale=2.0, size=clean.shape)
    _, lam, _, _ = se.event_terms(true, inp["x"], cfg, se.NUMPY)
    n_lab = rng.poisson(lam).astype(float)
    params, info = tse.fit_inputs(cfg, inp, y, np.ones_like(y), n_lab, np.ones_like(y),
                                  log_rates=(-4.0,), max_iter=300)
    fit = se.forward(params, inp, cfg)
    assert np.sqrt(np.mean((fit - clean) ** 2)) < 3.0
    # the labels pin the intensity's intercept and its slopes
    np.testing.assert_allclose(params["lam"], true["lam"], atol=0.25)


def test_asset_round_trip():
    pytest.importorskip("torch")
    from training import stream_events as tse

    rng = np.random.default_rng(7)
    n_hours = 24 * 300
    t = np.arange(n_hours)
    values = 420 + 120 * np.maximum(0, np.sin(2 * np.pi * t / (27 * 24))) ** 4 \
        + rng.normal(scale=15, size=n_hours)
    positions = np.arange(28 * 24 + 60, n_hours - 80, 12)
    cfg = tse.events_config(LinearConfig())
    model = se.StreamEventsModel(cfg)
    model.set_statistics(values, positions, None, n_hours - 80)
    inp = model.prepare(values, positions, None)
    y, q = model.training_targets(values, positions, n_hours - 80, inp["has_last"])
    n_lab = np.zeros_like(y)
    model.params, _ = tse.fit_inputs(cfg, inp, y, q, n_lab, np.ones_like(y), max_iter=20)
    f = model.forecast(values, positions)
    again = se.StreamEventsModel.from_dict(model.to_dict())
    assert again.config == cfg
    np.testing.assert_allclose(again.forecast(values, positions), f)
