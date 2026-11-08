"""models.structural: forward equation, numpy/torch parity, parameter recovery, assets."""

import numpy as np
import pytest

from models import structural as st
from models.linear_model import Config as LinearConfig

RNG = np.random.default_rng(0)


def synthetic_inputs(cfg, n=400, seed=0):
    rng = np.random.default_rng(seed)
    p_in = cfg.n_inputs
    x = rng.normal(size=(n, st.HORIZON, p_in)) * 0.5
    x[:, :, 0] = 1.0
    x[:, :, 1] = x[:, :1, 1]  # the last speed is the same at every lead
    amb = np.column_stack([np.ones(n), rng.normal(size=(n, 2)) * 0.5])
    last = 400.0 + 100.0 * x[:, 0, 1]
    d = st.LEADS[:, None] - st.LEADS[None, :]
    return {"x": x, "amb": amb, "last": last, "leads": st.LEADS,
            "lag": np.maximum(d, 0.0), "ahead": (d >= 0).astype(float)}


def random_params(cfg, seed=1):
    rng = np.random.default_rng(seed)
    p = st.initial_params(cfg, hazard=-4.0)
    return {k: v + 0.3 * rng.normal(size=v.shape) for k, v in p.items()}


def test_forward_structure():
    cfg = st.Config()
    inp = synthetic_inputs(cfg, n=50)
    p = random_params(cfg)
    v, parts = st.forward(p, inp, cfg, parts=True)
    assert v.shape == (50, st.HORIZON) and np.isfinite(v).all()
    lo, hi = st.AMBIENT_RANGE
    assert ((parts["ambient"] > lo) & (parts["ambient"] < hi)).all()
    assert (parts["stream"] >= 0).all()  # arrivals only add speed
    # the current excess relaxes towards the ambient
    gap = np.abs(parts["current"] - parts["ambient"][:, None])
    assert (np.diff(gap, axis=1) <= 1e-9).all()
    # no arrivals: the stream term vanishes and the forecast is ambient + decay
    off = st.Config(arrivals=False)
    v_off, parts_off = st.forward(p, inp, off, parts=True)
    np.testing.assert_allclose(v_off, np.clip(parts_off["current"], *st.VALID_RANGE))


def test_first_arrival_probabilities_sum_below_one():
    cfg = st.Config()
    inp = synthetic_inputs(cfg, n=20)
    p = random_params(cfg)
    zh = inp["x"] @ p["haz"]
    h = 1 / (1 + np.exp(-zh))
    stay = np.cumprod(1 - h, axis=1)
    first = h * np.concatenate([np.ones((20, 1)), stay[:, :-1]], axis=1)
    assert (first.sum(axis=1) <= 1 + 1e-12).all()
    np.testing.assert_allclose(first.sum(axis=1), 1 - stay[:, -1])


def test_numpy_torch_parity():
    torch = pytest.importorskip("torch")
    from training import structural as ts

    cfg = st.Config()
    inp = synthetic_inputs(cfg, n=60)
    p = random_params(cfg)
    v_np = st.forward(p, inp, cfg)
    t_inp = {k: ts.to_torch(v) for k, v in inp.items()}
    t_p = {k: ts.to_torch(v) for k, v in p.items()}
    with torch.no_grad():
        v_t = ts.to_numpy(st.forward(t_p, t_inp, cfg, ts.torch_ops()))
    np.testing.assert_allclose(v_t, v_np, atol=1e-6)


def test_parameter_recovery():
    pytest.importorskip("torch")
    from training import structural as ts

    cfg = st.Config(ridge=0.0)
    inp = synthetic_inputs(cfg, n=600, seed=3)
    true = random_params(cfg, seed=4)
    y = st.forward(true, inp, cfg) + RNG.normal(scale=2.0, size=(600, st.HORIZON))
    q = np.ones_like(y)
    params, info = ts.fit_inputs(cfg, inp, y, q, hazards=(-4.0,), max_iter=300)
    fitted = st.forward(params, inp, cfg)
    clean = st.forward(true, inp, cfg)
    assert np.sqrt(np.mean((fitted - clean) ** 2)) < 3.0  # the noise is 2 km/s
    assert info["rmse_weighted"] < 3.0


def test_asset_round_trip_and_regressor_columns():
    pytest.importorskip("torch")
    from training import structural as ts

    rng = np.random.default_rng(5)
    n_hours = 24 * 400
    t = np.arange(n_hours)
    values = 420 + 120 * np.maximum(0, np.sin(2 * np.pi * t / (27 * 24))) ** 4 \
        + rng.normal(scale=15, size=n_hours)
    positions = np.arange(28 * 24 + 60, n_hours - 80, 6)
    lin = LinearConfig()
    cfg = ts.structural_config(lin)
    model = st.StructuralModel(cfg)
    ts.fit(model, values, positions, None, end=n_hours - 80, hazards=(-5.0,), max_iter=60)
    f = model.forecast(values, positions, None)
    again = st.StructuralModel.from_dict(model.to_dict())
    np.testing.assert_allclose(again.forecast(values, positions, None), f)
    cols = ts.structural_columns(f)
    assert set(cols) == {f"structural_speed_at_{lag}h" for lag in range(-72, 1, 6)}
    np.testing.assert_array_equal(cols["structural_speed_at_0h"], f[:, 0])
    np.testing.assert_array_equal(cols["structural_speed_at_-12h"], f[:, 11])
