"""Structural model v2: stream events with an arrival time, a duration and a free speed profile.

At t0, for leads k = 1..72 (x(u): the inputs for hour t0 + u, as models.structural):

    v(k) = a + e0 psi(k / D_c) + sum_{u=1..72} lambda(u) A(u) phi((k - u + L) / (D(u) + L))

    ambient a        as models.structural (250-500 km/s from the 27-day q20 and recurrence)
    current event    e0 = last speed - a, over a remaining duration D_c = 6 + 24 softplus(
                     w_c . x(1)) h, with a free relaxation curve psi (piecewise linear on
                     PSI_KNOTS interior knots over the phase [0, 1], psi(0) = 1, psi(1) = 0)
    new events       Poisson arrivals: intensity lambda(u) = exp(w_lam . x(u)) per hour, so
                     any number of events in the 72 h; overlapping effects add (the expected
                     excess is the intensity-weighted sum)
      duration       D(u) = 6 + 24 softplus(w_D . x(u)) h, from the inputs at arrival
      speeds         peak excess A(u) = 100 softplus(w_A . x(u)) km/s times a free profile
                     phi: piecewise linear on PHI_KNOTS interior knots (softplus values,
                     normalised to a maximum of 1) over the phase [0, 1], 0 at both ends;
                     it starts L = LEAD_IN h before the arrival hour (the labels mark the
                     midpoint of the rise, training.solar_wind_events)
    single=True      only the first arrival counts (as v1): weights (1 - e^-lambda(u))
                     exp(-sum_{s<u} lambda(s)) instead of lambda(u)
Output: the expected speed, clipped to VALID_RANGE. training.stream_events fits every
parameter in one step on the speed loss plus beta x the Poisson likelihood of the detected
stream arrivals. numpy only here; `forward` takes an `ops` namespace (torch in training).
See docs/stream_events.md.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from models import structural as st
from models.final_baseline import VALID_RANGE

FORMAT_VERSION = 1
LEADS = st.LEADS
AMP_SCALE = st.AMP_SCALE
DUR_MIN, DUR_SCALE = 6.0, 24.0  # h
ETA_LAMBDA = (-12.0, 1.0)  # log intensity bounds (per hour)

NUMPY = SimpleNamespace(**vars(st.NUMPY), floor=np.floor, take=lambda v, i: v[i.astype(int)],
                        concat=np.concatenate, stack=np.stack, log=np.log)


@dataclass(frozen=True)
class Config(st.Config):
    beta: float = 1.0  # weight of the arrival-label likelihood (0: speed only)
    single: bool = False  # only the first arrival counts
    phi_knots: int = 10
    psi_knots: int = 7
    lead_in: float = 12.0  # h before the arrival hour where an event's profile starts
    smooth: float = 1e-3  # second-difference penalty on the profile knots

    def shapes(self):
        p = self.n_inputs
        return {"amb": 3, "cur": p, "psi": self.psi_knots, "lam": p, "dur": p, "amp": p,
                "phi": self.phi_knots}


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


def initial_params(cfg, log_rate=-4.8):
    p = {k: np.zeros(n) for k, n in cfg.shapes().items()}
    p["amb"][:2] = [0.08, 1.6]
    p["cur"][0] = _inv_softplus(24.0 / DUR_SCALE)
    p["psi"][:] = 1.0 - np.arange(1, cfg.psi_knots + 1) / (cfg.psi_knots + 1)
    p["lam"][0] = log_rate
    p["dur"][0] = _inv_softplus(30.0 / DUR_SCALE)
    p["amp"][0] = _inv_softplus(1.5)
    t = np.arange(1, cfg.phi_knots + 1) / (cfg.phi_knots + 1)
    p["phi"][:] = [_inv_softplus(max(v, 0.05)) for v in np.sin(np.pi * t)]
    return p


def penalty_mask(cfg):
    """Ridge mask: 1 on the input weights except intercepts; 0 on the profile knots."""
    out = {k: np.ones(n) for k, n in cfg.shapes().items()}
    for k in out:
        out[k][0] = 0.0
    out["psi"][:] = 0.0
    out["phi"][:] = 0.0
    return out


def piecewise(values, phase, ops):
    """Piecewise-linear curve through `values` (end knots included) at equally spaced
    knots over [0, 1], at `phase` (any shape); 0 outside [0, 1]."""
    n = values.shape[0] - 1
    inside = (phase >= 0) * (phase <= 1)
    s = ops.clip(phase, 0.0, 1.0) * n
    i = ops.clip(ops.floor(s), 0.0, n - 1.0)
    frac = s - i
    out = ops.take(values, i) * (1.0 - frac) + ops.take(values, i + 1.0) * frac
    return out * inside


def profiles(p, ops):
    """(phi with end knots, psi with end knots)."""
    knots = ops.softplus(p["phi"])
    knots = knots / ops.max(knots)
    zero = 0.0 * knots[:1]
    phi = ops.concat([zero, knots, zero])
    psi = ops.concat([zero + 1.0, p["psi"], zero])
    return phi, psi


def event_terms(p, x, cfg, ops):
    """(log lambda, lambda, D, A), each (n, 72)."""
    eta = ops.clip(x @ p["lam"], *ETA_LAMBDA)
    dur = DUR_MIN + DUR_SCALE * ops.softplus(x @ p["dur"])
    amp = AMP_SCALE * ops.softplus(x @ p["amp"])
    return eta, ops.exp(eta), dur, amp


def forward(p, inp, cfg, ops=NUMPY, parts=False):
    """(n, 72) expected speeds (and the parts and log intensities if parts=True)."""
    x, last, amb = inp["x"], inp["last"], inp["amb"]
    lo, hi = st.AMBIENT_RANGE
    a = lo + (hi - lo) * ops.sigmoid(amb @ p["amb"])
    e0 = last - a
    phi, psi = profiles(p, ops)
    k = inp["leads"]
    d_cur = DUR_MIN + DUR_SCALE * ops.softplus(x[:, 0, :] @ p["cur"])
    current = a[:, None] + e0[:, None] * piecewise(psi, k[None, :] / d_cur[:, None], ops)
    eta, lam, dur, amp = event_terms(p, x, cfg, ops)
    if cfg.single:
        before = ops.cumsum(lam, 1) - lam
        weight = (1.0 - ops.exp(-lam)) * ops.exp(-before)
    else:
        weight = lam
    # phase[n, k, u] of the event arriving at u, at lead k
    gap = k[None, :, None] - k[None, None, :] + cfg.lead_in
    phase = gap / (dur[:, None, :] + cfg.lead_in)
    stream = ops.einsum("nku,nu->nk", piecewise(phi, phase, ops), weight * amp)
    v = ops.clip(current + stream, *VALID_RANGE)
    if parts:
        return v, {"ambient": a, "current": current, "stream": stream, "eta": eta,
                   "duration": dur, "amplitude": amp}
    return v


class StreamEventsModel(st.StructuralModel):
    """Input preparation from models.structural; parameters from training.stream_events."""

    def forecast(self, values, positions, regressors=None, parts=False):
        inp = self.prepare(values, positions, regressors)
        return forward(self.params, inp, self.config, parts=parts)

    def to_dict(self):
        d = super().to_dict()
        d["format"] = FORMAT_VERSION
        d["kind"] = "stream_events"
        return d

    @classmethod
    def from_dict(cls, d):
        cfg = Config(**{k: tuple(v) if isinstance(v, list) else v
                        for k, v in d["config"].items()})
        base_fields = st.Config.__dataclass_fields__
        base = st.StructuralModel.from_dict(
            {**d, "config": {k: v for k, v in d["config"].items() if k in base_fields}})
        m = cls(cfg)
        for attr in ("mean", "medians", "x_low", "x_high", "x_mean", "x_std"):
            setattr(m.pre, attr, getattr(base.pre, attr))
        m.params = base.params
        return m
