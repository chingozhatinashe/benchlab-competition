"""Wang-Sheeley-Arge (WSA) solar wind speed from a PFSS ring, carried to Earth by HUX.

Numpy only (it runs in the image). The input is a PFSS ring (feature_engineering.pfss): for
each of 360 Carrington longitudes along the sub-Earth latitude on the source surface, the
expansion factor fs and the footpoint's distance theta_b (deg) to the coronal-hole boundary.

1. Source speed, the empirical WSA relation (Arge et al. 2003; McGregor et al. 2011):
       v = v0 + dv / (1 + fs)^alpha * (1 - 0.8 exp(-(theta_b / w)^delta))^3
   capped at V_MAX. Slow wind comes from strongly expanding tubes and from near hole edges.
2. Propagation from R_INNER to Earth by the HUX upwind scheme (Riley & Lionello 2011), the
   kinematic model HUXt is built on, in the frame corotating with the Sun:
       v(r + dr, phi) = v(r, phi) + dr * OMEGA / v(r, phi) * (v(r, phi + dphi) - v(r, phi)) / dphi
   Plasma reaching Earth came from a longitude that faced Earth earlier (a larger Carrington
   longitude); fast wind overtaking slow wind steepens into stream interactions instead of
   overlapping, as a plain ballistic mapping would.
3. The steady corotating solution at 1 AU is read at the Carrington longitude facing Earth at
   each forecast hour (pfss.earth_longitude), shifted by `shift` deg (the corotation between the
   source surface and R_INNER, and any systematic timing error; calibrated).

The parameters (Params) are calibrated on the training years (training.wsa_model); the
rolling variant re-fits an offset and a scale on each run's last days (rolling_fit).
Not built: neither variant passed the fold study (docs/wsa.md).
"""

from dataclasses import asdict, dataclass

import numpy as np

R_SUN_KM = 695_700.0
AU_RSUN = 215.0
R_INNER = 21.5  # R_sun, the usual HUX/Enlil inner boundary (0.1 AU)
OMEGA = 2 * np.pi / (25.38 * 86400)  # rad/s, sidereal (Carrington) rotation
DR = 1.0  # R_sun per HUX step (stable for v > ~110 km/s at 1-deg longitude cells)
V_MIN, V_MAX = 200.0, 950.0
N_LON = 360


@dataclass(frozen=True)
class Params:
    v0: float = 285.0  # km/s, slow-wind floor
    dv: float = 625.0  # km/s, fast minus slow
    alpha: float = 2 / 9  # expansion-factor exponent
    w: float = 2.0  # deg, width of the boundary layer
    delta: float = 1.25  # boundary-layer exponent
    shift: float = 0.0  # deg of Carrington longitude added before reading at Earth

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_vector(cls, x, names):
        return cls(**dict(zip(names, map(float, x), strict=True)))


def fill_cyclic(values):
    """NaNs filled by linear interpolation along the (cyclic) last axis; all-NaN rows stay."""
    v = np.array(values, dtype=float, copy=True)
    flat = v.reshape(-1, v.shape[-1])
    x = np.arange(flat.shape[1])
    for row in flat:
        bad = np.isnan(row)
        if bad.all() or not bad.any():
            continue
        good = np.flatnonzero(~bad)
        xp = np.concatenate([good - flat.shape[1], good, good + flat.shape[1]])
        row[bad] = np.interp(x[bad], xp, np.tile(row[good], 3))
    return v


def source_speed(log_fs, theta_b, p=None):
    """WSA speed (km/s) at the inner boundary, per ring longitude."""
    p = p or Params()
    fs = 10.0 ** fill_cyclic(log_fs)
    tb = np.maximum(fill_cyclic(theta_b), 0.0)
    w = max(p.w, 1e-3)
    edge = (1.0 - 0.8 * np.exp(-((tb / w) ** p.delta))) ** 3
    v = p.v0 + p.dv / (1.0 + fs) ** p.alpha * edge
    return np.clip(v, V_MIN, V_MAX)


def hux(v_inner, r_inner=R_INNER, r_outer=AU_RSUN, dr=DR):
    """Steady HUX solution at r_outer (km/s) from the speed at r_inner, per longitude.

    v_inner: (..., 360), longitude j = Carrington longitude j deg.
    """
    v = np.clip(np.asarray(v_inner, dtype=float), V_MIN, V_MAX)
    steps = int(np.ceil((r_outer - r_inner) / dr))
    dr_km = (r_outer - r_inner) / steps * R_SUN_KM
    dphi = 2 * np.pi / v.shape[-1]
    k = dr_km * OMEGA / dphi
    for _ in range(steps):
        v = v + k / v * (np.roll(v, -1, axis=-1) - v)
    return v


def at_longitude(ring, lon):
    """ring (n, 360) read at longitudes lon (n, m) deg, linear and cyclic -> (n, m)."""
    lon = np.asarray(lon, dtype=float) % N_LON
    lo = np.floor(lon).astype(int)
    hi = (lo + 1) % N_LON
    w = lon - lo
    rows = np.arange(ring.shape[0])[:, None]
    return (1 - w) * ring[rows, lo] + w * ring[rows, hi]


def earth_speed(log_fs, theta_b, earth_lon, p=None):
    """WSA/HUX speed at Earth (km/s): rings (n, 360), Earth's longitude per hour (n, m)."""
    p = p or Params()
    v1 = hux(source_speed(log_fs, theta_b, p))
    return at_longitude(v1, np.asarray(earth_lon) + p.shift)


def rolling_fit(hindcast, observed, ridge=1.0, prior=(0.0, 1.0)):
    """(offset, scale) of observed ~ offset + scale * hindcast over a run's past hours.

    Least squares with a ridge pull toward `prior` (in units of the hindcast's variance), so
    a short or flat window keeps the offline calibration. NaN hours are ignored.
    """
    h, o = np.asarray(hindcast, float), np.asarray(observed, float)
    ok = np.isfinite(h) & np.isfinite(o)
    if ok.sum() < 24:
        return prior
    h, o = h[ok], o[ok]
    hm, om = h.mean(), o.mean()
    var = ((h - hm) ** 2).sum()
    lam = ridge * len(h) * max(h.var(), 1.0)
    scale = (((h - hm) * (o - om)).sum() + lam * prior[1]) / (var + lam)
    return om - scale * hm, scale
