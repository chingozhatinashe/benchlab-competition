"""Potential-field source-surface (PFSS) summaries of one GONG synoptic magnetogram.

The solver and the field-line tracer are sunkit-magex's (`sunkit_magex.pfss`, the successor
of pfsspy) on sunpy maps; this module only chooses the seeds and reduces the traced field to
numbers. One map gives a "ring": values at 360 Carrington longitudes (1 deg apart) along the
sub-Earth latitude B0 on the source surface, the latitude the solar wind that reaches Earth
leaves from. A forecast for hour t0 + k reads the ring at the longitude that faced Earth some
days earlier (feature_engineering.windows), so one map serves every lead.

Ring variables (RING_VARS), for the field line traced down from each source-surface point:
    log_fs     log10 of the flux-tube expansion factor (r_sun^2 B_sun) / (r_ss^2 B_ss)
    theta_b    angular distance (deg) from its photospheric footpoint to the open/closed
               boundary (midpoints of neighbouring open and closed seeds): the distance to
               the coronal-hole boundary (WSA's theta_b / DCHB)
    foot_lat   |heliographic latitude| (deg) of the footpoint
    foot_pol   the footpoint's polarity (+1 / -1)
    br_ss      the source-surface radial field (G) at the point
    d_hcs      angular distance (deg) on the source surface to the current sheet (Br_ss = 0)

Scalars per map (SCALARS): the open share of the photosphere (area-weighted), the open flux
(|Br_ss| over the source surface, in G R_sun^2) and the current sheet's highest latitude.

Settings are the usual ones for GONG maps: 35 radial cells, source surface at 2.5 R_sun,
the photosphere seeded every 2 deg.
"""

import contextlib
import gzip
import io
import warnings
from dataclasses import dataclass

import numpy as np

NR = 35
RSS = 2.5
N_LON = 360
PHOTO_STEP = 2.0  # deg, photospheric seed grid for the open/closed mask
SEED_HEIGHT = 1.01  # R_sun, just above the inner boundary
RING_VARS = ("log_fs", "theta_b", "foot_lat", "foot_pol", "br_ss", "d_hcs")
SCALARS = ("open_area", "open_flux", "hcs_max_lat")


@dataclass
class Ring:
    taken: object  # observation time of the map (pd.Timestamp, UTC)
    b0: float  # sub-Earth latitude (deg) used for the ring
    ring: np.ndarray  # float32 (len(RING_VARS), N_LON); column j is Carrington longitude j deg
    scalars: np.ndarray  # float32 (len(SCALARS),)


def ring_shape():
    return len(RING_VARS), N_LON


@dataclass
class RingTable:
    """Rings by the t0 they serve (history: training.pull_gong; live: one row at t0)."""

    t0: object  # pd.DatetimeIndex (UTC), sorted
    taken: object  # pd.DatetimeIndex with NaT where no map was usable
    ring: np.ndarray  # float32 (n, len(RING_VARS), N_LON), NaN rows where none
    scalars: np.ndarray  # float32 (n, len(SCALARS))

    @classmethod
    def single(cls, t0, ring):
        """A one-row table for a live run (ring may be None: an all-NaN row)."""
        import pandas as pd

        t0 = pd.DatetimeIndex([pd.Timestamp(t0)])
        if ring is None:
            return cls(t0, pd.DatetimeIndex([pd.NaT], tz="UTC"),
                       np.full((1, *ring_shape()), np.nan, np.float32),
                       np.full((1, len(SCALARS)), np.nan, np.float32))
        return cls(t0, pd.DatetimeIndex([pd.Timestamp(ring.taken)]), ring.ring[None],
                   ring.scalars[None])

    def __len__(self):
        return len(self.t0)


def earth_longitude(times, step_hours=6):
    """Earth's Carrington longitude (deg, [0, 360)) at each time (sunpy's L0).

    Computed on a step_hours grid around the times and interpolated: L0 is smooth, and sunpy
    takes ~0.4 ms per time.
    """
    import pandas as pd
    from astropy.time import Time
    from sunpy.coordinates import sun

    times = pd.DatetimeIndex(times)
    if len(times) == 0:
        return np.array([])
    step = pd.Timedelta(hours=step_hours)
    grid = pd.date_range(times.min().floor(step) - step, times.max().ceil(step) + step,
                         freq=step)
    l0 = np.unwrap(np.radians(sun.L0(Time(grid.tz_convert(None).to_numpy())).deg))
    x = np.interp(times.asi8.astype(float), grid.asi8.astype(float), l0)
    return np.degrees(x) % 360


def _unit(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def nearest_angle(points, targets):
    """Angle (deg) from each point (n, 3 unit vectors) to its nearest target (m, 3)."""
    if len(targets) == 0:
        return np.full(len(points), 180.0)
    dots = np.clip(points @ targets.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots.max(axis=1)))


def sheet_points(lat, lon, br):
    """Unit vectors on a sign boundary: midpoints between neighbouring cells of opposite sign.

    Used for the current sheet (Br_ss = 0) and the coronal-hole boundary (open vs closed).

    lat, lon: 2-D (deg) cell centres of a lat x lon grid; br: the field on it. Longitude wraps.
    """
    sign = np.sign(br)
    pts = []
    flip = sign != np.roll(sign, -1, axis=1)  # east neighbour, cyclic
    v = _unit(lat, lon)
    pts.append(((v + np.roll(v, -1, axis=1)) / 2)[flip])
    flip = sign[:-1] != sign[1:]  # north neighbour
    pts.append(((v[:-1] + v[1:]) / 2)[flip])
    p = np.concatenate(pts)
    return p / np.linalg.norm(p, axis=1, keepdims=True) if len(p) else p


def load_map(fits_gz):
    """A sunpy map from a GONG .fits.gz file's bytes."""
    import sunpy.map
    from astropy.io import fits

    raw = gzip.decompress(fits_gz) if fits_gz[:2] == b"\x1f\x8b" else fits_gz
    with fits.open(io.BytesIO(raw)) as hdul:
        hdu = hdul[0]
        data = np.nan_to_num(hdu.data.astype(np.float64))
        header = hdu.header.copy()
    return sunpy.map.Map(data, header)


def solve(br_map, nr=NR, rss=RSS):
    from sunkit_magex import pfss

    return pfss.pfss(pfss.Input(br_map, nr, rss))


def clear_caches():
    """Empty sunkit-magex's functools.lru_cache method caches.

    They are keyed on the field-line and output objects, so each entry keeps a whole solved
    model (about 150 MB) alive: without this, memory grows with every map solved.
    """
    from sunkit_magex.pfss import fieldline, grid, output

    for module in (fieldline, grid, output):
        for cls in vars(module).values():
            if not isinstance(cls, type):
                continue
            for attr in vars(cls).values():
                fn = attr.fget if isinstance(attr, property) else attr
                if hasattr(fn, "cache_clear"):
                    fn.cache_clear()


def reduce(fits_gz, taken=None, nr=NR, rss=RSS):
    """Ring of one GONG map (bytes of the .fits.gz), or None if it can't be read or solved."""
    try:
        return _reduce(fits_gz, taken, nr, rss)
    finally:
        with contextlib.suppress(Exception):
            clear_caches()


def _reduce(fits_gz, taken, nr, rss):
    import astropy.units as u
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from sunkit_magex.pfss import tracing
    from sunpy.coordinates import sun
    from sunpy.map import all_coordinates_from_map

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            br_map = load_map(fits_gz)
            if not np.isfinite(br_map.data).all() or np.abs(br_map.data).max() == 0:
                return None
            taken = pd.Timestamp(taken if taken is not None else br_map.date.datetime)
            taken = taken.tz_localize("UTC") if taken.tzinfo is None else taken
            out = solve(br_map, nr, rss)
            frame = out.coordinate_frame
            tracer = tracing.PerformanceTracer()

            # Open/closed mask of the photosphere.
            lat = np.arange(-90 + PHOTO_STEP / 2, 90, PHOTO_STEP)
            lon = np.arange(PHOTO_STEP / 2, 360, PHOTO_STEP)
            glon, glat = np.meshgrid(lon, lat)
            seeds = SkyCoord(glon.ravel() * u.deg, glat.ravel() * u.deg,
                             SEED_HEIGHT * u.R_sun, frame=frame)
            pol = tracer.trace(seeds, out).polarities.reshape(glat.shape)
            boundary = sheet_points(glat, glon, np.where(pol != 0, 1.0, -1.0))
            open_area = float((np.cos(np.radians(glat)) * (pol != 0)).sum()
                              / np.cos(np.radians(glat)).sum())

            # Source-surface field and current sheet.
            ss = out.source_surface_br
            ss_coord = all_coordinates_from_map(ss)
            ss_lat = ss_coord.lat.to_value(u.deg)
            ss_lon = ss_coord.lon.to_value(u.deg) % 360
            ss_br = np.asarray(ss.data, dtype=np.float64)
            hcs = sheet_points(ss_lat, ss_lon, ss_br)
            hcs_max_lat = float(np.degrees(np.arcsin(np.abs(hcs[:, 2]).max()))) if len(hcs) \
                else np.nan
            open_flux = float(np.abs(ss_br).mean() * 4 * np.pi * rss**2)

            # The sub-Earth ring, traced down to the photosphere.
            b0 = float(sun.B0(taken.to_pydatetime()).to_value(u.deg))
            ring_lon = np.arange(N_LON, dtype=np.float64)
            ring_lat = np.full(N_LON, b0)
            ring = SkyCoord(ring_lon * u.deg, ring_lat * u.deg, rss * 0.999 * u.R_sun,
                            frame=frame)
            lines = tracer.trace(ring, out)
            fs = np.array([fl.expansion_factor if fl.is_open else np.nan for fl in lines],
                          dtype=np.float64)
            foot_pol = np.array([fl.polarity if fl.is_open else np.nan for fl in lines],
                                dtype=np.float64)
            foot_lat = np.full(N_LON, np.nan)
            foot_lon = np.full(N_LON, np.nan)
            for j, fl in enumerate(lines):
                if fl.is_open:
                    c = fl.solar_footpoint
                    foot_lat[j] = c.lat.to_value(u.deg)
                    foot_lon[j] = c.lon.to_value(u.deg)
            theta_b = np.full(N_LON, np.nan)
            ok = np.isfinite(foot_lat)
            theta_b[ok] = nearest_angle(_unit(foot_lat[ok], foot_lon[ok]), boundary)

            # Source-surface Br at the ring, from the nearest cell of the map.
            px = ss.wcs.world_to_pixel(ring)
            ix = np.clip(np.rint(px[0]).astype(int), 0, ss_br.shape[1] - 1)
            iy = np.clip(np.rint(px[1]).astype(int), 0, ss_br.shape[0] - 1)
            br_ring = ss_br[iy, ix]
            d_hcs = nearest_angle(_unit(ring_lat, ring_lon), hcs)
        except Exception:
            return None

    with np.errstate(divide="ignore", invalid="ignore"):
        log_fs = np.log10(fs)
    values = np.stack([log_fs, theta_b, np.abs(foot_lat), foot_pol, br_ring, d_hcs])
    return Ring(taken, b0, values.astype(np.float32),
                np.array([open_area, open_flux, hcs_max_lat], dtype=np.float32))
