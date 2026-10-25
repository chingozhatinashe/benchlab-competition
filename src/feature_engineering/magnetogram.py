"""Photospheric magnetic field from SDO/HMI quick-look magnetogram browse images.

The browse image ("HMIB", e.g. 20240510_120000_1024_HMIB.jpg on sdo.gsfc.nasa.gov) is
the line-of-sight field in grey: ZERO_LEVEL is no field, brighter is positive (out of
the Sun), darker negative, with strong fields saturated. The gauss scale is not
recoverable from the JPEG, so everything here is in grey levels; the models learn
what the numbers mean (docs/magnetogram.md). History and live read the same images.

Per image (`reduce`):
1. The disk is found in the image itself (centre and radius from the on-disk pixels),
   so small pointing or size changes don't matter; a frame whose disk doesn't match
   the expected size, or whose disk median is far from ZERO_LEVEL, is unusable.
2. Heliographic latitude and longitude from the disk centre (longitude relative to the
   central meridian, Earth's B0 tilt included: coronal_holes.heliographic_b0).
3. A grid of LAT_EDGES x LON_EDGES cells over the visible disk. Per cell, STATS:
       signed     mean of (grey - ZERO_LEVEL) / 128   (the net polarity)
       unsigned   mean of |grey - ZERO_LEVEL| / 128   (total field, noise included)
       positive   share of pixels above ZERO_LEVEL + STRONG
       negative   share of pixels below ZERO_LEVEL - STRONG
       mu         mean cos(angle from disk centre) (line-of-sight to radial, later)
       pixels     pixel count (0: the cell is off the disk)
   Output: float32 array (len(STATS), n_lat, n_lon), or None for an unusable frame.
"""

import functools

import numpy as np

from feature_engineering import coronal_holes as ch

ZERO_LEVEL = 127.0
STRONG = 40.0  # grey levels from zero that count a pixel as strong field (network, plage)
LAT_EDGES = np.arange(-90.0, 90.1, 10.0)
LON_EDGES = np.arange(-80.0, 80.1, 10.0)
STATS = ("signed", "unsigned", "positive", "negative", "mu", "pixels")
PLATE_SCALE_4096 = 0.504  # HMI arcsec per pixel at 4096 x 4096
DISK_LEVEL = 8.0  # grey level separating the disk from the black sky
MAX_RADIUS_ERROR = 0.03  # relative difference to the expected radius
MAX_MEDIAN_OFFSET = 12.0  # an on-disk median further than this from ZERO_LEVEL is not a magnetogram
INNER = 0.95  # fraction of the radius used (the limb is noisy and foreshortened)


def expected_radius(taken, size):
    """Solar radius in pixels for an HMI browse image of `size` pixels taken at `taken`."""
    return ch.R_SUN_1AU / ch.earth_sun_distance(taken) / (PLATE_SCALE_4096 * 4096 / size)


def find_disk(img):
    """(row centre, column centre, radius) of the disk, from the extent of the pixels
    above DISK_LEVEL within a centred circle (which leaves out the caption in the
    lower-left corner; saturated black spots inside the disk don't change the extent)."""
    size = img.shape[0]
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    on = (img > DISK_LEVEL) & (np.hypot(yy - c, xx - c) < 0.49 * size)
    rows, cols = np.flatnonzero(on.any(axis=1)), np.flatnonzero(on.any(axis=0))
    if len(rows) < 10 or len(cols) < 10:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    return (r0 + r1) / 2, (c0 + c1) / 2, ((r1 - r0) + (c1 - c0) + 2) / 4


@functools.lru_cache(maxsize=8)
def _grid(size, radius_px, b0_cdeg):
    """Cell index (flat; -1 off the grid) and mu per pixel, for a centred disk."""
    radius, b0 = radius_px / 10.0, b0_cdeg / 100.0
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    x, y = (xx - c) / radius, (c - yy) / radius
    rho2 = x**2 + y**2
    on = rho2 < INNER**2
    z = np.sqrt(np.clip(1.0 - rho2, 0.0, None))
    sb, cb = np.sin(np.radians(b0)), np.cos(np.radians(b0))
    lat = np.degrees(np.arcsin(np.clip(y * cb + z * sb, -1, 1)))
    lon = np.degrees(np.arctan2(x, z * cb - y * sb))
    i = np.searchsorted(LAT_EDGES, lat, side="right") - 1
    j = np.searchsorted(LON_EDGES, lon, side="right") - 1
    ok = on & (i >= 0) & (i < len(LAT_EDGES) - 1) & (j >= 0) & (j < len(LON_EDGES) - 1)
    cell = np.where(ok, i * (len(LON_EDGES) - 1) + j, -1)
    return cell, z


def grid_shape():
    return len(STATS), len(LAT_EDGES) - 1, len(LON_EDGES) - 1


# ---------------------------------------------------------------------- per-image signals
# Generic summaries of the grid (path (i) of item 4: measurements, no model of the corona).
# Cells are weighted by their surface area (cos latitude). "Br" is the line-of-sight
# value divided by mu, the usual radial-field estimate (|lat| <= 60 keeps mu >= 0.5 in
# the central strip).
CM_LON = 10.0  # the central-meridian strip, |longitude| <= 10 deg (as coronal_holes.ch_cm)
DISK_LON = 60.0
MAX_LAT = 60.0
POLAR_LAT = (60.0, 80.0)
SIGNALS = ("mg_cm_signed", "mg_cm_unipolar", "mg_cm_unsigned", "mg_cm_strong",
           "mg_cm_imbalance", "mg_disk_unsigned", "mg_polar")


def _centres(edges):
    return (edges[:-1] + edges[1:]) / 2


def _mean(stat, pixels, weight, region):
    """Area-weighted mean of `stat` over the cells in `region` that have pixels: (n,)."""
    w = np.where(region & (pixels > 0) & np.isfinite(stat), weight, 0.0)
    total = w.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (np.nan_to_num(stat) * w).sum(axis=(-2, -1)) / total, np.nan)


def signals(grid):
    """SIGNALS per image from grids of shape (n, len(STATS), n_lat, n_lon): {name: (n,)}."""
    grid = np.asarray(grid, dtype=float)
    g = {s: grid[:, k] for k, s in enumerate(STATS)}
    lat, lon = np.meshgrid(_centres(LAT_EDGES), _centres(LON_EDGES), indexing="ij")
    weight = np.cos(np.radians(lat))
    band = np.abs(lat) <= MAX_LAT
    cm = band & (np.abs(lon) <= CM_LON)
    disk = band & (np.abs(lon) <= DISK_LON)
    # the JPEG's zero level drifts by about half a grey level (2016-2018: mean -0.46, sd
    # 0.25): take each image's zero as the median signed value of its disk cells (mostly
    # quiet Sun, whose net field is near zero)
    px = g["pixels"]
    quiet = np.where(disk & (px > 0), g["signed"], np.nan)
    has = np.isfinite(quiet).any(axis=(-2, -1))
    zero = np.full(len(grid), np.nan)
    zero[has] = np.nanmedian(quiet[has], axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        br = (g["signed"] - zero[:, None, None]) / g["mu"]
    signed = _mean(br, px, weight, cm)
    pos, neg = _mean(g["positive"], px, weight, cm), _mean(g["negative"], px, weight, cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        imbalance = np.abs(pos - neg) / (pos + neg)
    north = (lat >= POLAR_LAT[0]) & (lat <= POLAR_LAT[1]) & (np.abs(lon) <= DISK_LON)
    south = (lat <= -POLAR_LAT[0]) & (lat >= -POLAR_LAT[1]) & (np.abs(lon) <= DISK_LON)
    return {
        "mg_cm_signed": signed,
        "mg_cm_unipolar": np.abs(signed),
        "mg_cm_unsigned": _mean(g["unsigned"], px, weight, cm),
        "mg_cm_strong": pos + neg,
        "mg_cm_imbalance": np.where(pos + neg > 0, imbalance, np.nan),
        "mg_disk_unsigned": _mean(g["unsigned"], px, weight, disk),
        "mg_polar": _mean(br, px, weight, north) - _mean(br, px, weight, south),
    }


def reduce(jpeg, taken):
    """STATS per (latitude, longitude) cell, or None if the frame is unusable."""
    img = ch.grayscale(jpeg)
    if img is None or img.ndim != 2 or img.shape[0] != img.shape[1]:
        return None
    size = img.shape[0]
    disk = find_disk(img)
    if disk is None:
        return None
    r0, c0, radius = disk
    if abs(radius / expected_radius(taken, size) - 1) > MAX_RADIUS_ERROR:
        return None
    # shift the image so the disk is centred (whole pixels; sub-pixel offsets don't matter)
    c = (size - 1) / 2
    img = np.roll(img, (round(c - r0), round(c - c0)), axis=(0, 1))
    cell, mu = _grid(size, round(radius * 10), round(float(ch.b0_angle(taken)) * 100))
    on = cell >= 0
    if abs(np.median(img[on]) - ZERO_LEVEL) > MAX_MEDIAN_OFFSET:
        return None
    v = (img[on] - ZERO_LEVEL) / 128.0
    k = cell[on]
    n_cells = (len(LAT_EDGES) - 1) * (len(LON_EDGES) - 1)
    count = np.bincount(k, minlength=n_cells).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        def mean(w):
            return np.bincount(k, weights=w, minlength=n_cells) / count

        strong = STRONG / 128.0
        out = np.stack([mean(v), mean(np.abs(v)), mean((v > strong).astype(float)),
                        mean((v < -strong).astype(float)), mean(mu[on]), count])
    return out.reshape(grid_shape()).astype(np.float32)
