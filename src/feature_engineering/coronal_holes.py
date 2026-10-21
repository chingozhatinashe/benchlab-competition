"""Coronal-hole area from SDO/AIA 193 Å browse images.

Coronal holes are the dark, open-field regions in 193 Å that feed fast solar wind.
Wind from a hole near the central meridian reaches Earth about 3-4 days later, and
the fraction of a central-meridian strip covered by holes predicts the coming peak
speed (Vršnak et al. 2007, Solar Phys. 240, 315; Rotter et al. 2012).

Method, on a 512 x 512 browse JPEG:
1. Grayscale, smoothed with a 5 x 5 box to suppress JPEG noise and bright points.
2. Disk geometry from the date: the image is centred and has a fixed plate scale,
   so the radius follows the Earth-Sun distance (about 197-203 px).
3. A pixel is coronal hole when it is below THRESHOLD x the median on-disk
   intensity (within 0.9 R, which avoids the bright limb).
4. Heliographic latitude/longitude of each pixel (B0 tilt ignored, < 7.25 deg),
   and an area weight 1/mu so each pixel counts the solar surface it covers.

Outputs (area fractions, 0-1, latitudes within +-60 deg):
    ch_cm    central-meridian strip, |longitude| <= 10 deg
    ch_east  eastern strip, longitude -40..-20 deg (holes rotating towards the meridian)
    ch_disk  |longitude| <= 60 deg

Filament channels are dark in 193 Å too and are counted as holes; without a
magnetogram the hole polarity is unknown. A blank or corrupt frame (it happens
during calibrations and eclipses) fails `usable` and should be skipped.
"""

import io

import numpy as np

SIZE = 512
PLATE_SCALE = 0.6 * 4096 / SIZE  # arcsec per pixel
R_SUN_1AU = 959.63  # apparent solar radius at 1 AU, arcsec
CENTER = (SIZE - 1) / 2
THRESHOLD = 0.40  # set by eye on training-year images: holes kept, quiet-Sun lanes left out
INNER = 0.9  # fraction of the radius used for the median, away from the limb
MAX_LAT = 60.0
MIN_DISK_MEDIAN = 30.0  # a real disk has a median grey level of ~90; blank frames ~0
BLACK = 8.0  # grey level of missing pixels
MAX_BLACK = 0.15  # largest share of near-black pixels on the inner disk (hole cores reach ~5%)
MAX_BLACK_ROWS = 4  # rows of the inner disk that are almost all black (a truncated frame)
SMOOTH = 2  # half-width of the box filter, pixels
REGIONS = {"ch_cm": (-10.0, 10.0), "ch_east": (-40.0, -20.0), "ch_disk": (-60.0, 60.0)}
NAMES = tuple(REGIONS)


def earth_sun_distance(t):
    """Earth-Sun distance (AU) on a date, to about 0.1%."""
    day = t.dayofyear + t.hour / 24
    return 1.0 - 0.01672 * np.cos(2 * np.pi * (day - 4.0) / 365.256)


def disk_radius(t, size=SIZE):
    """Solar radius (pixels) in a browse image taken at t."""
    return R_SUN_1AU / earth_sun_distance(t) / (0.6 * 4096 / size)


def grayscale(jpeg):
    """JPEG bytes -> float grey levels (0-255), or None if it can't be decoded."""
    from PIL import Image  # imported here so the rest of the package works without Pillow

    try:
        with Image.open(io.BytesIO(jpeg)) as im:
            return np.asarray(im.convert("L"), dtype=float)
    except Exception:
        return None


def box_smooth(img, half):
    """Mean over a (2*half+1)^2 box, with the edges padded by reflection."""
    if half <= 0:
        return img
    k = 2 * half + 1
    padded = np.pad(img, half, mode="reflect")
    c = padded.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    total = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return total / (k * k)


def heliographic(radius, size=SIZE):
    """Latitude, longitude (deg) and mu = cos(angle from disk centre) for each pixel.

    NaN off the disk. Solar north is up and east is left, so longitude is
    negative on the left (east) half.
    """
    yy, xx = np.mgrid[:size, :size]
    x = (xx - CENTER) / radius
    y = (CENTER - yy) / radius
    rho2 = x**2 + y**2
    on = rho2 < 1.0
    mu = np.where(on, np.sqrt(np.clip(1.0 - rho2, 0.0, None)), np.nan)
    lat = np.where(on, np.degrees(np.arcsin(np.clip(y, -1, 1))), np.nan)
    lon = np.where(on, np.degrees(np.arctan2(x, mu)), np.nan)
    return lat, lon, mu


def disk_median(img, radius):
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    inner = np.hypot(xx - CENTER, yy - CENTER) < INNER * radius
    return float(np.median(img[inner]))


def usable(img, taken):
    """A full 512 x 512 frame with a normal disk brightness in every quadrant.

    Rejects blank frames and truncated ones (part of the image black), which both
    occur in the browse archive. Even a large hole is not pure black in 193 Å.
    """
    if img is None or img.shape != (SIZE, SIZE):
        return False
    radius = disk_radius(taken)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    x, y = xx - CENTER, yy - CENTER
    inner = np.hypot(x, y) < INNER * radius
    black = (img < BLACK) & inner
    if black.sum() > MAX_BLACK * inner.sum():
        return False
    row_share = black.sum(1) / np.maximum(inner.sum(1), 1)
    if np.sum((row_share > 0.9) & (inner.sum(1) > 20)) > MAX_BLACK_ROWS:
        return False
    quadrants = [inner & (x < 0) & (y < 0), inner & (x >= 0) & (y < 0),
                 inner & (x < 0) & (y >= 0), inner & (x >= 0) & (y >= 0)]
    return all(np.median(img[q]) >= MIN_DISK_MEDIAN for q in quadrants)


def segment(img, taken, threshold=THRESHOLD):
    """Boolean coronal-hole mask (on-disk pixels darker than threshold x the disk median)."""
    radius = disk_radius(taken)
    smooth = box_smooth(img, SMOOTH)
    level = threshold * disk_median(smooth, radius)
    _, _, mu = heliographic(radius)
    return np.isfinite(mu) & (smooth < level)


def area_fractions(mask, taken):
    """Area-weighted hole fraction in each region of REGIONS (|lat| <= MAX_LAT)."""
    lat, lon, mu = heliographic(disk_radius(taken))
    # 1/mu turns a pixel into the surface it covers; capped near the limb
    weight = np.where(np.isfinite(mu), 1.0 / np.maximum(mu, 0.2), 0.0)
    band = np.isfinite(lat) & (np.abs(np.nan_to_num(lat)) <= MAX_LAT)
    out = {}
    for name, (lo, hi) in REGIONS.items():
        region = band & (np.nan_to_num(lon, nan=999) >= lo) & (np.nan_to_num(lon, nan=999) <= hi)
        w = weight[region]
        out[name] = float((w * mask[region]).sum() / w.sum()) if w.sum() > 0 else np.nan
    return out


def metrics(jpeg, taken, threshold=THRESHOLD):
    """{ch_cm, ch_east, ch_disk} for one image; all NaN if the frame is unusable."""
    img = grayscale(jpeg)
    if not usable(img, taken):
        return dict.fromkeys(NAMES, np.nan)
    return area_fractions(segment(img, taken, threshold), taken)
