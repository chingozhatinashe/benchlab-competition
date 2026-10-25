"""Coronagraph brightness by position angle, from SOHO/LASCO C2 and C3 browse images.

A coronal mass ejection shows in a coronagraph as new brightness moving out from the
occulter. One heading towards (or away from) Earth surrounds the occulter on all sides
(a "halo"); one leaving the side of the Sun brightens a few position angles. No CME
model is used here: the images are reduced to generic numbers, and the forecast
models learn what they mean (docs/coronagraph.md).

Per image (512 x 512 browse JPEG, colour-mapped; north up, east left):
1. Grey levels (the colour maps are monotonic in brightness).
2. The median grey level in each of N_SECTORS position-angle sectors (measured from
   north through east, i.e. counter-clockwise on the image) and each of two radial
   rings above the occulter (RINGS, in pixels from the image centre). Medians ignore
   stars and the odd cosmic ray.
Output: cell values, shape (2, N_SECTORS) per image, or None for an unusable frame.

Brightening (`excess`): each cell relative to its own median over the previous
BACKGROUND hours, so steady streamers, planets and stars cancel; a CME is a positive
excess. Summaries per image: the share of sectors whose outer-ring excess passes
EXCESS_LEVEL (a halo brightens all sectors), the largest excess, and the mean excess.
They are the signals SIGNALS ("c2max", ...), windowed per t0 by
feature_engineering.windows ("c2max_max_24h": the largest c2max in the last 24 h).

Cadence and availability: C2 uses the first image of every hour, C3 of every second
hour (CADENCE_HOURS, `select_times`), the same in history (training.pull_lasco_images)
and live (connectors.lasco). An image counts for t0 if it was taken by t0 - LATENCY
(images are posted about 15 min after they are taken).
"""

import functools
import io

import numpy as np
import pandas as pd

SIZE = 512
CENTER = (SIZE - 1) / 2
N_SECTORS = 12
INSTRUMENTS = ("c2", "c3")
# (inner, outer) ring radii in pixels; just above each occulter and further out
RINGS = {"c2": ((100, 140), (140, 200)), "c3": ((45, 90), (90, 180))}
MIN_MEDIAN = 5.0  # a frame whose ring cells are this dark is blank or truncated
BACKGROUND = pd.Timedelta(hours=6)
MIN_BACKGROUND = 2  # images needed in the background window (C3: 3 per 6 h)
CADENCE_HOURS = {"c2": 1, "c3": 2}
LATENCY = pd.Timedelta(minutes=30)
EXCESS_LEVEL = 0.10  # relative brightening that counts a sector as brightened
SUMMARIES = ("share", "max", "mean")
SIGNALS = tuple(f"{i}{s}" for i in INSTRUMENTS for s in SUMMARIES)

@functools.lru_cache(maxsize=1)
def _geometry():
    """(sector index, radius) of every pixel, computed once (thread-safe: one return)."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    dx, dy = xx - CENTER, CENTER - yy
    pa = (np.degrees(np.arctan2(-dx, dy)) + 360.0) % 360.0  # north 0, east 90
    return (pa // (360.0 / N_SECTORS)).astype(int), np.hypot(dx, dy)


def grayscale(jpeg):
    from PIL import Image  # imported here so the rest of the package works without Pillow

    try:
        with Image.open(io.BytesIO(jpeg)) as im:
            img = np.asarray(im.convert("L"), dtype=float)
    except Exception:
        return None
    return img if img.shape == (SIZE, SIZE) else None


def cells(jpeg, instrument):
    """(2, N_SECTORS) median grey level per ring and sector, or None if unusable."""
    img = grayscale(jpeg)
    if img is None:
        return None
    sector, radius = _geometry()
    out = np.full((2, N_SECTORS), np.nan)
    for r, (lo, hi) in enumerate(RINGS[instrument]):
        ring = (radius >= lo) & (radius < hi)
        for s in range(N_SECTORS):
            px = img[ring & (sector == s)]
            out[r, s] = np.median(px) if px.size else np.nan
    if not np.isfinite(out).all() or np.nanmedian(out) < MIN_MEDIAN:
        return None
    return out


def excess(table):
    """Relative brightening of each cell against its median over the previous BACKGROUND.

    `table`: time-indexed (sorted) frame of cell values for one instrument, columns
    "r{ring}s{sector}". The background uses only earlier images, so a value at time t
    depends on images up to t only.
    """
    table = table.sort_index()
    background = table.rolling(BACKGROUND, closed="left", min_periods=MIN_BACKGROUND).median()
    # a black cell (a partly blanked frame) would divide by zero: floor it at MIN_MEDIAN
    return table / background.clip(lower=MIN_MEDIAN) - 1.0


def summaries(excess_table):
    """Per image: the share of outer-ring sectors past EXCESS_LEVEL, the max and the mean."""
    outer = excess_table[[c for c in excess_table.columns if c.startswith("r1")]]
    out = pd.DataFrame(index=excess_table.index)
    out["share"] = (outer > EXCESS_LEVEL).mean(axis=1).where(outer.notna().all(axis=1))
    out["max"] = excess_table.max(axis=1, skipna=False)
    out["mean"] = excess_table.mean(axis=1, skipna=False)
    return out


def cell_columns():
    return [f"r{r}s{s}" for r in range(2) for s in range(N_SECTORS)]


def select_times(times, instrument):
    """The first image time of each CADENCE_HOURS block of the day, sorted."""
    times = pd.DatetimeIndex(sorted(times))
    if len(times) == 0:
        return times
    block = times.floor(f"{CADENCE_HOURS[instrument]}h")
    return times[~block.duplicated()]


def signal_frame(cell_tables):
    """{instrument: time-indexed cell table} -> time-indexed SIGNALS (NaN where n/a)."""
    parts = [summaries(excess(table[cell_columns()])).add_prefix(instrument)
             for instrument, table in cell_tables.items() if table is not None and len(table)]
    if not parts:
        return pd.DataFrame(columns=list(SIGNALS), dtype=float)
    return pd.concat(parts).sort_index().reindex(columns=list(SIGNALS))
