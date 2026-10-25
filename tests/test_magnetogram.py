import io

import numpy as np
import pandas as pd
import pytest

from connectors.sdo import parse_listing
from feature_engineering import magnetogram as mg

T = pd.Timestamp("2024-03-20T12:00Z")  # B0 about -7 deg
SIZE = 512


def synthetic(shift=(0, 0), patch=None, level=mg.ZERO_LEVEL):
    """A grey disk at the expected radius (optionally off-centre) with a bright patch.

    `patch` = (x, y) in disk-radius units from the disk centre (x west, y north).
    """
    r = mg.expected_radius(T, SIZE)
    c = (SIZE - 1) / 2
    yy, xx = np.mgrid[:SIZE, :SIZE]
    cy, cx = c + shift[0], c + shift[1]
    img = np.zeros((SIZE, SIZE))
    img[np.hypot(yy - cy, xx - cx) < r] = level
    if patch is not None:
        px, py = cx + patch[0] * r, cy - patch[1] * r
        img[np.hypot(yy - py, xx - px) < 0.04 * r] = 250.0
    return img


def jpeg(img):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def cell_of(lat, lon):
    i = np.searchsorted(mg.LAT_EDGES, lat, side="right") - 1
    j = np.searchsorted(mg.LON_EDGES, lon, side="right") - 1
    return i, j


def test_quiet_disk_has_no_field_and_covers_the_visible_cells():
    g = mg.reduce(jpeg(synthetic()), T)
    assert g.shape == mg.grid_shape()
    signed, pixels = g[mg.STATS.index("signed")], g[mg.STATS.index("pixels")]
    assert np.nanmax(np.abs(signed)) < 0.02
    i, j = cell_of(5.0, 5.0)
    assert pixels[i, j] > 0 and np.isnan(signed[-1, 0])  # the far north-east corner is not seen


@pytest.mark.parametrize("shift", [(0, 0), (6, -9)])
def test_a_patch_lands_in_its_heliographic_cell_even_off_centre(shift):
    g = mg.reduce(jpeg(synthetic(shift=shift, patch=(0.3, 0.0))), T)
    positive = g[mg.STATS.index("positive")]
    # x = 0.3 R on the disk-centre row: longitude ~asin(0.3 / cos(B0)) ~ 17.6 deg,
    # latitude ~B0 (the row through the disk centre is at the B0 latitude)
    i, j = np.unravel_index(np.nanargmax(positive), positive.shape)
    assert (i, j) == cell_of(-7.0 + 0.1, 17.6)


def test_unusable_frames():
    assert mg.reduce(jpeg(np.zeros((SIZE, SIZE))), T) is None  # blank
    assert mg.reduce(jpeg(synthetic(level=200.0)), T) is None  # not a magnetogram
    assert mg.reduce(b"not a jpeg", T) is None


def test_sdo_listing_reads_hmi_names_and_keeps_aia():
    html = ('<a href="20240510_120000_1024_HMIB.jpg"></a><a href="20240510_120000_1024_HMIBC.jpg">'
            '<a href="20240510_121500_512_HMIB.jpg"></a><a href="20240510_120403_512_0193.jpg">')
    assert list(parse_listing(html, 1024, "HMIB")) == [pd.Timestamp("2024-05-10T12:00Z")]
    assert list(parse_listing(html, 512, "0193")) == [pd.Timestamp("2024-05-10T12:04:03Z")]


def grid_with(signed_cm=0.0, signed_elsewhere=0.0):
    """One synthetic grid: uniform signed field, a different value in the central strip."""
    n_lat, n_lon = len(mg.LAT_EDGES) - 1, len(mg.LON_EDGES) - 1
    g = np.zeros((1, len(mg.STATS), n_lat, n_lon))
    lon = (mg.LON_EDGES[:-1] + mg.LON_EDGES[1:]) / 2
    cm = np.abs(lon) <= mg.CM_LON
    g[0, mg.STATS.index("signed")] = np.where(cm, signed_cm, signed_elsewhere)[None, :]
    g[0, mg.STATS.index("unsigned")] = np.abs(g[0, mg.STATS.index("signed")])
    g[0, mg.STATS.index("mu")] = 1.0
    g[0, mg.STATS.index("pixels")] = 100.0
    return g


def test_signals_read_the_central_strip():
    s = mg.signals(grid_with(signed_cm=-0.2, signed_elsewhere=0.05))
    # the rest of the disk (0.05) is taken as the image's zero level
    assert s["mg_cm_signed"][0] == pytest.approx(-0.25)
    assert s["mg_cm_unipolar"][0] == pytest.approx(0.25)
    assert s["mg_cm_unsigned"][0] == pytest.approx(0.2)
    assert s["mg_polar"][0] == pytest.approx(0.0)
    assert 0.05 < s["mg_disk_unsigned"][0] < 0.2


def test_magnetogram_windows_use_only_cutoffs_before_t0():
    from feature_engineering import build, windows

    t0 = pd.Timestamp("2024-05-10T12:00Z")
    cut = pd.date_range(t0 - pd.Timedelta(days=4), t0 + pd.Timedelta(days=1), freq="6h")
    frame = pd.DataFrame({s: np.arange(len(cut), dtype=float) for s in mg.SIGNALS}, index=cut)
    spec = [windows.WindowSpec.parse("mg_cm_unipolar_at_0h"),
            windows.WindowSpec.parse("mg_cm_unipolar_at_24h")]
    row = windows.window_table(build.Inputs(magnetogram=frame), [t0], spec).iloc[0]
    latest = cut.get_loc(cut[cut <= t0 - build.CH_LATENCY][-1])
    assert row["mg_cm_unipolar_at_0h"] == latest
    assert row["mg_cm_unipolar_at_24h"] == latest - 4
    future = frame.copy()
    future.loc[future.index > t0 - build.CH_LATENCY] = 1e6  # scrambling the future changes nothing
    again = windows.window_table(build.Inputs(magnetogram=future), [t0], spec).iloc[0]
    assert again.equals(row)
