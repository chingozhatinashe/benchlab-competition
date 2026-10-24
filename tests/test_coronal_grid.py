"""feature_engineering.coronal_holes: B0 and the latitude-grid metrics (training.coronal_grid)."""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import coronal_holes as c


@pytest.mark.parametrize("day,b0", [("2024-03-06", -7.25), ("2024-06-06", 0.0),
                                    ("2024-09-08", 7.25), ("2024-12-07", 0.0)])
def test_b0_follows_the_known_yearly_cycle(day, b0):
    assert float(c.b0_angle(pd.Timestamp(day, tz="UTC"))) == pytest.approx(b0, abs=0.15)


def test_b0_geometry_reduces_to_the_untilted_one():
    lat, lon, mu = c.heliographic(200.0)
    lat2, lon2, mu2 = c.heliographic_b0(200.0, 0.0)
    assert np.allclose(np.nan_to_num(lat), np.nan_to_num(lat2))
    assert np.allclose(np.nan_to_num(lon), np.nan_to_num(lon2))
    lat3, _, _ = c.heliographic_b0(200.0, 7.0)
    centre = int(round(c.CENTER))
    assert lat3[centre, centre] == pytest.approx(7.0, abs=0.3)  # disk centre sits at B0


def test_grid_metrics_find_a_low_latitude_hole():
    taken = pd.Timestamp("2024-06-06T00:00Z")  # B0 ~ 0
    img = np.full((c.SIZE, c.SIZE), 100.0)
    radius = c.disk_radius(taken)
    yy, xx = np.mgrid[: c.SIZE, : c.SIZE]
    img[np.hypot(xx - c.CENTER, yy - c.CENTER) >= radius] = 0.0
    img[(np.abs(xx - c.CENTER) < 15) & (np.abs(yy - c.CENTER) < 40)] = 10.0  # equatorial hole
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="JPEG", quality=95)
    m = c.grid_metrics(buf.getvalue(), taken)
    assert m["ch_cm_low"] > m["ch_cm_b0"] > 0
    assert m["ch_cm_toward"] == pytest.approx(0.0, abs=0.02)
    assert 0 < m["ch_cm_depth"] < m["ch_cm_b0"]
