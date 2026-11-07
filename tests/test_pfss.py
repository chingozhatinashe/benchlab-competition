import gzip
import io

import numpy as np
import pandas as pd
import pytest

from connectors import gong
from feature_engineering import pfss

LISTING = """
<tr><td><a href="mrzqs261106t1214c2317_113.fits.gz">mrzqs261106t1214c2317_113.fits.gz</a></td>
<td align="right">2026-11-06 05:46  </td><td align="right">237K</td></tr>
<tr><td><a href="mrzqs261106t1314c2317_112.fits.gz">mrzqs261106t1314c2317_112.fits.gz</a></td>
<td align="right">2026-11-06 06:46  </td><td align="right">237K</td></tr>
"""


def listing():
    # each row on one line, as the server writes them
    return gong.parse_listing(LISTING.replace("</a></td>\n", "</a></td>"))


def test_listing_times_are_utc():
    df = listing()
    assert list(df["name"]) == ["mrzqs261106t1214c2317_113.fits.gz",
                                "mrzqs261106t1314c2317_112.fits.gz"]
    assert df["observed"].iloc[0] == pd.Timestamp("2026-11-06T12:14Z")
    assert df["modified"].iloc[0] == pd.Timestamp("2026-11-06T12:46Z")  # MST + 7 h
    assert df["rotation"].iloc[0] == 2317 and df["longitude"].iloc[0] == 113


def test_map_url():
    assert gong.map_url("mrzqs261106t1214c2317_113.fits.gz") == (
        "https://gong2.nso.edu/oQR/zqs/202611/mrzqs261106/mrzqs261106t1214c2317_113.fits.gz")


@pytest.mark.parametrize("t0, expected", [
    ("2026-11-06T12:56Z", []),  # posted 12:46:xx, margin 10 min: not yet
    ("2026-11-06T12:57Z", ["mrzqs261106t1214c2317_113.fits.gz"]),
    ("2026-11-06T18:00Z", ["mrzqs261106t1314c2317_112.fits.gz",
                           "mrzqs261106t1214c2317_113.fits.gz"]),
    ("2026-11-09T13:30Z", []),  # observed more than 72 h before the cut-off
])
def test_only_maps_posted_by_t0_are_usable(t0, expected):
    assert list(gong.usable(listing(), pd.Timestamp(t0))["name"]) == expected


def test_history_choice_matches_the_live_rule():
    from training.pull_gong import choose

    df = listing()
    t0s = pd.DatetimeIndex(["2026-11-06T12:56Z", "2026-11-06T12:57Z", "2026-11-06T18:00Z"])
    chosen = choose(df, t0s)
    for t0 in t0s:
        assert chosen[t0] == list(gong.usable(df, t0)["name"][:3])


def test_sheet_points_and_angles():
    lat, lon = np.meshgrid(np.arange(-85, 90, 10.0), np.arange(5, 360, 10.0), indexing="ij")
    pts = pfss.sheet_points(lat, lon, np.sign(lat))  # boundary on the equator
    assert np.allclose(pts[:, 2], 0, atol=1e-9)
    d = pfss.nearest_angle(pfss._unit(np.array([20.0, -5.0]), np.array([5.0, 95.0])), pts)
    assert d == pytest.approx([20.0, 5.0], abs=0.01)


def dipole_map_bytes(t="2024-03-20T12:00"):
    """A GONG-like CEA Carrington map of a dipole field, Br = 5 G sin(lat), as .fits.gz."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from sunpy.map import header_helper

    shape = (180, 360)
    ref = SkyCoord(0 * u.deg, 0 * u.deg, frame="heliographic_carrington", obstime=t,
                   observer="earth")
    header = header_helper.make_heliographic_header(ref.obstime, ref.observer, shape,
                                                    frame="carrington", projection_code="CEA")
    s = np.linspace(-1, 1, shape[0] + 1)
    s = (s[:-1] + s[1:]) / 2
    data = np.repeat((5.0 * s)[:, None], shape[1], axis=1)
    buf = io.BytesIO()
    fits.PrimaryHDU(data, fits.Header(header)).writeto(buf)
    return gzip.compress(buf.getvalue())


def test_dipole_ring():
    pytest.importorskip("sunkit_magex")
    ring = pfss.reduce(dipole_map_bytes(), pd.Timestamp("2024-03-20T12:00Z"))
    assert ring is not None
    v = dict(zip(pfss.RING_VARS, ring.ring, strict=True))
    s = dict(zip(pfss.SCALARS, ring.scalars, strict=True))
    assert ring.b0 == pytest.approx(-7.0, abs=0.3)
    # the current sheet is the equator, |B0| away from every ring point
    assert s["hcs_max_lat"] < 3
    assert np.nanmedian(v["d_hcs"]) == pytest.approx(abs(ring.b0), abs=1.5)
    # open field only near the poles; the ring maps down to the southern polar hole (B0 < 0)
    assert 0.02 < s["open_area"] < 0.5
    assert np.nanmin(v["foot_lat"]) > 35
    assert np.all(v["foot_pol"][np.isfinite(v["foot_pol"])] == -1)
    assert np.isfinite(v["log_fs"]).mean() > 0.9
    # the ring is uniform in longitude
    assert np.nanstd(v["log_fs"]) < 0.1


def test_bad_bytes_give_none():
    pytest.importorskip("sunkit_magex")
    assert pfss.reduce(b"not a fits file") is None


def ring_table(t0s, seed=0):
    rng = np.random.default_rng(seed)
    n = len(t0s)
    return pfss.RingTable(pd.DatetimeIndex(t0s), pd.DatetimeIndex(t0s) - pd.Timedelta(hours=1),
                          rng.normal(size=(n, *pfss.ring_shape())).astype(np.float32),
                          rng.normal(size=(n, len(pfss.SCALARS))).astype(np.float32))


def test_pfss_windows_use_only_rings_up_to_t0():
    from feature_engineering import build, windows

    t0s = pd.date_range("2024-03-01", "2024-03-10", freq="6h", tz="UTC")
    table = ring_table(t0s)
    specs = windows.pfss_candidates()
    t0 = t0s[20]
    before = windows.window_table(build.Inputs(pfss=table), [t0], specs)
    later = t0s > t0
    scrambled = ring_table(t0s, seed=1)
    scrambled.ring[~later] = table.ring[~later]
    scrambled.scalars[~later] = table.scalars[~later]
    after = windows.window_table(build.Inputs(pfss=scrambled), [t0], specs)
    pd.testing.assert_frame_equal(before, after)
    assert before.notna().all().all()


def test_pfss_window_reads_the_longitude_facing_earth():
    from feature_engineering import build, windows

    t0 = pd.Timestamp("2024-03-01T12:00Z")
    lon = np.arange(pfss.N_LON, dtype=np.float32)
    table = ring_table([t0])
    table.ring[0, pfss.RING_VARS.index("d_hcs")] = lon  # value = its own longitude
    specs = [windows.WindowSpec("pfss_d_hcs", "at", h) for h in (0, 24)]
    row = windows.window_table(build.Inputs(pfss=table), [t0], specs).iloc[0]
    expect = pfss.earth_longitude([t0, t0 - pd.Timedelta(hours=24)])
    assert row.to_numpy() == pytest.approx(expect, abs=1e-3)
    # the Sun rotates about 13.2 deg a day under Earth, so Carrington longitude falls
    assert (expect[1] - expect[0]) % 360 == pytest.approx(13.2, abs=0.3)


def test_no_rings_give_nan():
    from feature_engineering import build, windows

    specs = windows.pfss_candidates()[:3]
    out = windows.window_table(build.Inputs(), [pd.Timestamp("2024-03-01T12:00Z")], specs)
    assert out.isna().all().all()


def test_gong_falls_back_to_a_mirror(monkeypatch):
    calls = []

    def fake_get(url, timeout, retries):
        calls.append(url)
        if url.startswith(gong.MIRRORS[0]):
            raise TimeoutError("timed out")
        return b"ok"

    monkeypatch.setattr(gong, "http_get_bytes", fake_get)
    c = gong.GONGConnector()
    assert c.download("mrzqs261106t1214c2317_113.fits.gz") == b"ok"
    assert calls[1] == gong.MIRRORS[1] + "/202611/mrzqs261106/mrzqs261106t1214c2317_113.fits.gz"
    c.download("mrzqs261106t1314c2317_112.fits.gz")
    assert calls[2].startswith(gong.MIRRORS[1])  # the working mirror first from then on


SWPC_LISTING = (
    '<a href="mrzqs261107t0514c2317_103.fits.gz">mrzqs261107t0514c2317_103.fits.gz</a>'
    ' 2026-11-07 05:50  237K\n'
    '<a href="mrzqs261107t0614c2317_103.fits.gz">mrzqs261107t0614c2317_103.fits.gz</a>'
    ' 2026-11-07 06:50  237K\n')


def test_live_runs_read_swpc_first(monkeypatch):
    urls = []

    def fake_get(url, timeout, retries):
        urls.append(url)
        return SWPC_LISTING.encode() if url == gong.SWPC_URL else b"map"

    monkeypatch.setattr(gong, "http_get_bytes", fake_get)
    row, data = gong.GONGConnector().latest(pd.Timestamp("2026-11-07T07:00Z"))
    # SWPC lists UTC: the 06:14 map was posted 06:50, after t0 - 10 min
    assert row["name"] == "mrzqs261107t0514c2317_103.fits.gz" and row["source"] == "swpc"
    assert urls == [gong.SWPC_URL, gong.SWPC_URL + row["name"]]


def test_nso_is_used_when_swpc_has_nothing(monkeypatch):
    def fake_get(url, timeout, retries):
        if url == gong.SWPC_URL:
            raise TimeoutError("timed out")
        if url.endswith("/"):
            return LISTING.replace("</a></td>\n", "</a></td>").encode() if "261106" in url \
                else b""
        return b"map"

    monkeypatch.setattr(gong, "http_get_bytes", fake_get)
    row, _ = gong.GONGConnector().latest(pd.Timestamp("2026-11-06T18:00Z"))
    assert row["source"] == "nso" and row["name"] == "mrzqs261106t1314c2317_112.fits.gz"


def test_ring_signals_take_negative_lead_lags():
    from models import linear_model as lm

    assert lm.lead_lags(24, "ch_cm").min() == 0  # images: never after t0
    lags = lm.lead_lags(24, "pfss_d_hcs")
    assert lags[0] == 24 and lags[-1] == -48  # lead 72 reads the longitude facing Earth at t0+48
    cfg = lm.Config(features=("pfss_d_hcs_lead_24h",))
    assert "pfss_d_hcs_at_-48h" in cfg.input_features
