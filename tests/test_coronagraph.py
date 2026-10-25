"""feature_engineering.coronagraph, connectors.lasco and the windowed coronagraph signals."""

import io

import numpy as np
import pandas as pd
import pytest

from feature_engineering import build, windows
from feature_engineering import coronagraph as cg

T0 = pd.Timestamp("2026-03-15T12:00Z")
H = pd.Timedelta(hours=1)


def jpeg(img):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def corona(east_boost=0.0):
    """A synthetic C2 frame: brightness falling with radius, optionally brighter due east."""
    yy, xx = np.mgrid[: cg.SIZE, : cg.SIZE]
    r = np.hypot(xx - cg.CENTER, yy - cg.CENTER)
    img = 200.0 * np.exp(-(r - 90) / 80.0)
    img[r < 95] = 30.0
    east = (xx < cg.CENTER - 100) & (np.abs(yy - cg.CENTER) < 30)
    img[east] += east_boost
    return img


def test_sectors_follow_position_angle_from_north_through_east():
    plain = cg.cells(jpeg(corona()), "c2")
    bright = cg.cells(jpeg(corona(east_boost=60.0)), "c2")
    gain = bright - plain
    # the patch straddles PA 90 (east = image left): the sectors either side of it
    east = [int(90 // (360 / cg.N_SECTORS)) - 1, int(90 // (360 / cg.N_SECTORS))]
    assert set(np.flatnonzero(gain[1] > 5)) == set(east) and gain[1, east].sum() > 20
    assert cg.cells(jpeg(np.zeros((cg.SIZE, cg.SIZE))), "c2") is None  # blank frame


def test_excess_is_zero_for_a_steady_corona_and_flags_a_halo():
    times = pd.date_range(T0 - 10 * H, T0, freq="h")
    base = np.tile(np.linspace(50, 100, 24), (len(times), 1))
    tab = pd.DataFrame(base, index=times, columns=cg.cell_columns())
    tab.iloc[-1] *= 1.5  # everything brightens at once: a halo
    s = cg.summaries(cg.excess(tab))
    assert s["max"].iloc[-2] == pytest.approx(0.0) and s["share"].iloc[-2] == 0.0
    assert s["share"].iloc[-1] == 1.0 and s["max"].iloc[-1] == pytest.approx(0.5)
    assert s["max"].iloc[:2].isna().all()  # not enough background yet


def test_excess_stays_finite_over_a_black_background():
    times = pd.date_range(T0 - 10 * H, T0, freq="h")
    tab = pd.DataFrame(np.zeros((len(times), 24)), index=times, columns=cg.cell_columns())
    tab.iloc[-1] = 40.0  # a blanked stretch, then a normal frame
    e = cg.excess(tab)
    assert np.isfinite(e.iloc[2:]).all().all()
    assert e.iloc[-1].max() == pytest.approx(40.0 / cg.MIN_MEDIAN - 1)


def test_select_times_keeps_the_first_image_of_each_cadence_block():
    t = pd.DatetimeIndex(["2026-03-15T00:12Z", "2026-03-15T00:36Z", "2026-03-15T01:05Z",
                          "2026-03-15T03:59Z"])
    assert list(cg.select_times(t, "c2")) == [t[0], t[2], t[3]]
    assert list(cg.select_times(t, "c3")) == [t[0], t[3]]


def test_listing_parser_reads_capture_times():
    from connectors.lasco import image_url, parse_listing

    day = pd.Timestamp("2024-05-09", tz="UTC")
    html = ('<a href="20240509_0012_c2_512.jpg">x</a><a href="20240509_0012_c2_1024.jpg">'
            '<a href="20240509_0936_c2_512.jpg">')
    times = parse_listing(html, "c2", day)
    assert list(times) == [pd.Timestamp("2024-05-09T00:12Z"), pd.Timestamp("2024-05-09T09:36Z")]
    assert image_url("c3", times[1]).endswith("/2024/c3/20240509/20240509_0936_c3_512.jpg")


def signal_inputs():
    times = pd.date_range(T0 - 80 * H, T0 + 5 * H, freq="h") + pd.Timedelta(minutes=12)
    frame = pd.DataFrame(index=times, columns=list(cg.SIGNALS), dtype=float)
    for i, s in enumerate(cg.SIGNALS):
        frame[s] = np.linspace(0, 1, len(times)) + i
    frame.loc[frame.index.hour % 2 == 1, [s for s in cg.SIGNALS if s.startswith("c3")]] = np.nan
    return build.Inputs(coronagraph=frame)


def test_coronagraph_windows_use_no_image_after_the_cutoff():
    specs = windows.coronagraph_candidates()
    inputs = signal_inputs()
    a = windows.window_table(inputs, [T0], specs)
    late = inputs.coronagraph.copy()
    late.loc[late.index > T0 - cg.LATENCY] = 1e6
    b = windows.window_table(build.Inputs(coronagraph=late), [T0], specs)
    pd.testing.assert_frame_equal(a, b)
    assert a.notna().all(axis=None)
    last = inputs.coronagraph.loc[inputs.coronagraph.index <= T0 - cg.LATENCY, "c2max"].iloc[-1]
    assert a.loc[T0, "c2max_max_6h"] == pytest.approx(last)  # the signal rises: max = latest


def test_coronagraph_windows_need_half_the_images_and_size_the_fetch():
    inputs = signal_inputs()
    sparse = inputs.coronagraph.iloc[::5]
    row = windows.window_table(build.Inputs(coronagraph=sparse), [T0],
                               [windows.WindowSpec("c2max", "max", 6)]).iloc[0]
    assert np.isnan(row.iloc[0])  # 1-2 images in 6 h < 3
    assert np.isnan(windows.window_table(build.Inputs(), [T0], windows.coronagraph_candidates()[:1])
                    .iloc[0, 0])
    assert windows.lasco_hours(windows.coronagraph_candidates()) == 72 + 6 + 1
    assert windows.lasco_hours(windows.candidates()) == 0
