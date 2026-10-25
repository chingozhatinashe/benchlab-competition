"""Pull SDO/HMI quick-look magnetogram history and reduce each image to a lat/lon grid.

    python -m training.pull_hmi_images --pilot 2024-05          # one month, timed, not saved
    python -m training.pull_hmi_images [--max-seconds 3000]     # 2016-10 .. now; resumable

Source: the HMI browse JPEGs ("HMIB") on sdo.gsfc.nasa.gov/assets/img/browse/YYYY/MM/DD/,
the same directories (and the same SDOConnector) as the AIA 193 images, posted about
30 min after capture, so live runs read the same product as this history.

Cadence: like the coronal-hole images, the newest usable image taken by each 6-hourly
cut-off t0 - build.CH_LATENCY (and no more than 24 h earlier). Each image is reduced by
feature_engineering.magnetogram.reduce and then discarded. Output (dev-only), one file
per month: src/data/features/hmi_grid/YYYY-MM.npz with
    cutoff  int64 ns (UTC)
    taken   int64 ns (UTC), -1 where no usable image was found
    grid    float32 (n, len(STATS), n_lat, n_lon), NaN where no usable image was found
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from connectors.sdo import SDOConnector
from feature_engineering import build
from feature_engineering import magnetogram as mg
from training.datasets import EVAL_END
from training.pull_features import AIA_START, OUT, t0_grid

SIZE = 512  # the 10-deg grid agrees with the 1024 images (r >= 0.98, May 2024) at half the time
CHANNEL = "HMIB"
GRID_DIR = OUT / "hmi_grid"
WORKERS = 8


def log(msg):
    print(msg, flush=True)


def pull_month(cutoffs, sdo):
    """(cutoff, taken, grid) arrays for one month's cut-offs."""
    reduced = {}

    def accept(jpeg, taken):
        g = mg.reduce(jpeg, taken)
        if g is not None:
            reduced[taken] = g
        return g is not None

    images = sdo.latest(cutoffs, accept=accept)
    grid = np.full((len(cutoffs), *mg.grid_shape()), np.nan, dtype=np.float32)
    taken = np.full(len(cutoffs), -1, dtype=np.int64)
    for i, c in enumerate(cutoffs):
        t, _ = images[c]
        if t is not None:
            taken[i] = t.value
            grid[i] = reduced[t]
    return pd.DatetimeIndex(cutoffs).asi8, taken, grid


def load_grid(directory=GRID_DIR):
    """(cutoffs DatetimeIndex, taken DatetimeIndex with NaT, grid) over all pulled months."""
    parts = [np.load(p) for p in sorted(directory.glob("*.npz"))]
    if not parts:
        raise FileNotFoundError(f"no HMI grids in {directory}; run training.pull_hmi_images")
    cut = np.concatenate([p["cutoff"] for p in parts])
    taken = np.concatenate([p["taken"] for p in parts])
    grid = np.concatenate([p["grid"] for p in parts])
    order = np.argsort(cut, kind="stable")
    taken_idx = pd.DatetimeIndex(np.where(taken[order] < 0, np.datetime64("NaT"),
                                          taken[order].astype("datetime64[ns]")), tz="UTC")
    return pd.DatetimeIndex(cut[order], tz="UTC"), taken_idx, grid[order]


def load_signals(directory=GRID_DIR):
    """magnetogram.SIGNALS per pulled cut-off (NaN rows where no usable image was found)."""
    cut, _, grid = load_grid(directory)
    frame = pd.DataFrame(mg.signals(grid), index=pd.DatetimeIndex(cut, name="cutoff"))
    return frame[~frame.index.duplicated(keep="last")]


def run(max_seconds=None, pilot=None):
    started = time.perf_counter()
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    end = min(pd.Timestamp.now(tz="UTC").floor("D"), EVAL_END)
    cutoffs = t0_grid(AIA_START, end) - build.CH_LATENCY
    months = sorted({f"{c:%Y-%m}" for c in cutoffs})
    if pilot:
        months = [pilot]
    sdo = SDOConnector(size=SIZE, channel=CHANNEL, budget=1800, workers=WORKERS)
    for key in months:
        path = GRID_DIR / f"{key}.npz"
        if path.exists() and not pilot:
            continue
        batch = [c for c in cutoffs if f"{c:%Y-%m}" == key]
        t = time.perf_counter()
        cut, taken, grid = pull_month(batch, sdo)
        secs = time.perf_counter() - t
        found = int((taken >= 0).sum())
        if pilot:
            log(f"pilot {key}: {found}/{len(batch)} usable images, {secs:.0f} s; "
                f"{len(months_all := sorted({f'{c:%Y-%m}' for c in cutoffs}))} months -> "
                f"~{len(months_all) * secs / 3600:.1f} h")
            return cut, taken, grid
        np.savez_compressed(path, cutoff=cut, taken=taken, grid=grid)
        log(f"{key}: {found}/{len(batch)} usable images ({secs:.0f} s)")
        if max_seconds and time.perf_counter() - started > max_seconds:
            log("time limit reached; run again to continue")
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--pilot", default=None, help="one month, YYYY-MM, timed, not saved")
    args = parser.parse_args(argv)
    run(args.max_seconds, args.pilot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
