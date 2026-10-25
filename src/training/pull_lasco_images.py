"""Pull SOHO/LASCO C2/C3 browse-image history and reduce each image to sector brightness.

    python -m training.pull_lasco_images --pilot 2024-05          # one month, timed
    python -m training.pull_lasco_images [--max-seconds 3000]     # 2016-10 .. now; resumable

Source: https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/YYYY/{c2,c3}/YYYYMMDD/,
the 512-pixel browse JPEGs. The same directories fill in near real time (about 15 min
after the image), so live runs read the same product as this history.

Cadence: C2 the first image of every hour, C3 the first image of every even hour. Each
image is reduced to its (2 rings x 12 sectors) median grey levels by
feature_engineering.coronagraph.cells and then discarded. Output (dev-only):
src/data/features/lasco_cells.csv (time, instrument, file, 24 cells; NaN cells for an
unusable frame), months done in lasco_cells_manifest.json.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from feature_engineering import coronagraph as cg
from training.datasets import EVAL_END
from training.pull_features import OUT

BASE = "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed"
START = pd.Timestamp("2016-10-01", tz="UTC")
CELLS = OUT / "lasco_cells.csv"
MANIFEST = OUT / "lasco_cells_manifest.json"
WORKERS = 8
TIMEOUT = 60


def log(msg):
    print(msg, flush=True)


def _get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(2 * (i + 1))
    return None


def day_images(instrument, day):
    """[(time, url)] of the 512-pixel images in a day's directory (empty if none)."""
    d = f"{day:%Y%m%d}"
    html = _get(f"{BASE}/{day:%Y}/{instrument}/{d}/")
    if html is None:
        return []
    names = sorted(set(re.findall(rf"({d}_(\d{{4}})_{instrument}_512\.jpg)", html.decode())))
    return [(pd.Timestamp(f"{d}T{hhmm}", tz="UTC"), f"{BASE}/{day:%Y}/{instrument}/{d}/{name}")
            for name, hhmm in names]


def select(images, instrument):
    """The images at the cadence live runs use (coronagraph.select_times)."""
    urls = dict(images)
    return [(t, urls[t]) for t in cg.select_times(list(urls), instrument)]


def _row(args):
    instrument, t, url = args
    jpeg = _get(url)
    try:
        v = cg.cells(jpeg, instrument) if jpeg is not None else None
    except Exception:  # a corrupt frame must not stop the month
        v = None
    vals = v.ravel() if v is not None else np.full(2 * cg.N_SECTORS, np.nan)
    return {"time": t, "instrument": instrument, "file": url.rsplit("/", 1)[1],
            "fetched": jpeg is not None, **dict(zip(cg.cell_columns(), vals, strict=True))}


def pull_month(month, pool):
    """Rows for every selected image of one month (both instruments)."""
    days = pd.date_range(month, month + pd.offsets.MonthEnd(0), freq="D")
    days = days[days < EVAL_END + pd.Timedelta(days=7)]
    todo = []
    listings = list(pool.map(lambda a: (a[0], day_images(*a)),
                             [(i, d) for i in cg.INSTRUMENTS for d in days]))
    for instrument, images in listings:
        todo += [(instrument, t, url) for t, url in select(images, instrument)]
    return list(pool.map(_row, todo))


def load_cells():
    df = pd.read_csv(CELLS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates(["instrument", "time"], keep="last").sort_values("time")


def months_done():
    return set(json.loads(MANIFEST.read_text())["months"]) if MANIFEST.exists() else set()


def run(max_seconds=None, pilot=None):
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    done = months_done()
    end = min(pd.Timestamp.now(tz="UTC").floor("D"), EVAL_END + pd.Timedelta(days=7))
    months = ([pd.Timestamp(pilot, tz="UTC")] if pilot else
              list(pd.date_range(START, end, freq="MS")))
    with ThreadPoolExecutor(WORKERS) as pool:
        for month in months:
            key = f"{month:%Y-%m}"
            if key in done and not pilot:
                continue
            t = time.perf_counter()
            rows = pull_month(month, pool)
            frame = pd.DataFrame(rows)
            if pilot:
                n = frame.groupby("instrument").size().to_dict() if len(frame) else {}
                bad = int(frame[cg.cell_columns()[0]].isna().sum()) if len(frame) else 0
                secs = time.perf_counter() - t
                n_months = len(pd.date_range(START, end, freq="MS"))
                log(f"pilot {key}: {n} images, {bad} unusable, {secs:.0f} s; "
                    f"{n_months} months -> ~{n_months * secs / 3600:.1f} h")
                return frame
            if len(frame):
                frame.to_csv(CELLS, mode="a", header=not CELLS.exists(), index=False,
                             date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.2f")
            done.add(key)
            MANIFEST.write_text(json.dumps({"months": sorted(done), "source": BASE,
                                            "cadence_hours": cg.CADENCE_HOURS}, indent=1))
            per = frame.groupby("instrument").size().to_dict() if len(frame) else {}
            bad = int(frame[cg.cell_columns()[0]].isna().sum()) if len(frame) else 0
            log(f"{key}: {per} images, {bad} unusable ({time.perf_counter() - t:.0f} s)")
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
