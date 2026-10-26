"""Pull the official ground truth: SWPC's hourly ACE/SWEPAM files (2016-10 -> now).

    python -m training.pull_ace_1h            # every month; past months are cached

Source (the organisers' spec, docs/timse_implementation.pdf, section 2):
sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/YYYYMM_ace_swepam_1h.txt, Bulk Speed,
parsed by training.competition_score.parse_monthly (status != 0 or fill -> missing).
Raw files are cached in src/data/ace_1h/raw/; the current and previous month are always
fetched again (they are still being written). Output: src/data/ace_1h/speed_hourly.csv
(time | speed | speed_any_status), indexed by the hour's start, UTC: `speed` is the official
truth (status 0 only); `speed_any_status` keeps every non-fill speed, for sensitivity checks.
Dev-only.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from connectors.base import HTTPNotFound, http_get_bytes
from training import competition_score as cs
from training.datasets import DATA_DIR, DATA_START

OUT_DIR = DATA_DIR.parent / "ace_1h"
RAW_DIR = OUT_DIR / "raw"
CSV = OUT_DIR / "speed_hourly.csv"
WORKERS = 4


def log(msg):
    print(f"[pull_ace_1h] {msg}", file=sys.stderr, flush=True)


def months(start=DATA_START, end=None):
    end = end or pd.Timestamp.now(tz="UTC")
    return pd.date_range(start.tz_convert(None).to_period("M").to_timestamp(),
                         end.tz_convert(None), freq="MS")


def fetch(month, refresh):
    path = RAW_DIR / f"{month:%Y%m}_ace_swepam_1h.txt"
    if path.exists() and not refresh:
        return path.read_text(encoding="ascii")
    try:
        text = http_get_bytes(cs.MONTHLY_URL.format(month=month)).decode("ascii")
    except HTTPNotFound:
        log(f"{month:%Y-%m}: not found")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii", newline="\n")
    return text


def run(start=DATA_START):
    all_months = months(start)
    recent = set(all_months[-2:])
    with ThreadPoolExecutor(WORKERS) as pool:
        texts = list(pool.map(lambda m: fetch(m, m in recent), all_months))
    texts = [t for t in texts if t]

    def series(any_status):
        s = pd.concat([cs.parse_monthly(t, any_status) for t in texts])
        return s[~s.index.duplicated(keep="last")].sort_index()

    speed, loose = series(False), series(True)
    full = pd.date_range(loose.index[0], loose.index[-1], freq="h", tz="UTC")
    frame = pd.DataFrame({"speed": speed.reindex(full), "speed_any_status": loose.reindex(full)})
    frame.index.name = "time"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CSV, date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.1f")
    log(f"{len(all_months)} months, {len(frame)} hours, {frame.speed.notna().mean():.1%} valid "
        f"({frame.speed_any_status.notna().mean():.1%} with any status; "
        f"{full[0]:%Y-%m-%d} .. {full[-1]:%Y-%m-%d %H:%M}) -> {CSV}")
    return frame


def load(path=CSV, column="speed"):
    """The official hourly speed (NaN where missing), indexed by the hour's start, UTC.
    column="speed_any_status" gives the sensitivity series."""
    f = pd.read_csv(path, index_col=0)
    f.index = pd.to_datetime(f.index, utc=True)
    return f[column].astype(float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
