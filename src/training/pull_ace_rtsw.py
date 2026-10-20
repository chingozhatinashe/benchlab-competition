"""Pull SWPC real-time ACE SWEPAM speed and average it to hourly values.

    python -m training.pull_ace_rtsw                          # the pinned 5-year window
    python -m training.pull_ace_rtsw --start 20250101T000000Z --end 20250201T000000Z

Source: SWPC's daily real-time lists, archived on the NASA SOHO mirror
(YYYYMMDD_ace_swepam_1m.txt). Raw files are cached in src/data/ace_rtsw/raw/.

Output: src/data/ace_rtsw/speed_hourly.csv (Time | status | value | n_valid) and
manifest.json. Each hour is the mean of its valid minutes, labelled by the hour's
start. An hour with no valid minutes is left missing; nothing is imputed.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from connectors.base import STATUS_MISSING, STATUS_OK, HTTPNotFound, http_get_bytes, to_utc
from training.datasets import DATA_DIR, DATA_START, EVAL_END

BASE_URL = "https://soho.nascom.nasa.gov/sdb/ace/daily"
FILE_PATTERN = "{day:%Y%m%d}_ace_swepam_1m.txt"
FILL_SPEED = -9999.9
MINUTES_PER_DAY = 1440
WORKERS = 4
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
COLUMNS = ["year", "month", "day", "hhmm", "mjd", "sod", "status", "density", "speed", "temp"]


def log(msg):
    print(f"[pull_ace_rtsw] {msg}", file=sys.stderr)


def download_day(day):
    """Raw text of one day's file, or None if the archive has no file for that day."""
    try:
        return http_get_bytes(f"{BASE_URL}/{FILE_PATTERN.format(day=day)}").decode("ascii")
    except HTTPNotFound:
        return None


def fetch_day(day, raw_dir):
    """One day's raw text, from the cache when it holds a complete file."""
    path = Path(raw_dir) / FILE_PATTERN.format(day=day)
    if path.exists():
        text = path.read_text()
        if len(parse_minutes(text)) == MINUTES_PER_DAY:
            return text
    text = download_day(day)
    if text is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def parse_minutes(text):
    """1-minute speeds from one file, NaN where the status is bad or the value is fill."""
    rows = [line.split() for line in text.splitlines() if line[:1].isdigit()]
    rows = [r for r in rows if len(r) == len(COLUMNS)]
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC", name="Time"))
    df = pd.DataFrame(rows, columns=COLUMNS)
    stamp = df["year"] + df["month"] + df["day"] + df["hhmm"]
    times = pd.to_datetime(stamp, format="%Y%m%d%H%M", utc=True)
    speed = df["speed"].astype(float)
    valid = (df["status"].astype(int) == 0) & (speed != FILL_SPEED)
    return pd.Series(
        np.where(valid, speed, np.nan),
        index=pd.DatetimeIndex(times, name="Time"),
        name="speed",
    )


def to_hourly(minutes, start, end):
    """Hourly mean of valid minutes over [start, end). No valid minutes gives NaN."""
    hours = pd.date_range(start, end, freq="h", inclusive="left", name="Time")
    grouped = minutes.groupby(minutes.index.floor("h"))
    df = pd.DataFrame({"value": grouped.mean(), "n_valid": grouped.count()}).reindex(hours)
    df["n_valid"] = df["n_valid"].fillna(0).astype(int)
    df["status"] = np.where(df["n_valid"] > 0, STATUS_OK, STATUS_MISSING)
    return df.reset_index()[["Time", "status", "value", "n_valid"]]


def run(start=DATA_START, end=EVAL_END, out=DATA_DIR):
    """Pull [start, end), write the hourly CSV and manifest. Returns the manifest."""
    start, end = to_utc(start).floor("D"), to_utc(end).ceil("D")
    out = Path(out)
    raw_dir = out / "raw"
    days = pd.date_range(start, end, freq="D", inclusive="left")
    print(
        f"Pulling ACE RTSW speed {start:%Y-%m-%d} .. {end:%Y-%m-%d} ({len(days)} days) into {out}"
    )

    parts, missing_days = [], []
    # A few parallel downloads: each file takes 1-4 s from the mirror.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        texts = pool.map(lambda d: fetch_day(d, raw_dir), days)
        for i, (day, text) in enumerate(zip(days, texts, strict=True), 1):
            if text is None:
                missing_days.append(f"{day:%Y-%m-%d}")
            else:
                parts.append(parse_minutes(text))
            if i % 100 == 0 or i == len(days):
                print(f"  {i}/{len(days)} days", flush=True)

    minutes = pd.concat(parts) if parts else parse_minutes("")
    minutes = minutes[~minutes.index.duplicated(keep="first")]
    hourly = to_hourly(minutes, start, end)
    hourly.to_csv(out / "speed_hourly.csv", index=False, date_format=TIME_FORMAT)

    manifest = {
        "source": f"{BASE_URL}/{FILE_PATTERN.replace('{day:%Y%m%d}', 'YYYYMMDD')}",
        "start": start.strftime(TIME_FORMAT),
        "end": end.strftime(TIME_FORMAT),
        "pulled_at": pd.Timestamp.now(tz="UTC").strftime(TIME_FORMAT),
        "hours": len(hourly),
        "missing_hours": int((hourly["status"] == STATUS_MISSING).sum()),
        "missing_days": missing_days,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    share = manifest["missing_hours"] / max(len(hourly), 1)
    print(
        f"  {len(hourly)} hours, {manifest['missing_hours']} missing ({share:.1%}), "
        f"{len(missing_days)} days without a file"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=DATA_START, help="default: the pinned DATA_START")
    parser.add_argument("--end", default=EVAL_END, help="exclusive; default: the pinned EVAL_END")
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    run(args.start, args.end, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
