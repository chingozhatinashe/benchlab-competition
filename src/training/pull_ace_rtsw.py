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

import pandas as pd

# The live connector parses and averages the same way, so training and live data match.
from connectors.ace_archive import BASE_URL, FILE_PATTERN, parse_minutes
from connectors.base import STATUS_MISSING, HTTPNotFound, http_get_bytes, to_utc
from feature_engineering.hourly import to_hourly
from training.datasets import DATA_DIR, DATA_START, EVAL_END

MINUTES_PER_DAY = 1440
WORKERS = 4
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


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
