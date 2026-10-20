"""Pull OMNI hourly solar wind history from CDAWeb into src/data/omni_hourly/.

    python -m training.pull_omni                      # last 2 years, all variables
    python -m training.pull_omni --vars V N --start 20250101T000000Z

Writes one CSV per variable (Time | status | value) plus manifest.json. KP is
stored as OMNI gives it, Kp x 10 (e.g. 37 means 3.7).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from connectors.base import http_get_json, to_utc
from connectors.cdaweb import BASE_URL, CDAWebConnector

DATASET = "OMNI2_H0_MRG1HR"
VARIABLES = ["V", "N", "T", "F", "BZ_GSM", "KP", "F10_INDEX"]
DEFAULT_SPAN = pd.Timedelta(days=730)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "omni_hourly"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def inventory_end(dataset=DATASET):
    """Last available time of a CDAWeb dataset, floored to the hour."""
    payload = http_get_json(f"{BASE_URL}/{dataset}/inventory")
    intervals = payload["InventoryDescription"][0]["TimeInterval"]
    return to_utc(intervals[-1]["End"]).floor("h")


def pull_variable(var, start, end):
    conn = CDAWebConnector(dataset=DATASET, variable=var, request_variable=f"{var}1800")
    return conn.fetch(start, end)


def summarize(df):
    return {
        "rows": len(df),
        "status": {k: int(v) for k, v in df["status"].value_counts().items()},
        "first": df["Time"].min().strftime(TIME_FORMAT),
        "last": df["Time"].max().strftime(TIME_FORMAT),
    }


def run(start=None, end=None, variables=VARIABLES, out=DEFAULT_OUT):
    """Pull each variable and write CSVs. Returns the list of variables that failed."""
    end = to_utc(end) if end is not None else inventory_end()
    start = to_utc(start) if start is not None else end - DEFAULT_SPAN
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Pulling {DATASET} {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC into {out}")

    manifest = {
        "dataset": DATASET,
        "start": start.strftime(TIME_FORMAT),
        "end": end.strftime(TIME_FORMAT),
        "pulled_at": pd.Timestamp.now(tz="UTC").strftime(TIME_FORMAT),
        "variables": {},
    }
    failed = []
    for var in variables:
        df = pull_variable(var, start, end)
        if df.empty:
            # The connector returns an empty frame on errors; keep any older file.
            print(f"  {var:<10} FAILED: no rows returned", file=sys.stderr)
            failed.append(var)
            continue
        df.to_csv(out / f"{var}.csv", index=False, date_format=TIME_FORMAT)
        info = summarize(df)
        manifest["variables"][var] = info
        print(f"  {var:<10} {info['rows']:>6} rows  {info['status']}")

    manifest["failed"] = failed
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", help="default: end minus 730 days")
    parser.add_argument("--end", help="default: last hour available on CDAWeb")
    parser.add_argument("--vars", nargs="+", default=VARIABLES, choices=VARIABLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    failed = run(args.start, args.end, args.vars, args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
