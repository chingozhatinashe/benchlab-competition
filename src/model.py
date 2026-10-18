#!/usr/bin/env python3
# Entry point called by run.sh. Forecasting logic lives in the inference
# package; this file only handles the CLI and the output contract.
#
# Usage:  python3 model.py 20261111T060000Z   ->  JSON on stdout

import json
import sys
from datetime import datetime, timedelta, timezone

from inference import FORECAST_HOURS, run_forecast


def predict(t0):
    return run_forecast(t0)


def main():
    if len(sys.argv) != 2:
        print("usage: model.py <YYYYMMDDTHHMMSSZ>", file=sys.stderr)
        return 2

    t0 = datetime.strptime(sys.argv[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    speeds = predict(t0)

    if len(speeds) != FORECAST_HOURS:
        print(
            "FATAL: produced %d values, need %d" % (len(speeds), FORECAST_HOURS),
            file=sys.stderr,
        )
        return 3

    json.dump(
        {
            "t0": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_from": (t0 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_speed_kms": [float(v) for v in speeds],
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
