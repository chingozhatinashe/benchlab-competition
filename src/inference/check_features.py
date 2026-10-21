"""Dry run of the regressor pipeline for one t0: fetch, build, impute, report.

    python -m inference.check_features 20261021T180000Z
    docker run --rm --entrypoint python3 swp-local -m inference.check_features <t0>

Prints a JSON report to stdout: per-source status, rows received, time taken,
megabytes downloaded, the raw regressors and which of them had to be filled.
It does not change the forecast; it shows the inputs work in the live setting.
"""

import json
import math
import sys

from connectors.base import to_utc
from feature_engineering import build, impute
from inference.fetch_inputs import fetch_inputs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    t0 = to_utc(argv[0])
    inputs, report = fetch_inputs(t0)
    raw = build.build_table(inputs, [t0]).iloc[0]
    filled = impute.impute(raw.to_frame().T).iloc[0]
    report["features"] = {k: None if not math.isfinite(v) else round(float(v), 4)
                          for k, v in raw.items()}
    report["imputed"] = [k for k, v in raw.items() if not math.isfinite(v)]
    report["filled_values"] = {k: round(float(filled[k]), 4) for k in report["imputed"]}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
