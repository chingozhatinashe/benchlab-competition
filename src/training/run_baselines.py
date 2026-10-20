"""Score every speed-only baseline on the eval year.

python -m training.run_baselines
"""

import sys
import time

import pandas as pd

from training.baselines import ALL
from training.evaluate import evaluate


def run(models=ALL):
    rows = []
    for name, cls in models.items():
        started = time.perf_counter()
        result = evaluate(cls())
        result["seconds"] = round(time.perf_counter() - started, 1)
        rows.append({"model": name, **result})
        print(f"  {name:<20} TIMSE {result['timse']:.4g}  RMSE {result['rmse']:.1f}", flush=True)
    return pd.DataFrame(rows).sort_values("timse").reset_index(drop=True)


def main():
    table = run()
    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 120):
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
