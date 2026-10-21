"""Fill the regressors that are still unknown after the builders' carry rules.

Order of the gap handling (the first three steps happen in feature_engineering.build):
1. out-of-range values -> NaN (insitu.VALID, coronal_holes.usable);
2. short gaps carried forward from the past only (registry `carry`), never
   interpolated, since interpolation would use later values;
3. window statistics need at least half their hours;
4. here: anything still NaN gets the training-years median and a
   `<name>_missing` flag of 1, so a model can tell a real value from a fill.

Counts are not special-cased: a flare count of 0 from a working feed is a value,
while an unreachable feed gives NaN and so the median plus a flag.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import registry

ASSET = Path(__file__).resolve().parents[1] / "assets" / "feature_climatology.json"
FORMAT_VERSION = 1


def fit_climatology(table):
    """Median of each regressor over the rows given (the training years)."""
    med = table[registry.NAMES].median(skipna=True)
    return {name: float(med[name]) if np.isfinite(med[name]) else registry.BY_NAME[name].default
            for name in registry.NAMES}


def impute(table, climatology=None):
    """Regressors with gaps filled, followed by one `<name>_missing` flag per regressor."""
    climatology = climatology or load_climatology()
    values = table[registry.NAMES].astype(float)
    missing = values.isna()
    filled = values.fillna(pd.Series(climatology))
    flags = missing.astype(float).add_suffix("_missing")
    return pd.concat([filled, flags], axis=1)


def save_climatology(medians, meta, path=ASSET):
    payload = {"format_version": FORMAT_VERSION, **meta,
               "medians": {k: round(v, 6) for k, v in medians.items()}}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_climatology(path=ASSET):
    """Fitted medians; the registry defaults for any the asset lacks (or if it is unreadable)."""
    out = {f.name: f.default for f in registry.FEATURES}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") == FORMAT_VERSION:
            out.update({k: float(v) for k, v in payload["medians"].items() if k in out})
    except Exception:
        pass
    return out
