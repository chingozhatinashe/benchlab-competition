"""Pull the SOHO/LASCO CME catalogue (CDAW) and estimate Earth arrival of halo CMEs.

    python -m training.pull_lasco_cme                                  # 2025-01 .. latest month
    python -m training.pull_lasco_cme --start 2021-10 --end 2026-07

Source: https://cdaw.gsfc.nasa.gov/CME_list/, one text file per month
(UNIVERSAL_ver2/text_ver/univYYYY_MM.txt). Raw files are cached in
src/data/lasco_cme/raw/. Output: src/data/lasco_cme/cmes.csv and manifest.json.

Dev-only. The catalogue is compiled by hand and published months after the
events, so it was not available at forecast time and must never feed a live
forecast. Use it to label and analyse past periods.

The catalogue records CMEs leaving the Sun, not arrivals at L1. For halo and
partial-halo CMEs (width >= 120 deg, the candidates for hitting Earth), an
arrival time at 1 AU is estimated with the empirical CME arrival (ESA) model of
Gopalswamy et al. (2001, JGR 106, 29207): constant acceleration
a = 2.193 - 0.0054 u (m/s^2, u = plane-of-sky linear speed in km/s) out to
0.76 AU, then constant speed. It predicts the arrival of the CME itself; the
shock ahead of it usually comes several hours earlier. Typical errors are
+-10-20 h. A halo can also be a back-side event, which the catalogue can't tell
apart, so a predicted arrival is a candidate, not a confirmed hit.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from connectors.base import HTTPNotFound, http_get_bytes
from training.datasets import DATA_DIR

BASE_URL = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver"
FILE_PATTERN = "univ{month:%Y_%m}.txt"
OUT_DIR = DATA_DIR.parent / "lasco_cme"
AU_KM = 1.495978707e8
ACCEL_STOP_AU = 0.76
PARTIAL_HALO_WIDTH = 120

COLUMNS = [
    "time", "central_pa", "width", "linear_speed", "speed_initial", "speed_final",
    "speed_20r", "accel", "mass", "kinetic_energy", "mpa", "remarks",
]
NUMERIC = COLUMNS[1:-1]
LINE = re.compile(r"^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s")


def log(msg):
    print(f"[pull_lasco_cme] {msg}", file=sys.stderr)


def _number(token):
    """'9.0*' -> 9.0; '-------' or '----' -> NaN. A trailing '*' marks an uncertain value."""
    token = token.rstrip("*")
    try:
        return float(token)
    except ValueError:
        return np.nan


def parse_month(text):
    """One monthly catalogue file as a DataFrame (one row per CME)."""
    rows = []
    for line in text.splitlines():
        if not LINE.match(line):
            continue
        tokens = line.split()
        if len(tokens) < 12:
            continue
        date, time, pa, *values = tokens[:12]
        remarks = " ".join(tokens[12:])
        row = {"time": pd.Timestamp(f"{date.replace('/', '-')}T{time}Z"),
               "central_pa": np.nan if pa == "Halo" else _number(pa),
               "remarks": remarks}
        row.update({name: _number(v) for name, v in zip(NUMERIC[1:], values, strict=True)})
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["halo"] = df["width"] >= 360
    df["partial_halo"] = (df["width"] >= PARTIAL_HALO_WIDTH) & ~df["halo"]
    remarks = df["remarks"].str.lower()
    df["quality"] = np.select(
        [remarks.str.contains("very poor"), remarks.str.contains("poor")],
        ["very poor", "poor"], "ok",
    )
    df["only_c2"] = remarks.str.contains("only c2")
    return df


def arrival_hours(speed_kms):
    """CME travel time (h) to 1 AU from the Gopalswamy et al. (2001) ESA model; NaN if unknown."""
    u = np.asarray(speed_kms, dtype=float)
    a = (2.193 - 0.0054 * u) / 1000.0  # km/s^2
    d1 = ACCEL_STOP_AU * AU_KM
    v1 = np.sqrt(np.maximum(u**2 + 2 * a * d1, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(np.abs(a) > 1e-12, (v1 - u) / a, d1 / np.maximum(u, 1.0))
    t2 = (AU_KM - d1) / v1
    hours = (t1 + t2) / 3600.0
    return np.where(np.isfinite(u) & (u > 0), hours, np.nan)


def add_arrival(df):
    """Estimated 1 AU arrival for halo and partial-halo CMEs (NaT for the rest)."""
    df = df.copy()
    candidate = df["halo"] | df["partial_halo"]
    hours = np.where(candidate, arrival_hours(df["linear_speed"]), np.nan)
    df["transit_hours"] = hours
    df["est_arrival"] = df["time"] + pd.to_timedelta(hours, unit="h")
    return df


def fetch_month(month, raw_dir, refresh=False):
    """Text of one month's file, from the cache unless refresh; None if not published."""
    path = Path(raw_dir) / FILE_PATTERN.format(month=month)
    if path.exists() and not refresh:
        return path.read_text(encoding="latin-1")
    try:
        text = http_get_bytes(f"{BASE_URL}/{FILE_PATTERN.format(month=month)}").decode("latin-1")
    except HTTPNotFound:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")
    return text


def run(start="2025-01", end=None, out=OUT_DIR, refresh_last=3):
    """Pull months [start, end] (default: up to this month), write cmes.csv and a manifest.

    The newest `refresh_last` months are re-downloaded, since the catalogue keeps
    filling them in.
    """
    end = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    months = pd.period_range(start, end, freq="M").to_timestamp()
    out = Path(out)
    parts, missing = [], []
    for i, month in enumerate(months):
        refresh = i >= len(months) - refresh_last
        text = fetch_month(month, out / "raw", refresh=refresh)
        if text is None:
            missing.append(f"{month:%Y-%m}")
            continue
        parts.append(parse_month(text))
    cmes = add_arrival(pd.concat(parts, ignore_index=True)) if parts else add_arrival(
        parse_month(""))
    cmes = cmes.sort_values("time").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    cmes.to_csv(out / "cmes.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    covered = [f"{m:%Y-%m}" for m in months if f"{m:%Y-%m}" not in missing]
    manifest = {
        "source": f"{BASE_URL}/univYYYY_MM.txt",
        "pulled_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "months": covered,
        "months_not_published": missing,
        "cmes": len(cmes),
        "halo": int(cmes["halo"].sum()),
        "partial_halo": int(cmes["partial_halo"].sum()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"{len(cmes)} CMEs from {len(covered)} months ({manifest['halo']} halo, "
        f"{manifest['partial_halo']} partial halo); not yet published: {missing or 'none'}")
    return cmes


def load(path=OUT_DIR / "cmes.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m training.pull_lasco_cme`")
    return pd.read_csv(path, parse_dates=["time", "est_arrival"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2025-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default=None, help="last month, YYYY-MM (default: this month)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    run(args.start, args.end, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
