"""Pull the regressor sources' history and build the training feature table.

    python -m training.pull_features all           # every step below, in order
    python -m training.pull_features insitu        # ACE SWEPAM + MAG + EPAM -> hourly frame, shocks
    python -m training.pull_features aia --max-seconds 1800   # resumable
    python -m training.pull_features flares       # GOES-16/18 flare summaries (NCEI)
    python -m training.pull_features f107          # OMNI daily F10.7 (CDAWeb)
    python -m training.pull_features table         # features for every 6-hourly t0 + climatology

Everything goes through the same code as a live run (connectors.*,
feature_engineering.*), on the same sources where they keep history:
- ACE lists from the SOHO mirror, cached per day in src/data/ace_{rtsw,mag,epam}/raw/.
- AIA 193 browse images, one per 6-hourly t0 cut-off, cached in src/data/aia193/;
  their coronal-hole metrics in src/data/features/coronal_holes.csv.
- Flares: the live feed keeps 7 days, so history comes from NCEI's GOES-R XRS
  science flare summaries (GOES-16 to 2025-04, GOES-18 from 2022-06), peaks merged.
- F10.7: the live feed keeps ~40 days, so history comes from OMNI (CDAWeb).

Output in src/data/features/: insitu_hourly.csv.gz, shocks.csv, coronal_holes.csv,
flares.csv, f107_daily.csv, table.csv.gz (raw regressors, NaN where unknown) and
manifest.json. The climatology medians for imputation are fitted on the training
years only and written to src/assets/feature_climatology.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from connectors.ace_archive import ACEDailyArchive
from connectors.base import http_get_bytes
from connectors.sdo import SDOConnector
from feature_engineering import build, coronal_holes, impute, insitu
from training.datasets import DATA_DIR, DATA_START, EVAL_END, TRAIN_END

ROOT = DATA_DIR.parent
OUT = ROOT / "features"
CACHE = {"swepam": DATA_DIR / "raw", "mag": ROOT / "ace_mag" / "raw",
         "epam": ROOT / "ace_epam" / "raw"}
AIA_DIR = ROOT / "aia193"
FLARE_DIR = ROOT / "goes_flares"
NCEI = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
FLSUM = "{sat}/l2/data/xrsf-l2-flsum_science/sci_xrsf-l2-flsum_{name}_v2-2-2.nc"
FLARE_FILES = {
    "g16": FLSUM.format(sat="goes16", name="g16_s20170209_e20250406"),
    "g18": FLSUM.format(sat="goes18", name="g18_s20220617_e20261020"),
}
FLARE_MERGE = pd.Timedelta(minutes=5)
STEP = pd.Timedelta(hours=6)
HISTORY_START = DATA_START - build.HISTORY - pd.Timedelta(days=7)  # room for the lags
AIA_START = DATA_START - pd.Timedelta(days=7)
F107_START = HISTORY_START - pd.Timedelta(days=30)
CHUNK = pd.Timedelta(days=30)
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def log(msg):
    print(f"[pull_features] {msg}", file=sys.stderr, flush=True)


def t0_grid(start=DATA_START, end=EVAL_END):
    """Every 6-hourly forecast time in [start, end), like the live schedule."""
    return pd.date_range(start, end, freq=STEP, inclusive="left", name="t0")


# ---------------------------------------------------------------------- in-situ
def pull_insitu(start=HISTORY_START, end=EVAL_END, out=OUT):
    """ACE minutes -> the shifted hourly frame and the shock list, 30 days at a time."""
    archives = {p: ACEDailyArchive(p, budget=3600, cache_dir=CACHE[p]) for p in CACHE}
    pad = pd.Timedelta(days=1)
    frames, shocks = [], []
    for a in pd.date_range(start, end, freq=CHUNK):
        b = min(a + CHUNK, end)
        got = {p: arc.minutes(a - pad, b + pad) for p, arc in archives.items()}
        minutes = insitu.minute_frame(got["swepam"], got["mag"], got["epam"])
        frames.append(insitu.hourly_frame(minutes, a, b))
        found = insitu.detect_shocks(minutes)
        shocks.append(found[(found >= a) & (found < b)])
        log(f"{a:%Y-%m-%d}: {frames[-1]['bt'].notna().mean():.0%} field hours, "
            f"{len(shocks[-1])} shocks")
    hourly = pd.concat(frames)
    shock_index = pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in shocks]), name="shock")
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out / "insitu_hourly.csv.gz", date_format=TIME_FORMAT)
    shock_index.to_frame(index=False).to_csv(out / "shocks.csv", index=False,
                                             date_format=TIME_FORMAT)
    return hourly, shock_index


def load_hourly(out=OUT):
    df = pd.read_csv(out / "insitu_hourly.csv.gz", index_col="Time")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="Time")
    return df


def load_shocks(out=OUT):
    df = pd.read_csv(out / "shocks.csv")
    return pd.DatetimeIndex(pd.to_datetime(df["shock"], utc=True), name="shock")


# ---------------------------------------------------------------------- coronal holes
def _accept(jpeg, taken):
    return coronal_holes.usable(coronal_holes.grayscale(jpeg), taken)


def pull_aia(start=AIA_START, end=EVAL_END, out=OUT, max_seconds=None):
    """Coronal-hole metrics for each 6-hourly cut-off (t0 - 1 h). Resumable."""
    path = out / "coronal_holes.csv"
    done = load_coronal_holes(out) if path.exists() else None
    cutoffs = t0_grid(start, end) - build.CH_LATENCY
    todo = cutoffs if done is None else cutoffs[~cutoffs.isin(done.index)]
    log(f"AIA: {len(cutoffs) - len(todo)} cut-offs done, {len(todo)} to go")
    sdo = SDOConnector(budget=1800, workers=8)
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    for month in sorted({c.strftime("%Y-%m") for c in todo}):
        batch = [c for c in todo if c.strftime("%Y-%m") == month]
        images = sdo.latest(batch, accept=_accept)
        rows = []
        for cut, (taken, jpeg) in images.items():
            if taken is not None:
                dest = AIA_DIR / f"{taken:%Y}" / f"{taken:%Y%m%d_%H%M%S}_512_0193.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(jpeg)
                m = coronal_holes.metrics(jpeg, taken)
            else:
                m = dict.fromkeys(coronal_holes.NAMES, np.nan)
            rows.append({"cutoff": cut, "taken": taken, **m})
        frame = pd.DataFrame(rows)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False,
                     date_format=TIME_FORMAT)
        log(f"AIA {month}: {frame['taken'].notna().sum()}/{len(frame)} usable images")
        if max_seconds and time.monotonic() - started > max_seconds:
            log("AIA: time limit reached; run again to continue")
            return False
    return True


def load_coronal_holes(out=OUT):
    df = pd.read_csv(out / "coronal_holes.csv")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("cutoff"), utc=True), name="cutoff")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[list(coronal_holes.NAMES)]


# ---------------------------------------------------------------------- flares
def read_flsum(path):
    """Flare peaks from a GOES-R XRS flare-summary netCDF: peak_time | peak_flux | flare_class."""
    import h5py  # dev-only dependency

    with h5py.File(path, "r") as f:
        epoch = pd.Timestamp("2000-01-01T12:00Z")
        times = epoch + pd.to_timedelta(f["time"][:], unit="s")
        status = np.array([s.decode() for s in f["status"][:]])
        cls = np.array([s.decode() for s in f["flare_class"][:]])
        flux = f["xrsb_flux"][:].astype(float)
    peak = status == "EVENT_PEAK"
    return pd.DataFrame({"peak_time": times[peak], "peak_flux": flux[peak],
                         "flare_class": cls[peak]})


def merge_flares(frames):
    """One row per flare: peaks within FLARE_MERGE of each other are the same flare."""
    df = pd.concat(frames, ignore_index=True).dropna(subset=["peak_time", "peak_flux"])
    df = df.sort_values("peak_time").reset_index(drop=True)
    group = (df["peak_time"].diff() > FLARE_MERGE).cumsum()
    best = df.groupby(group)["peak_flux"].idxmax()
    return df.loc[best].reset_index(drop=True)


def pull_flares(out=OUT):
    frames = []
    for sat, rel in FLARE_FILES.items():
        path = FLARE_DIR / Path(rel).name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(http_get_bytes(f"{NCEI}/{rel}", timeout=300))
        frames.append(read_flsum(path))
        log(f"flares {sat}: {len(frames[-1])} peaks")
    flares = merge_flares(frames)
    out.mkdir(parents=True, exist_ok=True)
    flares.to_csv(out / "flares.csv", index=False, date_format=TIME_FORMAT)
    log(f"flares: {len(flares)} after merging")
    return flares


def load_flares(out=OUT):
    df = pd.read_csv(out / "flares.csv")
    df["peak_time"] = pd.to_datetime(df["peak_time"], utc=True)
    return df


# ---------------------------------------------------------------------- F10.7
def pull_f107(start=F107_START, end=EVAL_END, out=OUT):
    from training.pull_omni import pull_variable

    hourly = pull_variable("F10_INDEX", start, end)
    ok = hourly[hourly["status"] == "ok"]
    daily = ok.groupby(ok["Time"].dt.floor("D"))["value"].mean().rename("f107")
    daily.index.name = "day"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "f107_daily.csv", date_format=TIME_FORMAT)
    log(f"F10.7: {len(daily)} days, last {daily.index.max():%Y-%m-%d}")
    return daily


def load_f107(out=OUT):
    df = pd.read_csv(out / "f107_daily.csv")
    return pd.Series(df["f107"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(pd.to_datetime(df["day"], utc=True), name="day"),
                     name="f107")


# ---------------------------------------------------------------------- table
def build_features_table(out=OUT, asset=impute.ASSET):
    inputs = build.Inputs(hourly=load_hourly(out), shocks=load_shocks(out),
                          coronal_holes=load_coronal_holes(out), flares=load_flares(out),
                          f107=load_f107(out))
    t0s = t0_grid()
    table = build.build_table(inputs, t0s)
    table.to_csv(out / "table.csv.gz", date_format=TIME_FORMAT, float_format="%.6g")
    train = table[table.index < TRAIN_END]
    medians = impute.fit_climatology(train)
    impute.save_climatology(medians, {
        "fitted_on": [DATA_START.strftime(TIME_FORMAT), TRAIN_END.strftime(TIME_FORMAT)],
        "rows": len(train),
    }, asset)
    coverage = table.notna().mean().round(3)
    manifest = {
        "built_at": pd.Timestamp.now(tz="UTC").strftime(TIME_FORMAT),
        "t0s": len(table),
        "first_t0": t0s[0].strftime(TIME_FORMAT),
        "last_t0": t0s[-1].strftime(TIME_FORMAT),
        "coverage": coverage.to_dict(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"table: {len(table)} t0s; lowest coverage "
        f"{coverage.idxmin()} {coverage.min():.0%}")
    return table


def load_table(out=OUT):
    df = pd.read_csv(out / "table.csv.gz", index_col="t0")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True), name="t0")
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["all", "insitu", "aia", "flares", "f107", "table"])
    parser.add_argument("--max-seconds", type=float, default=None, help="AIA step time limit")
    args = parser.parse_args(argv)
    if args.step in ("all", "insitu"):
        pull_insitu()
    if args.step in ("all", "flares"):
        pull_flares()
    if args.step in ("all", "f107"):
        pull_f107()
    if args.step in ("all", "aia") and not pull_aia(max_seconds=args.max_seconds):
        return 0
    if args.step in ("all", "table"):
        build_features_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
