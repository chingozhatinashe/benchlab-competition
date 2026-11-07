"""Pull the GONG synoptic-map history and reduce each map to a PFSS ring.

    python -m training.pull_gong inventory                 # listings 2016-10 .. now (~10 min)
    python -m training.pull_gong rings --pilot 2024-05     # one month, timed, not saved
    python -m training.pull_gong rings [--workers 2] [--max-seconds 3000]   # resumable

Inventory: every day's listing (connectors.gong) -> src/data/gong/inventory/YYYY-MM.csv with
name, observed, modified (UTC), rotation, longitude. `inventory --report` prints the posting
delay (modified - observed) per year and the coverage of the 6-hourly t0s.

Rings: for each 12-hourly t0 (SOLVE_HOURS; the 06Z and 18Z t0s reuse it), the newest map
posted by t0 - gong.MARGIN and observed within gong.MAX_AGE (connectors.gong.usable), reduced
by feature_engineering.pfss.reduce. A map that fails is replaced by the next older usable one
(up to 3). Each map is solved once, however many t0s use it. Output (dev-only), one file per
month: src/data/features/pfss/YYYY-MM.npz with
    t0       int64 ns (UTC)
    taken    int64 ns (UTC), the map's observation time, -1 where none was usable
    b0       float32 (n,)
    ring     float32 (n, len(RING_VARS), 360), NaN where none was usable
    scalars  float32 (n, len(SCALARS))
The downloaded maps are kept in src/data/gong/raw/YYYYMM/ so other PFSS settings need no
second download.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from connectors import gong
from feature_engineering import pfss
from training.datasets import EVAL_END
from training.pull_features import AIA_START, OUT, ROOT, t0_grid

GONG_DIR = ROOT / "gong"
INVENTORY_DIR = GONG_DIR / "inventory"
RAW_DIR = GONG_DIR / "raw"
RING_DIR = OUT / "pfss"
START = AIA_START.floor("D") - pd.Timedelta(days=2)
TRIES = 3
SOLVE_HOURS = 12  # one map per 12 h; the 06Z and 18Z t0s reuse the previous one (pilot: 6-hourly
# solves would take ~15.7 h with 2 workers)


def log(msg):
    print(msg, flush=True)


# ---- inventory ---------------------------------------------------------------------------

def pull_inventory(start=START, end=None, workers=8):
    """Day listings into monthly CSVs. Complete months are skipped; the latest is redone."""
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.now(tz="UTC").floor("D")
    days = pd.date_range(start.floor("D"), end, freq="D")
    months = sorted({f"{d:%Y-%m}" for d in days})
    connector = gong.GONGConnector(timeout=30, retries=3, swpc=False)
    for key in months:
        path = INVENTORY_DIR / f"{key}.csv"
        if path.exists() and key != months[-1]:
            continue
        batch = [d for d in days if f"{d:%Y-%m}" == key]
        t = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            frames = list(pool.map(connector.listing, batch))
        frames = [f for f in frames if not f.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=gong.COLUMNS)
        df.to_csv(path, index=False)
        log(f"inventory {key}: {len(df)} maps, {time.perf_counter() - t:.0f} s")


def load_inventory(directory=INVENTORY_DIR):
    parts = [pd.read_csv(p) for p in sorted(directory.glob("*.csv"))]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise FileNotFoundError(f"no GONG inventory in {directory}; run training.pull_gong")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("name")
    for col in ("observed", "modified"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values("observed").reset_index(drop=True)


def choose(inventory, t0s, tries=TRIES):
    """{t0: [names newest first]}, the usable maps for each t0 (connectors.gong.usable)."""
    inv = inventory.sort_values("modified").reset_index(drop=True)
    posted = pd.DatetimeIndex(inv["modified"] + pd.Timedelta(minutes=1)).asi8
    out = {}
    for t0 in t0s:
        n = np.searchsorted(posted, (t0 - gong.MARGIN).value, side="right")
        recent = inv.iloc[:n]
        recent = recent[recent["observed"] >= t0 - gong.MARGIN - gong.MAX_AGE]
        out[t0] = list(recent.sort_values("observed", ascending=False)["name"][:tries])
    return out


def report(inventory):
    delay = (inventory["modified"] - inventory["observed"]).dt.total_seconds() / 60
    year = inventory["observed"].dt.year
    q = delay.groupby(year).quantile([0.5, 0.9, 0.99, 1.0]).unstack()
    q.columns = ["p50", "p90", "p99", "max"]
    q["maps"] = year.value_counts().sort_index()
    q["late_1d"] = (delay > 1440).groupby(year).sum()
    log("posting delay (min) by year:\n" + q.round(0).to_string())
    end = min(pd.Timestamp.now(tz="UTC").floor("D"), EVAL_END)
    t0s = t0_grid(AIA_START, end)
    chosen = choose(inventory, t0s, tries=1)
    have = pd.Series({t: bool(v) for t, v in chosen.items()})
    log("t0s with a usable map, by year:\n"
        + have.groupby(have.index.year).mean().round(3).to_string())


# ---- rings -------------------------------------------------------------------------------

def raw_path(name):
    return RAW_DIR / f"20{name[5:9]}" / name


def _reduce_name(args):
    """Worker: download (or read the cached) map and reduce it. Returns (name, Ring | None)."""
    name, observed = args
    path = raw_path(name)
    try:
        if path.exists():
            data = path.read_bytes()
        else:
            data = gong.GONGConnector(timeout=30, retries=3).download(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
    except Exception:
        return name, None
    return name, pfss.reduce(data, pd.Timestamp(observed))


def solve_time(t0, hours=SOLVE_HOURS):
    """The t0 whose map a history t0 uses: t0 floored to the solve cadence.

    Maps chosen for an earlier time were posted by then, so reusing one never uses data
    posted after t0; it is at most `hours` - 6 h older than the map a live run would pick.
    """
    return pd.Timestamp(t0).floor(f"{hours}h")


def pull_month(t0s, inventory, pool, hours=SOLVE_HOURS):
    """(t0, taken, b0, ring, scalars) arrays for one month's t0s."""
    chosen = choose(inventory, sorted({solve_time(t, hours) for t in t0s}))
    observed = dict(zip(inventory["name"], inventory["observed"], strict=True))
    rings = {}
    pending = {t0: list(names) for t0, names in chosen.items()}
    for _ in range(TRIES):
        wanted = sorted({names[0] for names in pending.values() if names} - set(rings))
        for name, ring in pool.map(_reduce_name, [(n, observed[n]) for n in wanted]):
            rings[name] = ring
        for names in pending.values():
            while names and names[0] in rings and rings[names[0]] is None:
                names.pop(0)
        if all(not names or names[0] in rings for names in pending.values()):
            break
    n = len(t0s)
    taken = np.full(n, -1, dtype=np.int64)
    b0 = np.full(n, np.nan, dtype=np.float32)
    ring = np.full((n, *pfss.ring_shape()), np.nan, dtype=np.float32)
    scalars = np.full((n, len(pfss.SCALARS)), np.nan, dtype=np.float32)
    for i, t0 in enumerate(t0s):
        names = pending[solve_time(t0, hours)]
        r = rings.get(names[0]) if names else None
        if r is not None:
            taken[i] = pd.Timestamp(r.taken).value
            b0[i], ring[i], scalars[i] = r.b0, r.ring, r.scalars
    return pd.DatetimeIndex(t0s).asi8, taken, b0, ring, scalars, len(rings)


def pull_rings(max_seconds=None, pilot=None, workers=2):
    started = time.perf_counter()
    RING_DIR.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory()
    end = min(pd.Timestamp.now(tz="UTC").floor("D"), EVAL_END)
    t0s = t0_grid(AIA_START, end)
    months = sorted({f"{t:%Y-%m}" for t in t0s})
    todo = [pilot] if pilot else months
    with ProcessPoolExecutor(workers) as pool:
        for key in todo:
            path = RING_DIR / f"{key}.npz"
            if path.exists() and not pilot:
                continue
            batch = [t for t in t0s if f"{t:%Y-%m}" == key]
            t = time.perf_counter()
            t0, taken, b0, ring, scalars, solved = pull_month(batch, inventory, pool)
            secs = time.perf_counter() - t
            found = int((taken >= 0).sum())
            if pilot:
                log(f"pilot {key}: {found}/{len(batch)} t0s with a ring, {solved} maps solved, "
                    f"{secs:.0f} s with {workers} workers -> {len(months)} months "
                    f"~{len(months) * secs / 3600:.1f} h")
                return t0, taken, b0, ring, scalars
            np.savez_compressed(path, t0=t0, taken=taken, b0=b0, ring=ring, scalars=scalars)
            log(f"{key}: {found}/{len(batch)} t0s with a ring, {solved} maps ({secs:.0f} s)")
            if max_seconds and time.perf_counter() - started > max_seconds:
                log("time limit reached; run again to continue")
                return False
    return True


def load_rings(directory=RING_DIR):
    """(t0 DatetimeIndex, taken DatetimeIndex with NaT, b0, ring, scalars) over all months."""
    parts = [np.load(p) for p in sorted(directory.glob("*.npz"))]
    if not parts:
        raise FileNotFoundError(f"no PFSS rings in {directory}; run training.pull_gong rings")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    order = np.argsort(cat["t0"], kind="stable")
    taken = cat["taken"][order]
    taken_idx = pd.DatetimeIndex(np.where(taken < 0, np.datetime64("NaT"),
                                          taken.astype("datetime64[ns]")), tz="UTC")
    return (pd.DatetimeIndex(cat["t0"][order], tz="UTC"), taken_idx, cat["b0"][order],
            cat["ring"][order], cat["scalars"][order])


def load_ring_table(directory=RING_DIR):
    """The pulled rings as a pfss.RingTable (for build.Inputs(pfss=...))."""
    t0, taken, _, ring, scalars = load_rings(directory)
    keep = ~t0.duplicated(keep="last")
    return pfss.RingTable(t0[keep], taken[keep], ring[keep], scalars[keep])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["inventory", "rings"])
    parser.add_argument("--report", action="store_true", help="inventory: delays and coverage")
    parser.add_argument("--pilot", default=None, help="rings: one month, YYYY-MM, not saved")
    parser.add_argument("--workers", type=int, default=2,
                        help="processes; each peaks at about 650 MB")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    if args.step == "inventory":
        if not args.report:
            pull_inventory()
        report(load_inventory())
    else:
        pull_rings(args.max_seconds, args.pilot, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
