"""Daily staging evaluation: score every upload candidate with the official metric.

    PYTHONPATH=src python ci/staging/evaluate.py --records records [--no-fetch] [--alerts a.json]

Reads the run records on the records branch (runs/**.json, written by events.py), refreshes
the official truth (SWPC's hourly ACE files, training.competition_score.MONTHLY_URL; the
current and previous month are fetched again, older months are kept in truth/), and
writes:
    reports/YYYY-MM-DD.md and README.md   the report (README is the latest)
    scores/YYYY-MM-DD.json                every score with its lead curves
    events/YYYY-MM.csv                    missed cycles and a stale schedule, appended
Scoring follows the organisers' spec (training.competition_score): per candidate and
variant, the runs whose 72 h window has passed. A run that failed, timed out or gave invalid
output counts as persistence, as Benchlab scores it. A cycle the staging schedule missed
is not scored for anyone. Persistence and 27-day recurrence are scored on the same t0s.
The "shared t0s" table repeats the scores on the cycles every current candidate ran, so
the numbers are comparable. The report only shows scores and health: choosing a candidate
and uploading stay with the user (the "Upload to Benchlab" workflow on main).
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from training import competition_score as cs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events as ev  # noqa: E402

CYCLE = pd.Timedelta(hours=6)
WINDOW = pd.Timedelta(hours=cs.HORIZON)
TRUTH_LAG = pd.Timedelta(hours=2)  # the hourly file is issued ~10 min after each hour
STALE = pd.Timedelta(hours=12)  # no cycle for this long: the schedule has stopped
GRACE = pd.Timedelta(hours=3)  # a cycle this recent may still be running (cron delays)
RECENT = pd.Timedelta(days=7)
FMT = "%Y%m%dT%H%M%SZ"


# ---------------------------------------------------------------------- data
def load_runs(records):
    rows = []
    for path in sorted(Path(records).glob("runs/*/*/*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        r["t0"] = pd.Timestamp(datetime.strptime(r["t0"], FMT).replace(tzinfo=timezone.utc))
        rows.append(r)
    return pd.DataFrame(rows)


def load_events(records):
    parts = [pd.read_csv(p, dtype=str, keep_default_na=False)
             for p in sorted(Path(records).glob("events/*.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ev.FIELDS)


def months_between(start, end):
    return pd.date_range(start.tz_convert(None).to_period("M").to_timestamp(),
                         end.tz_convert(None), freq="MS")


def refresh_truth(records, start, now, fetch=True):
    """The official hourly speed from truth/*.txt, fetching what is missing or recent."""
    folder = Path(records) / "truth"
    folder.mkdir(parents=True, exist_ok=True)
    months = months_between(start, now)
    for i, month in enumerate(months):
        path = folder / f"{month:%Y%m}_ace_swepam_1h.txt"
        if fetch and (not path.exists() or i >= len(months) - 2):
            try:
                with urllib.request.urlopen(cs.MONTHLY_URL.format(month=month), timeout=60) as r:
                    path.write_bytes(r.read())
            except Exception as e:  # keep the cached copy; the gap shows in the scores
                print(f"truth {month:%Y-%m}: {e!r}", file=sys.stderr)
    parts = [cs.parse_monthly(p.read_text(encoding="ascii", errors="replace"))
             for p in sorted(folder.glob("*_ace_swepam_1h.txt"))]
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------- scores
def forecast_matrix(runs):
    """(t0s, runs x 72) with NaN for a run without a valid forecast."""
    runs = runs.sort_values("t0")
    f = np.full((len(runs), cs.HORIZON), np.nan)
    for i, fc in enumerate(runs["forecast"]):
        if isinstance(fc, list) and len(fc) == cs.HORIZON:
            f[i] = fc
    return pd.DatetimeIndex(runs["t0"]), f


def score_set(t0s, forecasts, truth):
    obs, pers = cs.verifying(truth, t0s), cs.persistence(truth, t0s)
    out = {"model": cs.score(forecasts, obs, pers),
           "persistence": cs.score(np.repeat(pers[:, None], cs.HORIZON, 1), obs, pers),
           "recurrence 27 d": cs.score(cs.recurrence(truth, t0s), obs, pers)}
    return out


def scores(runs, truth, now, shared_only=False):
    """{(candidate, variant): {"model", "persistence", "recurrence 27 d", "runs", ...}}."""
    done = runs[runs["t0"] + WINDOW <= now - TRUTH_LAG]
    if shared_only and not done.empty:
        current = set(done.loc[done["t0"] == done["t0"].max(), "candidate"])
        per_t0 = done.groupby("t0")["candidate"].apply(set)
        keep = per_t0[per_t0.apply(lambda c: current <= c)].index
        done = done[done["t0"].isin(keep) & done["candidate"].isin(current)]
    out = {}
    for (cand, variant), g in done.groupby(["candidate", "variant"]):
        t0s, f = forecast_matrix(g)
        res = score_set(t0s, f, truth)
        res["runs"] = len(g)
        res["failed runs"] = int((~g["valid"].astype(bool)).sum())
        res["first t0"], res["last t0"] = f"{t0s[0]:%Y-%m-%d %HZ}", f"{t0s[-1]:%Y-%m-%d %HZ}"
        out[(cand, variant)] = res
    return out


# ---------------------------------------------------------------------- health
def health(runs, since):
    sel = runs[runs["t0"] >= since] if since is not None else runs
    out = {}
    for (cand, variant), g in sel.groupby(["candidate", "variant"]):
        runtime = pd.to_numeric(g["runtime_s"], errors="coerce")
        obs = pd.to_numeric(g["regressors_observed"], errors="coerce")
        tot = pd.to_numeric(g["regressors_total"], errors="coerce")
        bad_sources = {}
        for s in g["sources"]:
            for k, v in (s or {}).items():
                if v != "ok":
                    bad_sources[k] = bad_sources.get(k, 0) + 1
        hosts = sorted({h for hs in g["undeclared_hosts"] for h in (hs or [])})
        out[(cand, variant)] = {
            "runs": len(g), "failed": int((~g["valid"].astype(bool)).sum()),
            "fell back": int(g["fell_back"].astype(bool).sum()),
            "runtime p50 (s)": float(runtime.median()) if runtime.notna().any() else None,
            "runtime max (s)": float(runtime.max()) if runtime.notna().any() else None,
            "start delay p50 (min)": float(pd.to_numeric(g["start_delay_s"], errors="coerce")
                                           .median() / 60) if g["start_delay_s"].notna().any()
            else None,
            "regressors observed": (f"{(obs / tot).mean():.1%}" if tot.notna().any() else "—"),
            "sources not ok": ", ".join(f"{k} ×{n}" for k, n in sorted(bad_sources.items()))
            or "—",
            "undeclared hosts": ", ".join(hosts) or "—"}
    return out


def missed_cycles(events_df, now):
    """Scheduled cycles (every 6 h since the first one) with no "cycle" event."""
    ran = pd.to_datetime(events_df.loc[events_df["kind"] == "cycle", "t0"], format=FMT,
                         utc=True)
    if ran.empty:
        return [], None
    expected = pd.date_range(ran.min(), (now - GRACE).floor("6h"), freq=CYCLE)
    return sorted(set(expected) - set(ran)), ran.max()


# ---------------------------------------------------------------------- report
def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("—" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def f1(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.1f}"


def score_rows(res):
    rows = []
    for (cand, variant), r in sorted(res.items()):
        m, p = r["model"], r["persistence"]
        ratio = m["s_final"] / p["s_final"] if p["s_final"] else np.nan
        rows.append([cand, variant, r["runs"], r["failed runs"], f1(m["s_final"]),
                     "—" if not np.isfinite(ratio) else f"{ratio:.3f}", f1(m["rmse"]),
                     f1(m["trmse"]), f1(m["bias"]), f1(m["s_49_72"]),
                     f1(p["s_final"]), f1(r["recurrence 27 d"]["s_final"])])
    return rows


SCORE_HEADER = ["candidate", "variant", "runs", "failed (→ persistence)", "S_final",
                "/ persistence", "RMSE", "TRMSE", "bias", "S_49:72", "persistence S",
                "recurrence S"]


def report(today, now, res, shared, h7, h_all, missed, last_cycle, recent_events, truth):
    lines = [f"# Staging report {today}", "",
             f"Generated {now:%Y-%m-%d %H:%M} UTC. Every `upload_candidate/*` tag runs every 6 h "
             "as Benchlab runs it (`.github/workflows/staging.yml`); scores use the official "
             "metric (`training.competition_score`; S in km/s, lower is better). This report "
             "only shows scores and health: what ships is chosen by hand and uploaded by the "
             "\"Upload to Benchlab\" workflow on main.", "",
             f"Official truth up to {truth.dropna().index.max():%Y-%m-%d %H:00} UTC"
             if truth.notna().any() else "No official truth yet.", "",
             "## Scores (runs whose 72 h have passed)", ""]
    lines += [table(SCORE_HEADER, score_rows(res)) if res else "No completed runs yet.", ""]
    lines += ["## Scores on the t0s every current candidate ran", ""]
    lines += [table(SCORE_HEADER, score_rows(shared)) if shared else "—", ""]
    for title, h in (("## Health, last 7 days", h7), ("## Health, all time", h_all)):
        lines += [title, ""]
        if h:
            cols = list(next(iter(h.values())).keys())
            lines += [table(["candidate", "variant", *cols],
                            [[c, v, *r.values()] for (c, v), r in sorted(h.items())]), ""]
        else:
            lines += ["No runs.", ""]
    lines += ["## Schedule", "",
              f"Last cycle: {last_cycle:%Y-%m-%d %H:00} UTC" if last_cycle is not None
              else "No cycle has run yet.",
              f"Missed cycles: {len(missed)}"
              + (" (" + ", ".join(f"{t:%m-%d %HZ}" for t in missed[-10:]) + ")"
                 if missed else ""), ""]
    alerts = recent_events[recent_events["severity"] == "alert"]
    info = recent_events[recent_events["severity"] != "alert"]
    lines += ["## Alerts, last 7 days", ""]
    if len(alerts):
        lines += [table(["t0", "candidate", "variant", "kind", "detail"],
                        alerts[["t0", "candidate", "variant", "kind", "detail"]]
                        .values.tolist()[-50:]), ""]
    else:
        lines += ["None.", ""]
    counts = info.groupby("kind").size().sort_values(ascending=False)
    lines += ["## Other events, last 7 days (counts; details in events/)", "",
              ", ".join(f"{k}: {n}" for k, n in counts.items()) or "None.", ""]
    return "\n".join(lines)


def jsonable(res):
    return {f"{c} | {v}": {k: (x if not isinstance(x, float) or np.isfinite(x) else None)
                           for k, x in r.items()} for (c, v), r in res.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--records", required=True)
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--now", default=None, help="YYYYMMDDTHHMMSSZ (tests)")
    p.add_argument("--alerts", default=None)
    args = p.parse_args(argv)
    now = (pd.Timestamp(datetime.strptime(args.now, FMT).replace(tzinfo=timezone.utc))
           if args.now else pd.Timestamp.now(tz="UTC"))
    records = Path(args.records)
    runs = load_runs(records)
    events_df = load_events(records)
    start = (runs["t0"].min() if not runs.empty else now) - pd.Timedelta(days=30)
    truth = refresh_truth(records, start, now, fetch=not args.no_fetch)
    res = scores(runs, truth, now) if not runs.empty else {}
    shared = scores(runs, truth, now, shared_only=True) if not runs.empty else {}
    h7 = health(runs, now - RECENT) if not runs.empty else {}
    h_all = health(runs, None) if not runs.empty else {}
    missed, last_cycle = missed_cycles(events_df, now)

    # new events: missed cycles not logged yet, and a stopped schedule
    logged = set(events_df.loc[events_df["kind"] == "missed cycle", "t0"])
    rows, alerts = [], {}
    for t in missed:
        if f"{t:{FMT}}" not in logged:
            rows.append({"time": ev.now(), "t0": f"{t:{FMT}}", "candidate": "", "sha": "",
                         "variant": "", "kind": "missed cycle", "severity": "alert",
                         "detail": "no staging cycle ran for this t0"})
            alerts.setdefault("schedule", []).append(f"- `{t:{FMT}}`: missed cycle")
    stale_detail = f"last cycle {last_cycle:%Y-%m-%d %H:00} UTC" if last_cycle is not None else ""
    stale_logged = ((events_df["kind"] == "schedule stale")
                    & (events_df["detail"] == stale_detail)).any()
    if last_cycle is not None and now - last_cycle > STALE and not stale_logged:
        rows.append({"time": ev.now(), "t0": f"{now.floor('6h'):{FMT}}", "candidate": "",
                     "sha": "", "variant": "", "kind": "schedule stale", "severity": "alert",
                     "detail": stale_detail})
        alerts.setdefault("schedule", []).append(
            f"- no cycle since {last_cycle:%Y-%m-%d %H:00} UTC (the schedule may have stopped)")
    if rows:
        ev.append_events(records, rows)
        events_df = load_events(records)
    recent = events_df[(pd.to_datetime(events_df["t0"], format=FMT, utc=True, errors="coerce")
                        >= now - RECENT) & ~events_df["kind"].isin(["run ok", "cycle"])]

    today = f"{now:%Y-%m-%d}"
    text = report(today, now, res, shared, h7, h_all, missed, last_cycle, recent, truth)
    (records / "reports").mkdir(exist_ok=True)
    (records / "reports" / f"{today}.md").write_text(text + "\n", encoding="utf-8")
    (records / "README.md").write_text(text + "\n", encoding="utf-8")
    (records / "scores").mkdir(exist_ok=True)
    (records / "scores" / f"{today}.json").write_text(json.dumps(
        {"generated": f"{now:%Y-%m-%dT%H:%M:%SZ}", "all": jsonable(res),
         "shared": jsonable(shared)}, indent=1, default=str) + "\n", encoding="utf-8")
    if args.alerts:
        Path(args.alerts).write_text(json.dumps(alerts, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
