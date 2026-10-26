"""Staging events and alerts (stdlib only).

    python ci/staging/events.py cycle --records records --runs new_runs --t0 20261111T060000Z \
        --candidates '[{"name": "hybrid-v1", "sha": "abc"}]' --alerts alerts.json

`cycle` files the new run records under the records branch
(runs/YYYY-MM/<t0>/<candidate>__<variant>.json), appends every event to
events/YYYY-MM.csv (time, t0, candidate, sha, variant, kind, severity, detail) and writes
the alerts (severity "alert") grouped by candidate to --alerts, for the workflow to report in
the run (report_alerts.sh). Staging only records: it never ranks candidates or uploads anything.

Event kinds (rules in `run_events`):
    alert   build failed, failed (exit code), timeout, invalid output, over budget
            (runtime + estimated pull > 900 s), slow (> SLOW_S), image too large,
            fell back, few regressors (< MIN_REGRESSOR_SHARE observed),
            undeclared host
    info    missing regressors, source not ok, DNS not captured, run ok,
            candidate added / removed, cycle
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SLOW_S = 600
MAX_IMAGE_MB = 15360
MIN_REGRESSOR_SHARE = 0.75
FIELDS = ("time", "t0", "candidate", "sha", "variant", "kind", "severity", "detail")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_events(rec):
    """[(kind, severity, detail)] for one run record."""
    ev = []

    def add(kind, severity, detail=""):
        ev.append((kind, severity, detail))

    if rec.get("build_failed"):
        add("build failed", "alert", "the image did not build; scored as persistence")
        return ev
    if rec.get("timed_out"):
        add("timeout", "alert", f"killed after {rec.get('runtime_s')} s; scored as persistence")
    elif rec.get("exit_code") not in (0, None):
        add("failed", "alert", f"exit code {rec.get('exit_code')}; scored as persistence")
    if not rec.get("valid") and not rec.get("timed_out") and rec.get("exit_code") in (0, None):
        add("invalid output", "alert", f"{rec.get('invalid_reason')}; scored as persistence")
    if rec.get("over_budget"):
        add("over budget", "alert", f"run {rec.get('runtime_s')} s + estimated pull "
                                    f"{rec.get('est_pull_s')} s > 900 s")
    elif (rec.get("runtime_s") or 0) > SLOW_S:
        add("slow", "alert", f"run {rec.get('runtime_s')} s > {SLOW_S} s")
    if (rec.get("image_mb") or 0) > MAX_IMAGE_MB:
        add("image too large", "alert", f"{rec.get('image_mb')} MB > {MAX_IMAGE_MB} MB")
    if rec.get("fell_back"):
        add("fell back", "alert", f"model used: {rec.get('model_used')}")
    obs, total = rec.get("regressors_observed"), rec.get("regressors_total")
    if obs is not None and total:
        if obs < MIN_REGRESSOR_SHARE * total:
            add("few regressors", "alert", f"{obs}/{total} observed; missing "
                                           f"{rec.get('missing_regressors')}")
        elif obs < total:
            add("missing regressors", "info", f"{obs}/{total}: {rec.get('missing_regressors')}")
    for name, status in (rec.get("sources") or {}).items():
        if status != "ok":
            add("source not ok", "info", f"{name}: {status}")
    if rec.get("undeclared_hosts"):
        add("undeclared host", "alert", ", ".join(rec["undeclared_hosts"]))
    if not rec.get("dns_captured"):
        add("dns not captured", "info", "declared-domain check skipped")
    if not ev or all(s == "info" for _, s, _ in ev):
        add("run ok", "info", f"{rec.get('model_used')}, {rec.get('runtime_s')} s, "
                              f"{obs}/{total} regressors")
    return ev


def candidate_events(previous, current):
    """Candidates added or removed since the last cycle: [(name, sha, kind)]."""
    prev = {c["name"]: c["sha"] for c in previous}
    cur = {c["name"]: c["sha"] for c in current}
    out = [(n, s, "candidate added") for n, s in cur.items() if n not in prev]
    out += [(n, s, "candidate moved") for n, s in cur.items() if n in prev and prev[n] != s]
    out += [(n, s, "candidate removed") for n, s in prev.items() if n not in cur]
    return out


def append_events(records, rows):
    by_month = {}
    for r in rows:
        by_month.setdefault(r["t0"][:4] + "-" + r["t0"][4:6], []).append(r)
    for month, part in by_month.items():
        path = Path(records) / "events" / f"{month}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            if new:
                w.writeheader()
            w.writerows(part)


def cycle(records, runs, t0, candidates):
    """File the run records, append events; returns {candidate: [alert lines]}."""
    records, stamp = Path(records), now()
    rows, alerts = [], {}
    state = records / "candidates.json"
    previous = json.loads(state.read_text(encoding="utf-8")) if state.exists() else []
    for name, sha, kind in candidate_events(previous, candidates):
        rows.append({"time": stamp, "t0": t0, "candidate": name, "sha": sha, "variant": "",
                     "kind": kind, "severity": "info", "detail": ""})
    state.write_text(json.dumps(candidates, indent=1) + "\n", encoding="utf-8")
    month = f"{t0[:4]}-{t0[4:6]}"
    filed = 0
    for path in sorted(Path(runs).rglob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("schema") is None or rec.get("t0") != t0:
            continue
        dest = records / "runs" / month / t0 / f"{rec['candidate']}__{rec['variant']}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        filed += 1
        for kind, severity, detail in run_events(rec):
            rows.append({"time": stamp, "t0": t0, "candidate": rec["candidate"],
                         "sha": rec["sha"], "variant": rec["variant"], "kind": kind,
                         "severity": severity, "detail": detail})
            if severity == "alert":
                alerts.setdefault(rec["candidate"], []).append(
                    f"- `{t0}` **{rec['variant']}** ({rec['sha'][:7]}): {kind}: {detail}")
    expected = {c["name"] for c in candidates}
    seen = {json.loads(p.read_text(encoding="utf-8"))["candidate"]
            for p in (records / "runs" / month / t0).glob("*.json")} if filed else set()
    for name in sorted(expected - seen):
        rows.append({"time": stamp, "t0": t0, "candidate": name, "sha": "", "variant": "",
                     "kind": "no record", "severity": "alert",
                     "detail": "the run job left no record"})
        alerts.setdefault(name, []).append(f"- `{t0}`: no record (the run job failed early)")
    rows.append({"time": stamp, "t0": t0, "candidate": "", "sha": "", "variant": "",
                 "kind": "cycle", "severity": "info",
                 "detail": f"{len(candidates)} candidates, {filed} runs recorded"})
    append_events(records, rows)
    return alerts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cycle")
    c.add_argument("--records", required=True)
    c.add_argument("--runs", required=True)
    c.add_argument("--t0", required=True)
    c.add_argument("--candidates", required=True, help="JSON list of {name, sha}")
    c.add_argument("--alerts", required=True)
    args = p.parse_args(argv)
    alerts = cycle(args.records, args.runs, args.t0, json.loads(args.candidates))
    Path(args.alerts).write_text(json.dumps(alerts, indent=1) + "\n", encoding="utf-8")
    print(f"cycle {args.t0}: alerts for {sorted(alerts) or 'no candidate'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
