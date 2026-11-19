"""Benchlab live monitor: the submitted model's runs and logs -> the records branch (stdlib only).

    python ci/monitor/live.py --portal portal.json --records records --alerts alerts.json

Reads the portal's response (fetched by .github/workflows/monitor.yml with the
BENCHLAB_PORTAL secret) and writes, under the staging records branch:
    live/status.json              the submission's status fields (no links, no credentials)
    live/runs/YYYY-MM.jsonl       one line per finished live run (RUN_FIELDS only: no task
                                  ARNs, network ids or IPs)
    live/logs/YYYY-MM/*.log       each run log, scrubbed (links and IP addresses removed);
                                  gzip logs are unpacked, and a log that isn't text is not stored
    live/state.json               what has been seen (log keys as sha256, never the links)
    runs/YYYY-MM/<t0>/benchlab__live.json   (benchlab__dryrun.json off the 6-hourly marks:
                                  the qualification run after an upload)
                                  a run record in staging's format (ci/staging/record_run.py)
                                  for each finished run whose log carries the forecast
                                  ("[run] forecast {...}", src/run.sh since 942a451) or that
                                  failed, so the daily staging evaluation (evaluate.py) scores
                                  it next to the candidates; a failed run counts as persistence
    events/YYYY-MM.csv            events and alerts (staging's rules, ci/staging/events.py,
                                  plus the live ones below)
Live alerts: not ELIGIBLE, a missed cycle (no run for a 6-hourly t0), a failed run, a log
missing LOG_GRACE after the run stopped, a forecast missing from a successful run's log,
and a different model or a forecast far (> MAX_DIFF_KMS) from staging's run of the
submitted commit (candidate "submitted") at the same t0.

The portal's response holds presigned links and upload credentials, and this repository is
public: links are only used to download, in memory, and never printed or written. Errors are
reported by type and HTTP status only. Read-only towards Benchlab: nothing is uploaded.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "staging"))
import events as ev  # noqa: E402
import record_run as rr  # noqa: E402

CANDIDATE, VARIANT = "benchlab", "live"
DRY_RUN = "dryrun"  # a run off the 6-hourly marks: the qualification run after an upload
STAGING_CANDIDATE = "submitted"  # upload_candidate/submitted: the uploaded commit
STATUS_FIELDS = ("submission_id", "eligibility", "conformance", "conformance_at",
                 "failure_reason", "phase", "accepting", "cpu_requested", "cpu_granted",
                 "memory_requested", "memory_granted", "resources_snapped")
RUN_FIELDS = ("submission_id", "cycle_ts", "status", "exit_code", "failure_reason",
              "stop_code", "stopped_reason", "output_bytes", "pull_seconds", "run_seconds",
              "started_at", "stopped_at", "launch_retries", "relaunched_after")
CYCLE = timedelta(hours=6)
CYCLE_GRACE = timedelta(hours=2)  # a cycle this recent may still be running
LOG_GRACE = timedelta(hours=3)  # a finished run's log should be listed by then
MAX_LOG_BYTES = 2_000_000
MAX_DIFF_KMS = 25.0  # live vs staging: inputs can differ a little (data arriving late)
BUDGET_S = 900
GZIP_MAGIC = bytes([0x1F, 0x8B])
REPLACEMENT = chr(0xFFFD)  # what a bad UTF-8 byte decodes to
FMT = "%Y%m%dT%H%M%SZ"

_T0_LINE = re.compile(r"\[run\] t0=(\d{8}T\d{6}Z) submission=(\S+)")
_FORECAST = re.compile(r"\[run\] forecast (\{.*\})")
_LINK = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.I)
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_AMZ = re.compile(r"(X-Amz-[A-Za-z-]+|AWSAccessKeyId|Signature|x-amz-security-token)=\S+", re.I)


# ---------------------------------------------------------------------- helpers
def scrub(text):
    """The log without links, IP addresses or signed-query parameters."""
    text = _LINK.sub("<link>", text)
    text = _AMZ.sub(r"\1=<removed>", text)
    return _IP.sub("<ip>", text)


def variant(t0):
    """"live" for a 6-hourly cycle, DRY_RUN otherwise (scored apart from the cycles)."""
    t = rr.t0_time(t0)
    return VARIANT if t.hour % 6 == 0 and t.minute == 0 else DRY_RUN


def log_text(body):
    """The log as scrubbed text: gzip is unpacked; anything that isn't text is refused."""
    if body[:2] == GZIP_MAGIC:
        body = gzip.GzipFile(fileobj=io.BytesIO(body)).read(MAX_LOG_BYTES * 20 + 1)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("utf-8", errors="replace")
        if text.count(REPLACEMENT) > max(10, len(text) // 1000):
            raise ValueError("not text") from None
    return scrub(text)


def key_id(key):
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]


def to_time(value):
    """A UTC datetime from the portal's timestamps (t0 form, ISO 8601 or epoch), or None."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    try:
        return datetime.strptime(s, FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        x = float(s)
        return datetime.fromtimestamp(x / 1000 if x > 1e11 else x, timezone.utc)
    except ValueError:
        pass
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)
    except ValueError:
        return None


def to_t0(value):
    t = to_time(value)
    return f"{t:{FMT}}" if t else None


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def failed(run):
    """Why the platform's run failed, or None."""
    code = str(run.get("exit_code") or "").strip()
    if run.get("failure_reason"):
        return str(run["failure_reason"])
    if code not in ("", "0", "0.0", "None"):
        return f"exit code {code}"
    if num(run.get("output_bytes")) == 0:
        return "no output"
    return None


def timed_out(run):
    total = (num(run.get("run_seconds")) or 0) + (num(run.get("pull_seconds")) or 0)
    reason = f"{run.get('stopped_reason') or ''} {run.get('failure_reason') or ''}".lower()
    return "timeout" in reason or "timed out" in reason or total >= BUDGET_S


def http_get(url, timeout=60):
    """The body at a presigned link. Errors name the type and status, never the link."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read(MAX_LOG_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}") from None
    except Exception as e:  # noqa: BLE001 - message may hold the link
        raise RuntimeError(type(e).__name__) from None


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- the record
def run_record(t0, run, log, staging):
    """A staging-format run record for one live run (rr.build_record's fields)."""
    forecast, reason = None, failed(run)
    m = _FORECAST.search(log)
    if reason is None:
        if m is None:
            reason = "no forecast in the log"
        else:
            try:
                forecast, reason = rr.check_values(json.loads(m.group(1)))
            except ValueError:
                reason = "unreadable forecast line"
    started = to_time(run.get("started_at"))
    runtime, pull = num(run.get("run_seconds")), num(run.get("pull_seconds"))
    observed, total, missing = rr.regressors(log)
    srcs, mb, secs = rr.sources(log)
    code = num(run.get("exit_code"))
    rec = {
        "schema": rr.SCHEMA, "t0": t0, "candidate": CANDIDATE,
        "sha": str(run.get("submission_id") or ""), "variant": variant(t0),
        "started_at": f"{started:%Y-%m-%dT%H:%M:%SZ}" if started else None,
        "start_delay_s": (round((started - rr.t0_time(t0)).total_seconds())
                          if started else None),
        "build_failed": False, "build_s": None, "image_mb": None, "est_pull_s": pull,
        "runtime_s": runtime, "exit_code": int(code) if code is not None else None,
        "timed_out": timed_out(run),
        "over_budget": (runtime or 0) + (pull or 0) > BUDGET_S,
        "valid": forecast is not None, "invalid_reason": reason,
        "model_used": rr.model_used(log), "fell_back": rr.fell_back(log),
        "regressors_observed": observed, "regressors_total": total,
        "missing_regressors": missing, "sources": srcs, "fetch_mb": mb, "fetch_s": secs,
        "dns_captured": None, "hosts": [], "undeclared_hosts": [],
        "forecast": forecast, "log": log[-rr.LOG_KEEP:],
        "platform": {k: run.get(k) for k in RUN_FIELDS},
    }
    if staging and forecast and staging.get("forecast"):
        diff = max(abs(a - b) for a, b in zip(forecast, staging["forecast"], strict=True))
        rec["staging"] = {"model_used": staging.get("model_used"), "max_abs_diff_kms":
                          round(diff, 2)}
    return rec


def live_events(rec):
    """Staging's events for the record, plus the comparison with staging."""
    out = [e for e in ev.run_events(rec) if e[0] != "dns not captured"]
    st = rec.get("staging")
    if st:
        if st["model_used"] != rec["model_used"]:
            out.append(("differs from staging", "alert",
                        f"model {rec['model_used']} live, {st['model_used']} in staging"))
        elif st["max_abs_diff_kms"] > MAX_DIFF_KMS:
            out.append(("differs from staging", "alert",
                        f"max |live - staging| {st['max_abs_diff_kms']} km/s"))
        else:
            out.append(("vs staging", "info", f"max |live - staging| "
                                              f"{st['max_abs_diff_kms']} km/s"))
    return out


# ---------------------------------------------------------------------- main step
def monitor(portal, records, now=None, fetch=http_get):
    """Files the portal's news under `records`; returns ({group: [alert lines]}, summary)."""
    records = Path(records)
    now = now or datetime.now(timezone.utc)
    live = records / "live"
    state = load_json(live / "state.json", {})
    state.setdefault("logs", {})      # key_id -> {"t0", "submission", "path"} or {"error"}
    state.setdefault("runs", [])      # run ids filed to live/runs
    state.setdefault("closed", [])    # t0|submission handled (recorded or reported)
    state.setdefault("missed", [])    # t0s reported as missed
    rows, alerts = [], {}
    stamp = ev.now()

    def event(t0, kind, severity, detail, sub="", var=None):
        rows.append({"time": stamp, "t0": t0, "candidate": CANDIDATE, "sha": sub,
                     "variant": variant(t0) if var is None else var, "kind": kind,
                     "severity": severity,
                     "detail": detail})
        if severity == "alert":
            alerts.setdefault("benchlab (live)", []).append(f"- `{t0}`: {kind}: {detail}")

    # status
    status = {k: portal.get(k) for k in STATUS_FIELDS}
    previous = load_json(live / "status.json", {})
    write_json(live / "status.json", {**status, "read_at": f"{now:%Y-%m-%dT%H:%M:%SZ}"})
    if status.get("eligibility") != previous.get("eligibility") or \
            status.get("submission_id") != previous.get("submission_id"):
        sev = "info" if status.get("eligibility") == "ELIGIBLE" else "alert"
        event(f"{now:{FMT}}", "eligibility", sev,
              f"{status.get('submission_id')}: {status.get('eligibility')}"
              + (f" ({status['failure_reason']})" if status.get("failure_reason") else ""),
              str(status.get("submission_id") or ""), var="")

    # finished runs -> live/runs
    runs = []
    for r in portal.get("recent_runs") or []:
        t0 = to_t0(r.get("cycle_ts"))
        if t0 is None:
            continue
        runs.append((t0, {k: r.get(k) for k in RUN_FIELDS}))
    if portal.get("recent_runs") and not runs:
        event(f"{now:{FMT}}", "unreadable runs", "alert",
              "no run's cycle_ts could be read as a time (ci/monitor/live.py to_time)",
              var="")
    for t0, r in runs:
        rid = f"{r['submission_id']}|{r['cycle_ts']}|{r['started_at']}"
        if not r.get("stopped_at") or rid in state["runs"]:
            continue
        path = live / "runs" / f"{t0[:4]}-{t0[4:6]}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"t0": t0, **r}, sort_keys=True) + "\n")
        state["runs"].append(rid)

    # new logs -> live/logs (scrubbed)
    for item in portal.get("logs") or []:
        kid = key_id(item.get("key"))
        if kid in state["logs"] or not item.get("url"):
            continue
        if (num(item.get("size")) or 0) > MAX_LOG_BYTES:
            state["logs"][kid] = {"error": f"too large ({item.get('size')} bytes)"}
            continue
        try:
            body = fetch(item["url"])
        except RuntimeError as e:
            print(f"log {kid}: download failed ({e}); retried next time", file=sys.stderr)
            continue
        try:
            text = log_text(body)
        except (OSError, EOFError, ValueError) as e:  # bad gzip, or not text: not stored
            state["logs"][kid] = {"error": f"unreadable ({type(e).__name__})"}
            continue
        m = _T0_LINE.search(text)
        t0, sub = (m.group(1), m.group(2)) if m else (None, None)
        name = f"{t0}__{sub}__{kid}.log" if t0 else f"other__{kid}.log"
        month = f"{t0[:4]}-{t0[4:6]}" if t0 else "other"
        path = live / "logs" / month / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        state["logs"][kid] = {"t0": t0, "submission": sub,
                              "path": str(path.relative_to(records)).replace("\\", "/")}

    # run records: the latest attempt of each finished run
    latest = {}
    for t0, r in runs:
        key = f"{t0}|{r['submission_id']}"
        if r.get("stopped_at") and (key not in latest or str(r["started_at"]) >
                                    str(latest[key][1]["started_at"])):
            latest[key] = (t0, r)
    for key, (t0, r) in sorted(latest.items()):
        if key in state["closed"]:
            continue
        sub = str(r["submission_id"])
        logs = [records / v["path"] for v in state["logs"].values()
                if v.get("t0") == t0 and v.get("submission") == sub]
        texts = [p.read_text(encoding="utf-8") for p in logs if p.exists()]
        log = next((t for t in texts if _FORECAST.search(t)), texts[0] if texts else "")
        stopped = to_time(r.get("stopped_at"))
        if not texts and failed(r) is None:
            if stopped and now - stopped > LOG_GRACE:
                event(t0, "no log", "alert", f"no run log {LOG_GRACE} after the run", sub)
                state["closed"].append(key)
            continue  # wait for the log
        if failed(r) is None and not _FORECAST.search(log):
            # a log without the forecast line: a version before run.sh echoed it
            event(t0, "no forecast in log", "info", "not scored (log only)", sub)
            state["closed"].append(key)
            continue
        month = f"{t0[:4]}-{t0[4:6]}"
        staging = load_json(records / "runs" / month / t0 / f"{STAGING_CANDIDATE}__default.json",
                            None)
        rec = run_record(t0, r, log, staging)
        write_json(records / "runs" / month / t0 / f"{CANDIDATE}__{variant(t0)}.json", rec)
        for kind, severity, detail in live_events(rec):
            event(t0, kind, severity, detail, sub)
        state["closed"].append(key)

    # missed cycles: a 6-hourly t0 inside the portal's window with no run at all
    t0s = sorted({rr.t0_time(t0) for t0, _ in runs if variant(t0) == VARIANT})
    if t0s:
        t = t0s[0]
        while t <= now - CYCLE_GRACE:
            t0 = f"{t:{FMT}}"
            if t not in t0s and t0 not in state["missed"]:
                event(t0, "missed cycle", "alert", "no Benchlab run for this t0")
                state["missed"].append(t0)
            t += CYCLE

    if rows:
        ev.append_events(records, rows)
    write_json(live / "state.json", state)
    summary = (f"{status.get('submission_id')}: {status.get('eligibility')}; "
               f"{len(runs)} runs listed, {len(state['logs'])} logs seen, "
               f"{sum(1 for r in rows if r['kind'] in ('run ok', 'vs staging'))} ok, "
               f"{sum(len(v) for v in alerts.values())} alerts")
    return alerts, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--portal", required=True, help="the portal's JSON response (a file)")
    p.add_argument("--records", required=True)
    p.add_argument("--alerts", required=True)
    args = p.parse_args(argv)
    portal = json.loads(Path(args.portal).read_text(encoding="utf-8"))
    alerts, summary = monitor(portal, args.records)
    Path(args.alerts).write_text(json.dumps(alerts, indent=1) + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
