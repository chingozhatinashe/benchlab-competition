"""One staging run -> one JSON record (stdlib only; runs on the bare GitHub runner).

    python ci/staging/record_run.py --t0 20261111T060000Z --candidate hybrid-v1 --sha abc123 \
        --variant default --log run.log --forecast out/forecast.json --exit-code 0 \
        --runtime 21.4 --build-seconds 70 --image-mb 520 --dns dns.txt \
        --submission candidate/src/submission.json --started 2026-11-11T06:09:12Z --out rec.json

The record (SCHEMA) holds what the platform would see (exit code, runtime, a valid 72-value
forecast) and what the model's log says about its inputs:
    model_used    from the last "done" line: "xgboost done" -> hybrid, "linear model done"
                  -> linear, "[inference] done" -> baseline; "using the fallback forecast"
                  -> persistence
    fell_back     any "falling back" / "fallback forecast" line
    regressors    "N/M regressors observed" of the top-level model, and the missing names
    sources       the "[fetch_inputs] ... {source: status}" line, with MB and seconds
    hosts         DNS names the container queried (tcpdump), and those not in the
                  candidate's declared_domains (a subdomain of a declared domain counts)
A build failure is recorded with build_failed true and no forecast. A run that failed,
timed out or gave invalid output is scored as persistence (the organisers' rule).
"""

import argparse
import ast
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1
HORIZON = 72
VALID_RANGE = (0.0, 10000.0)  # the platform's check (local_tests/test_local.sh)
BUDGET_S = 900  # 15 min including the image pull
PULL_MB_PER_S = 100  # test_local.sh's estimate of the pull speed
LOG_KEEP = 20000  # characters of the log kept in the record

_REGRESSORS = re.compile(r"\[inference\] (xgboost|linear model): (\d+)/(\d+) regressors observed;"
                         r" (?:missing|imputed): (.*?)(?:; missing optional (\[.*\]))?$")
_FETCH = re.compile(r"\[fetch_inputs\] inputs in ([\d.]+) s, ([\d.]+) MB: (\{.*\})")
_DNS = re.compile(r"\b(?:A|AAAA|HTTPS)\? ([A-Za-z0-9.-]+?)\.? ")


def model_used(log):
    """Which model produced the forecast, from the log's last completion line."""
    used = None
    for line in log.splitlines():
        if "using the fallback forecast" in line:
            used = "persistence"
        elif "[inference] xgboost done" in line:
            used = "hybrid"
        elif "[inference] linear model done" in line:
            used = "linear"
        elif "[inference] done in" in line and used != "persistence":
            used = "baseline"
    return used


def fell_back(log):
    return any(("falling back" in line or "using the fallback forecast" in line)
               for line in log.splitlines())


def regressors(log):
    """(observed, total, missing names) of the first model that reported its regressors."""
    for line in log.splitlines():
        m = _REGRESSORS.search(line.strip())
        if m:
            missing = []
            for part in (m.group(4), m.group(5)):
                if part and part != "none":
                    try:
                        missing += list(ast.literal_eval(part))
                    except (ValueError, SyntaxError):
                        missing.append(part)
            return int(m.group(2)), int(m.group(3)), missing
    return None, None, []


def sources(log):
    """({source: status}, MB, seconds) from the fetch summary line, or ({}, None, None)."""
    for line in log.splitlines():
        m = _FETCH.search(line)
        if m:
            try:
                return ast.literal_eval(m.group(3)), float(m.group(2)), float(m.group(1))
            except (ValueError, SyntaxError):
                break
    return {}, None, None


def dns_hosts(text):
    return sorted({h.lower().rstrip(".") for h in _DNS.findall(text or "")})


def undeclared(hosts, declared):
    declared = [d.lower().rstrip(".") for d in declared]
    return [h for h in hosts
            if not any(h == d or h.endswith("." + d) for d in declared)]


def check_forecast(path):
    """(forecast list or None, reason it is invalid or None)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"no readable forecast ({type(e).__name__})"
    values = data.get("forecast_speed_kms") if isinstance(data, dict) else None
    if not isinstance(values, list) or len(values) != HORIZON:
        return None, "not 72 values"
    out = []
    for v in values:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return None, "non-numeric or non-finite value"
        if not VALID_RANGE[0] <= v <= VALID_RANGE[1]:
            return None, "value outside 0-10000 km/s"
        out.append(float(v))
    return out, None


def t0_time(t0):
    return datetime.strptime(t0, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def build_record(args):
    log = Path(args.log).read_text(encoding="utf-8", errors="replace") if args.log and Path(
        args.log).exists() else ""
    declared = []
    if args.submission and Path(args.submission).exists():
        declared = json.loads(Path(args.submission).read_text(encoding="utf-8")).get(
            "declared_domains", [])
    dns_text = Path(args.dns).read_text(encoding="utf-8", errors="replace") if args.dns and Path(
        args.dns).exists() else None
    hosts = dns_hosts(dns_text)
    forecast, reason = (None, "build failed") if args.build_failed else check_forecast(
        args.forecast)
    timed_out = args.exit_code == 124  # coreutils timeout
    if args.exit_code not in (None, 0) and forecast is not None:
        reason = f"exit code {args.exit_code}"  # the platform needs both
        forecast = None
    observed, total, missing = regressors(log)
    srcs, mb, secs = sources(log)
    started = datetime.strptime(args.started, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc) if args.started else None
    est_pull = round(args.image_mb / PULL_MB_PER_S, 1) if args.image_mb is not None else None
    return {
        "schema": SCHEMA, "t0": args.t0, "candidate": args.candidate, "sha": args.sha,
        "variant": args.variant,
        "started_at": args.started,
        "start_delay_s": (round((started - t0_time(args.t0)).total_seconds())
                          if started else None),
        "build_failed": bool(args.build_failed), "build_s": args.build_seconds,
        "image_mb": args.image_mb, "est_pull_s": est_pull,
        "runtime_s": args.runtime, "exit_code": args.exit_code, "timed_out": timed_out,
        "over_budget": (args.runtime is not None and est_pull is not None
                        and args.runtime + est_pull > BUDGET_S),
        "valid": forecast is not None, "invalid_reason": reason,
        "model_used": model_used(log), "fell_back": fell_back(log),
        "regressors_observed": observed, "regressors_total": total,
        "missing_regressors": missing,
        "sources": srcs, "fetch_mb": mb, "fetch_s": secs,
        "dns_captured": dns_text is not None and bool(dns_text.strip()),
        "hosts": hosts, "undeclared_hosts": undeclared(hosts, declared),
        "forecast": forecast,
        "log": log[-LOG_KEEP:],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--t0", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--log")
    p.add_argument("--forecast")
    p.add_argument("--exit-code", type=int)
    p.add_argument("--runtime", type=float)
    p.add_argument("--build-seconds", type=float)
    p.add_argument("--build-failed", action="store_true")
    p.add_argument("--image-mb", type=float)
    p.add_argument("--dns")
    p.add_argument("--submission")
    p.add_argument("--started")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    rec = build_record(args)
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"{args.candidate}/{args.variant}: valid={rec['valid']} model={rec['model_used']} "
          f"fell_back={rec['fell_back']} runtime={rec['runtime_s']} "
          f"regressors={rec['regressors_observed']}/{rec['regressors_total']} "
          f"undeclared={rec['undeclared_hosts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
